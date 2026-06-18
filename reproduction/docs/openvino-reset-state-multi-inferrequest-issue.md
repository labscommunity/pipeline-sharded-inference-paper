# [Draft GitHub issue — to be filed upstream at openvinotoolkit/openvino]

This bug was documented in an earlier draft of the paper (former §7.2, "An undocumented
OpenVINO constraint") and moved here so it can be filed as an upstream issue instead.
It is referred to as **Discovery #21** elsewhere in this reproduction package.

> **2026-06-17 update — corrected reproduction.** The earlier "minimal repro" in this
> file was **single-threaded and did not actually reproduce the bug**. On re-test it
> ran cleanly on OV 2026.1.0 / 2026.2.0 / 2026.2.1 across several models. The real
> trigger (recovered from the original `rainier` discovery, `DISCOVERIES.md #21`) is
> **concurrent, interleaved** driving of sibling `InferRequest`s sharing one
> `CompiledModel`. With a concurrent harness the bug reproduces reliably — and is
> **still present on the latest release, OV 2026.2.1** (see hit-rate below). A clean,
> self-contained repro on a public model is at
> `reproduction/scripts/repro_reset_state_concurrent.py`.

---

## Title

`[Bug] Concurrent reset_state() across sibling InferRequests of one CompiledModel corrupts shape inference (stateful model, GPU plugin)`

## System information

- OpenVINO: **2026.1.0, 2026.2.0, and 2026.2.1 — all affected** (2026.2.1 is the latest
  release as of 2026-06-17)
- Devices: Intel Arc 140V iGPU (Lunar Lake, Core Ultra 7 258V) — GPU plugin; originally
  also observed on Intel Arc B390 iGPU (Panther Lake, Core Ultra X7 358H). Not observed
  on the CPU plugin.
- OS: Windows 11
- Python: 3.12
- Models: stateful LLM IRs containing ReadValue/Assign KV-cache state pairs. Reproduced
  on a **public, ungated** model — `EmbeddedLLM/Llama-3.2-1B-Instruct-int4-sym-ov`
  (Llama architecture, INT4, with the `Gather(ReadValue, beam_idx, axis=0)` cache-reorder
  pattern that `optimum-intel`'s `fuse_cache_reorder` produces). Originally observed on
  Llama 3.1 8B INT4 / Llama 3.2 1B INT4 / Gemma 4 E2B FP32, single-graph and per-stage
  pipeline shards.

## Summary

Per the Model API documentation, multiple `InferRequest`s created from one
`CompiledModel` should each maintain fully independent state. In practice, when two (or
more) sibling `InferRequest`s from the **same** `CompiledModel` are driven
**concurrently from separate threads** — each doing prefill → several decodes →
`reset_state()` — the shared shape-inference state is corrupted, and a subsequent
`infer()` on one of the requests fails validation with some node resolved to the empty
shape `() -> ()`.

**Important:** a single-threaded, sequential "reset one request, then infer a sibling"
sequence does **not** reproduce this. The trigger is the concurrency — one stream's
`reset_state()`/`infer()` landing while a sibling is mid-generation.

## Steps to reproduce

Self-contained script (downloads a public stateful Llama IR if none is given):
`reproduction/scripts/repro_reset_state_concurrent.py`. The essential structure:

```python
import threading, time, numpy as np, openvino as ov

core = ov.Core()
compiled = core.compile_model("openvino_model.xml", "GPU")   # ONE shared CompiledModel
reqs = [compiled.create_infer_request() for _ in range(2)]   # sibling requests

def feed(req, n, past):
    req.infer({
        "input_ids":      np.ones((1, n), dtype=np.int64),
        "attention_mask": np.ones((1, past + n), dtype=np.int64),
        "position_ids":   np.arange(past, past + n, dtype=np.int64).reshape(1, -1),
        "beam_idx":       np.zeros(1, dtype=np.int32),
    })

errors = []
def worker(sid):
    req = reqs[sid]
    time.sleep(sid * 0.13)               # stagger so the two streams interleave
    try:
        for _ in range(4):               # several run-and-reset cycles
            feed(req, 4, 0)              # prefill
            for d in range(8):           # decode steps
                feed(req, 1, 4 + d); time.sleep(0.002)
            req.reset_state()            # reset amid the sibling's activity
    except Exception as e:
        errors.append((sid, repr(e)))

ts = [threading.Thread(target=worker, args=(s,)) for s in range(2)]
[t.start() for t in ts]; [t.join() for t in ts]
assert not errors, errors              # one stream fails with the shape error below
```

Note: forcing the two prefills to run at *exactly* the same instant (e.g. a lock-step
barrier) instead OOMs the iGPU (`CL_OUT_OF_RESOURCES`); the staggered/interleaved timing
above surfaces the shape-inference corruption.

## Observed error

The failing node varies between runs and OV versions, but it is consistently a
shape-inference validation failure with one operand resolved to the empty tuple `()`:

OV 2026.1.0 (an `opset1::Add`):
```
Check 'TRShape::broadcast_merge_into(output_shape, input_shapes[1], autob)' failed at
  src/core/shape_inference/include/eltwise_shape_inference.hpp:28:
While validating node 'opset1::Add Add_6583 () -> ()':
Argument shapes are inconsistent.
```

OV 2026.2.1 (same corruption, surfacing on an `opset1::MatMul`):
```
Check 'DimType::merge(...) || arg0_col_dim.is_dynamic() || arg1_row_dim.is_dynamic()'
  failed at src/core/shape_inference/include/matmul_shape_inference.hpp:76:
While validating node 'opset1::MatMul MatMul_4850 () -> ()':
Incompatible MatMul matrix dimension. First input dimension=8192 ... doesn't match the
  second input dimension=2048 ...
```

A concurrent GPU buffer race (`"The allocated input/output memory is necessary to set
kernel arguments"`, `ocl_stream.cpp`) is occasionally seen instead, consistent with the
same root cause: sharing one compiled graph's state across concurrently-driven requests.

## Reproduction hit-rate

12 concurrent trials per version (fresh `CompiledModel` each trial),
`Llama-3.2-1B-Instruct int4`, Intel Arc 140V iGPU, GPU plugin:

| OpenVINO | shape-inference corruption | GPU buffer race | clean |
|----------|:--:|:--:|:--:|
| 2026.1.0 | 10 / 12 | 1 / 12 | 1 / 12 |
| 2026.2.0 | 12 / 12 | 0 | 0 |
| **2026.2.1 (latest)** | **12 / 12** | 0 | 0 |

## Expected behavior

Each `InferRequest` maintains independent state; `reset_state()` on one request should
not affect shape inference on the request itself nor on sibling requests created from
the same `CompiledModel`, even when the requests are driven concurrently.

## Workaround

Call `compile_model()` separately per stream so each `InferRequest` owns a fully
independent compiled graph. This restores the documented independence semantics at the
cost of extra compile time (seconds per additional stream on Arc iGPU) and extra GPU
memory (~2 GB per additional Llama-8B-class stage per stream).

## Impact

Blocks multi-user serving patterns that interleave several stateful generation streams
over one compiled model (one `InferRequest` per user). With the per-stream
`compile_model` workaround the pattern works correctly; cross-stream outputs are
byte-identical to single-stream runs.

## Reproduction context

- `reproduction/scripts/repro_reset_state_concurrent.py` — minimal standalone repro on a
  public model (this is what the hit-rate table above was produced with).
- `reproduction/scripts/coord/mini_coord_spec_mbatch.py`,
  `reproduction/scripts/coord/gemma_2s_mbatch_coord.py` — the multi-stream coordinators
  where the bug originally surfaced (with the per-stream `compile_model` workaround
  applied).

## Pre-filing checklist (status as of 2026-06-17)

- [x] Reproduced on the latest released OpenVINO **2026.2.1** — 12/12 concurrent trials
  on an Arc 140V iGPU (and 2026.2.0 12/12, 2026.1.0 10/12).
- [x] Reproduced on a **public, ungated** model
  (`EmbeddedLLM/Llama-3.2-1B-Instruct-int4-sym-ov`), so the report needs no proprietary
  artifact.
- [x] Corrected the reproduction: the original single-threaded minimal repro did not
  trigger the bug; the trigger is concurrent interleaved driving of sibling requests.
- [ ] Re-run the prior-art search on the tracker immediately before filing
  (`reset_state InferRequest`, `broadcast_merge_into`, `multiple InferRequest state`,
  `concurrent reset_state`).
