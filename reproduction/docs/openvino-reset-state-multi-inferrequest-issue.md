# [Draft GitHub issue — to be filed upstream at openvinotoolkit/openvino]

This bug was documented in an earlier draft of the paper (former §7.2, "An undocumented
OpenVINO constraint") and moved here so it can be filed as an upstream issue instead.
It is referred to as **Discovery #21** elsewhere in this reproduction package.

---

## Title

`[Bug] reset_state() on one InferRequest corrupts shape inference for sibling InferRequests created from the same CompiledModel (stateful model, GPU plugin)`

## System information

- OpenVINO: 2026.1.0 and 2026.2 (both affected)
- Devices: Intel Arc B390 iGPU (Panther Lake, Core Ultra X7 358H), Intel Arc 140V iGPU
  (Lunar Lake, Core Ultra 7 258V) — GPU plugin
- OS: Windows 11
- Python: 3.11
- Models: stateful LLM IRs containing ReadValue/Assign KV-cache state pairs (observed
  across Llama 3.1 8B INT4, Llama 3.2 1B INT4, and Gemma 4 E2B FP32 exports, both
  single-graph and per-stage pipeline shards; the Llama graphs additionally carry the
  `Gather(ReadValue, beam_idx, axis=0)` cache-reorder pattern that
  `optimum-intel`'s `fuse_cache_reorder` produces)

## Summary

Per the Model API documentation, multiple `InferRequest`s created from one
`CompiledModel` should each maintain fully independent state. In practice, calling
`reset_state()` on **one** `InferRequest` among several created from the **same**
`CompiledModel` causes subsequent `infer()` calls on **any sibling request** to fail
shape validation.

## Steps to reproduce

```python
import numpy as np
import openvino as ov

core = ov.Core()
compiled = core.compile_model("llama-3.1-8b-int4-stateful.xml", "GPU")

req_a = compiled.create_infer_request()
req_b = compiled.create_infer_request()

def feed(req, ids, past_len):
    n = ids.shape[1]
    req.infer({
        "input_ids": ids,
        "attention_mask": np.ones((1, past_len + n), dtype=np.int64),
        "position_ids": np.arange(past_len, past_len + n, dtype=np.int64).reshape(1, -1),
        "beam_idx": np.zeros(1, dtype=np.int32),
    })

prompt = np.array([[1, 15043, 3186]], dtype=np.int64)

# 1. Run a full prefill + a few decode steps on req_a            -> OK
feed(req_a, prompt, 0)
feed(req_a, np.array([[42]], dtype=np.int64), prompt.shape[1])

# 2. Run req_b concurrently/interleaved                          -> OK
feed(req_b, prompt, 0)

# 3. Reset ONE of the requests
req_a.reset_state()

# 4. Infer again on either request                               -> FAILS
feed(req_a, prompt, 0)   # or feed(req_b, ...) — siblings also fail
```

The failure does not appear on the first prefill; it reliably occurs after at least one
full run-and-reset cycle.

## Observed error

```
Check 'TRShape::broadcast_merge_into(output_shape, input_shapes[1], autob)' failed at
  src/core/shape_inference/include/eltwise_shape_inference.hpp:28:
While validating node 'opset1::Add Add_NNN () -> ()':
Argument shapes are inconsistent.
```

The specific `Add_NNN` node varies between runs and models; consistently, it is an
`opset1::Add` whose broadcast merge resolves one argument shape to the empty tuple
`()`.

## Expected behavior

Each `InferRequest` maintains independent state; `reset_state()` on one request should
not affect shape inference on the request itself (after re-prefill) nor on sibling
requests created from the same `CompiledModel`.

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

A multi-stream coordinator that exercises this pattern (with the workaround applied) is
available in this repository:

- `reproduction/scripts/coord/mini_coord_spec_mbatch.py` (per-stream `compile_model`
  on both coordinator and worker)
- `reproduction/scripts/coord/gemma_2s_mbatch_coord.py`

## Pre-filing checklist (status as of 2026-06-11)

- [x] Latest released OpenVINO checked: **2026.2.0** (`pip index versions openvino`,
  2026-06-11) — the bug was observed on both 2026.1.0 and 2026.2.0, so the affected
  list already includes the latest release.
- [x] Prior-art search (2026-06-11): `gh search issues --repo=openvinotoolkit/openvino`
  for `reset_state InferRequest`, `reset_state stateful`, `broadcast_merge_into Add`,
  and `multiple InferRequest state` — no existing report found.
- [ ] Before filing, re-confirm on whatever OpenVINO release is current at filing time
  (re-run the repro above on an Arc iGPU machine) and re-search the tracker.
