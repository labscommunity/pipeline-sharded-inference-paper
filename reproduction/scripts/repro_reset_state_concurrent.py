"""
Minimal reproduction of the OpenVINO reset_state() multi-InferRequest
shape-inference corruption (Discovery #21), on the GPU plugin / Intel Arc iGPU.

KEY POINT: this bug requires CONCURRENT, interleaved driving of two (or more)
InferRequests created from the SAME CompiledModel, each running
prefill -> several decodes -> reset_state(). A single-threaded, sequential
"reset one, then infer the sibling" sequence does NOT trigger it. The trigger is
the concurrency: while one stream is mid-generation, another stream's
reset_state()/infer() corrupts the shared shape-inference state, and a subsequent
infer() then fails validating some node to the empty shape `() -> ()`.

The exact failing node varies between runs and OV versions (e.g. an opset1::Add
broadcast_merge_into on 2026.1.0, an opset1::MatMul dimension merge on 2026.2.1) --
they are the same underlying corruption.

Reproduced (Intel Arc 140V iGPU, Lunar Lake, Windows 11, GPU plugin),
12 concurrent trials per version, public model Llama-3.2-1B-Instruct int4:
    OV 2026.1.0 : 10/12 shape-corruption  (+1 GPU buffer race, 1 clean)
    OV 2026.2.0 : 12/12 shape-corruption
    OV 2026.2.1 : 12/12 shape-corruption   <-- still present on latest

Usage:
    python repro_reset_state_concurrent.py [model.xml] [DEVICE] [STREAMS] [DECODES] [ROUNDS]

If no model is given it downloads a public, ungated stateful Llama IR
(EmbeddedLLM/Llama-3.2-1B-Instruct-int4-sym-ov) into ./repro_model/.

Exit: 0 = no reproduction ; 7 = reproduced/failed under concurrency
"""
import os, sys, threading, time, urllib.request
import numpy as np
import openvino as ov

HF_BASE = "https://huggingface.co/EmbeddedLLM/Llama-3.2-1B-Instruct-int4-sym-ov/resolve/main"
DEADLINE = 180


def log(*a): print(*a, flush=True)


def ensure_model(model_path):
    d = os.path.dirname(model_path) or "."
    os.makedirs(d, exist_ok=True)
    for fn in ("openvino_model.xml", "openvino_model.bin"):
        dst = os.path.join(d, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        log(f"downloading {fn} ...")
        urllib.request.urlretrieve(HF_BASE + "/" + fn, dst)


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("repro_model", "openvino_model.xml")
    device = sys.argv[2] if len(sys.argv) > 2 else "GPU"
    STREAMS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    DECODES = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    ROUNDS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
    SEQ = 4

    if not os.path.exists(model_path):
        ensure_model(model_path)

    log("openvino:", ov.__version__)
    core = ov.Core()
    log("devices:", core.available_devices, "| device:", device)
    log(f"config: STREAMS={STREAMS} DECODES={DECODES} ROUNDS={ROUNDS}")

    compiled = core.compile_model(model_path, device)   # ONE shared CompiledModel
    names = {p.get_any_name() for p in compiled.inputs}
    has_beam = "beam_idx" in names
    reqs = [compiled.create_infer_request() for _ in range(STREAMS)]

    def feed(req, n, past):
        d = {
            "input_ids": np.ones((1, n), dtype=np.int64),
            "attention_mask": np.ones((1, past + n), dtype=np.int64),
            "position_ids": np.arange(past, past + n, dtype=np.int64).reshape(1, -1),
        }
        if has_beam:
            d["beam_idx"] = np.zeros(1, dtype=np.int32)
        req.infer({k: v for k, v in d.items() if k in names})

    errors = []

    def worker(sid):
        # Staggered, free-running (NOT lock-step: simultaneous prefills merely OOM
        # the iGPU). The stagger + per-step yield interleaves one stream's
        # reset_state() with the other stream's infers -- that is the trigger.
        req = reqs[sid]
        r = -1
        try:
            time.sleep(sid * 0.13)
            for r in range(ROUNDS):
                feed(req, SEQ, 0)
                past = SEQ
                for _ in range(DECODES):
                    feed(req, 1, past); past += 1
                    time.sleep(0.002)
                req.reset_state()
        except Exception as e:
            errors.append((sid, r, f"{type(e).__name__}: {e}"))

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in range(STREAMS)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=DEADLINE)

    if any(t.is_alive() for t in threads):
        log("RESULT: HANG/DEADLOCK under concurrent reset_state")
        return 7
    if not errors:
        log("RESULT: NO REPRO (all streams completed without error)")
        return 0
    for sid, rnd, msg in errors:
        log(f"--- stream {sid} round {rnd} FAILED ---\n{msg[:900]}")
    log("RESULT: REPRODUCED -- shape-inference corruption under concurrent reset_state")
    return 7


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback; traceback.print_exc(); sys.exit(2)
