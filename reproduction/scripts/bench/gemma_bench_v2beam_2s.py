"""Bench Gemma 4 E2B 2-stage v2_beam (rotary-fixed + beam_idx Gather injected)
on alpha localhost GPU.

The v5_beam pattern unlocks OV's IndirectKVCache fusion for stateful shards.
Speed should be >= v2 baseline (12.32 tok/s) — fusion can be neutral or
positive. Output is byte-correct vs v2.
"""
import os, time, statistics, numpy as np
import openvino as ov

SHARD_DIR = r"C:\cascadia\shards_e2b_cached_2s_v2_beam"
DEVICE = "GPU"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 50
N_RUNS = 5
N_WARMUP = 2


def s0_to_s1(stage0_compiled, r0_outputs, pos):
    outs_by_port = {o: r0_outputs[o] for o in stage0_compiled.outputs}
    def find(name_match):
        for o in stage0_compiled.outputs:
            if any(name_match in n for n in o.get_names()):
                return outs_by_port[o]
        raise KeyError(name_match)
    return {
        "hidden_states": find("hidden_states"),
        "position_ids": pos,
        "external_kv.0.key":   find("cross_kv.0.key"),
        "external_kv.0.value": find("cross_kv.0.value"),
        "external_kv.1.key":   find("cross_kv.1.key"),
        "external_kv.1.value": find("cross_kv.1.value"),
    }


def decode_one(s0, s1, r0, r1, ids, max_tokens):
    r0.reset_state()
    r1.reset_state()
    ids_np = np.array([ids], dtype=np.int64)
    pos_np = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
    beam = np.zeros(1, dtype=np.int32)
    res0 = r0.infer({"input_ids": ids_np, "position_ids": pos_np, "beam_idx": beam})
    res1 = r1.infer(s0_to_s1(s0, res0, pos_np))
    logits = res1[s1.outputs[0]]
    nt = int(np.argmax(logits[0, -1, :]))
    gens = [nt]
    for step in range(max_tokens - 1):
        ids_one = np.array([[gens[-1]]], dtype=np.int64)
        pos_one = np.array([[len(ids) + step]], dtype=np.int64)
        r0_out = r0.infer({"input_ids": ids_one, "position_ids": pos_one, "beam_idx": beam})
        r1_out = r1.infer(s0_to_s1(s0, r0_out, pos_one))
        nt = int(np.argmax(r1_out[s1.outputs[0]][0, -1, :]))
        gens.append(nt)
    return gens


def main():
    print(f"OV {ov.__version__}", flush=True)
    print(f"Shards: {SHARD_DIR}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(SHARD_DIR, "tokenizer"))

    core = ov.Core()
    print("compiling stage_0 (v2_beam, default precision)...", flush=True)
    t0 = time.perf_counter()
    s0 = core.compile_model(
        os.path.join(SHARD_DIR, "stage_0/openvino_model.xml"), DEVICE)
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)
    print("compiling stage_1 (v2_beam = same as v2)...", flush=True)
    t0 = time.perf_counter()
    s1 = core.compile_model(os.path.join(SHARD_DIR, "stage_1/openvino_model.xml"), DEVICE)
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

    # Quick: list stage_0 inputs to confirm beam_idx is present
    print(f"stage_0 inputs: {[(list(i.get_names())[0], str(i.get_partial_shape())) for i in s0.inputs]}", flush=True)

    r0 = s0.create_infer_request()
    r1 = s1.create_infer_request()

    msgs = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    print(f"prompt: {len(ids)} tokens", flush=True)

    print(f"\nwarmup x{N_WARMUP}...", flush=True)
    for _ in range(N_WARMUP):
        decode_one(s0, s1, r0, r1, ids, MAX_TOKENS)

    print(f"timed x{N_RUNS} ({MAX_TOKENS} tokens each):", flush=True)
    rates = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        gens = decode_one(s0, s1, r0, r1, ids, MAX_TOKENS)
        dt = time.perf_counter() - t0
        rates.append(MAX_TOKENS / dt)
        print(f"  run {i+1}: {MAX_TOKENS/dt:.2f} tok/s  ({dt*1000/MAX_TOKENS:.1f} ms/tok wall)", flush=True)

    print(f"\nmean: {statistics.mean(rates):.2f} tok/s  sd={statistics.stdev(rates):.2f}", flush=True)
    decoded = tok.decode(gens, skip_special_tokens=True)
    print(f"output: {decoded!r}", flush=True)


if __name__ == "__main__":
    main()
