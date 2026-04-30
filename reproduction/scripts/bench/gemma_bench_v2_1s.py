"""Bench Gemma 4 E2B 1-stage v2 (rotary-fixed) on alpha localhost GPU.

Single stage = embed + all 35 layers + head in one OV graph.
Paper claims 13.3 tok/s as the 1-stage single-node baseline.
With rotary fix, expect to compile at default precision (no FP32 hint).
"""
import os, time, statistics, json, numpy as np
import openvino as ov

SHARD_DIR = r"C:\cascadia\shards_e2b_1stage_v2"
DEVICE = "GPU"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 50
N_RUNS = 5
N_WARMUP = 2


def main():
    print(f"OV {ov.__version__}", flush=True)
    print(f"Shards: {SHARD_DIR}", flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(SHARD_DIR, "tokenizer"))

    core = ov.Core()
    print("compiling stage_0 (1-stage, DEFAULT precision)...", flush=True)
    t0 = time.perf_counter()
    s0 = core.compile_model(
        os.path.join(SHARD_DIR, "stage_0/openvino_model.xml"), DEVICE)
    print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

    in_names = [list(i.get_names())[0] for i in s0.inputs]
    out_names = [list(o.get_names())[0] for o in s0.outputs]
    print(f"input names: {in_names}", flush=True)
    print(f"output names: {out_names}", flush=True)

    r = s0.create_infer_request()

    msgs = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    print(f"prompt: {len(ids)} tokens", flush=True)

    in_id_name = "input_ids" if "input_ids" in in_names else in_names[0]
    pos_name = "position_ids" if "position_ids" in in_names else None

    def decode_one(ids, max_tokens):
        r.reset_state()
        ids_np = np.array([ids], dtype=np.int64)
        pos_np = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
        feed = {in_id_name: ids_np}
        if pos_name:
            feed[pos_name] = pos_np
        res = r.infer(feed)
        logits = res[s0.outputs[0]]
        nt = int(np.argmax(logits[0, -1, :]))
        gens = [nt]
        for step in range(max_tokens - 1):
            ids_one = np.array([[gens[-1]]], dtype=np.int64)
            pos_one = np.array([[len(ids) + step]], dtype=np.int64)
            feed = {in_id_name: ids_one}
            if pos_name:
                feed[pos_name] = pos_one
            res = r.infer(feed)
            nt = int(np.argmax(res[s0.outputs[0]][0, -1, :]))
            gens.append(nt)
        return gens

    print(f"\nwarmup x{N_WARMUP}...", flush=True)
    for _ in range(N_WARMUP):
        decode_one(ids, MAX_TOKENS)

    print(f"timed x{N_RUNS} ({MAX_TOKENS} tokens each):", flush=True)
    rates = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        gens = decode_one(ids, MAX_TOKENS)
        dt = time.perf_counter() - t0
        rates.append(MAX_TOKENS / dt)
        print(f"  run {i+1}: {MAX_TOKENS/dt:.2f} tok/s  ({dt*1000/MAX_TOKENS:.1f} ms/tok wall)", flush=True)

    print(f"\nmean: {statistics.mean(rates):.2f} tok/s  sd={statistics.stdev(rates):.2f}", flush=True)
    print(f"output: {tok.decode(gens, skip_special_tokens=True)!r}", flush=True)


if __name__ == "__main__":
    main()
