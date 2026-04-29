"""Bench Llama 3.1 8B INT4 v5_beam 2-stage shards in-process on alpha GPU.

Mirrors the bench_v5.py 1-stage protocol but chains two stages in a single
Python process — measures the §4 tab:shard_stages 2-stage row.
"""
import os, time, statistics
import numpy as np
import openvino as ov

SHARD_DIR = r"C:\cascadia\shards_2stage_v5_beam"
DEVICE = "GPU"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 50
N_RUNS = 5
N_WARMUP = 2


def main():
    print(f"OV {ov.__version__}", flush=True)
    core = ov.Core()
    print("Compiling stage_0 + stage_1 on GPU...", flush=True)
    t0 = time.perf_counter()
    s0 = core.compile_model(os.path.join(SHARD_DIR, "stage_0", "openvino_model.xml"), DEVICE)
    s1 = core.compile_model(os.path.join(SHARD_DIR, "stage_1", "openvino_model.xml"), DEVICE)
    print(f"  compile: {time.perf_counter()-t0:.1f}s", flush=True)
    r0 = s0.create_infer_request()
    r1 = s1.create_infer_request()

    print(f"s0 inputs: {[i.any_name for i in s0.inputs]}", flush=True)
    print(f"s0 outputs: {[o.any_name for o in s0.outputs]}", flush=True)
    print(f"s1 inputs: {[i.any_name for i in s1.inputs]}", flush=True)

    from transformers import AutoTokenizer
    tok_path = os.path.join(SHARD_DIR, "tokenizer")
    if not os.path.exists(tok_path):
        tok_path = r"C:\cascadia\models\llama-3.1-8b-instruct-int4"
    tok = AutoTokenizer.from_pretrained(tok_path)
    msgs = [{"role": "user", "content": PROMPT}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(chat, add_special_tokens=False)
    print(f"prompt: {len(ids)} tokens", flush=True)

    def decode_once(max_tokens):
        r0.reset_state()
        r1.reset_state()
        ids_np = np.array([ids], dtype=np.int64)
        pos_np = np.arange(len(ids), dtype=np.int64).reshape(1, -1)
        attn_np = np.ones((1, len(ids)), dtype=np.int64)
        beam_np = np.zeros((1,), dtype=np.int32)
        out0 = r0.infer({"input_ids": ids_np, "attention_mask": attn_np,
                         "position_ids": pos_np, "beam_idx": beam_np})
        h0 = list(out0.values())[0]
        out1 = r1.infer({"hidden_states": h0, "attention_mask": attn_np,
                         "position_ids": pos_np, "beam_idx": beam_np})
        logits = list(out1.values())[0]
        nt = int(np.argmax(logits[0, -1, :]))
        gens = [nt]
        for step in range(max_tokens - 1):
            ids_one = np.array([[gens[-1]]], dtype=np.int64)
            pos_one = np.array([[len(ids) + step]], dtype=np.int64)
            attn_one = np.ones((1, len(ids) + step + 1), dtype=np.int64)
            o0 = r0.infer({"input_ids": ids_one, "attention_mask": attn_one,
                           "position_ids": pos_one, "beam_idx": beam_np})
            h_one = list(o0.values())[0]
            o1 = r1.infer({"hidden_states": h_one, "attention_mask": attn_one,
                           "position_ids": pos_one, "beam_idx": beam_np})
            logits_one = list(o1.values())[0]
            gens.append(int(np.argmax(logits_one[0, -1, :])))
        return gens

    print(f"warmup x{N_WARMUP}...", flush=True)
    for _ in range(N_WARMUP):
        decode_once(MAX_TOKENS)
    rates = []
    for i in range(N_RUNS):
        t0 = time.perf_counter()
        gens = decode_once(MAX_TOKENS)
        dt = time.perf_counter() - t0
        rates.append(MAX_TOKENS / dt)
        if i == 0:
            print(f"first10: {gens[:10]}", flush=True)
        print(f"  run {i+1}: {MAX_TOKENS/dt:.2f} tok/s  ({dt*1000/MAX_TOKENS:.1f} ms/tok wall)", flush=True)
    print(f"mean: {statistics.mean(rates):.2f} tok/s  sd={statistics.stdev(rates):.2f}", flush=True)


if __name__ == "__main__":
    main()
