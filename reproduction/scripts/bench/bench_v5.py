"""Quick bench of v5 shard (canonical inputs: input_ids, attention_mask, position_ids)."""
import os
import sys
import time
import statistics
import numpy as np
import openvino as ov
from transformers import AutoTokenizer

SHARD_DIR = sys.argv[1] if len(sys.argv) > 1 else r"C:\cascadia\shards_1stage_v5\stage_0"
TOK_PATH = r"C:\cascadia\models\llama-3.1-8b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 50

tok = AutoTokenizer.from_pretrained(TOK_PATH)
msgs = [{"role": "user", "content": PROMPT}]
formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
input_ids = tok.encode(formatted, return_tensors="np", add_special_tokens=False).astype(np.int64)
seq_len = input_ids.shape[1]
print(f"Tokenized: {input_ids.shape}", flush=True)

core = ov.Core()
model = core.read_model(os.path.join(SHARD_DIR, "openvino_model.xml"))
print(f"Model inputs ({len(model.inputs)}):", flush=True)
for i, inp in enumerate(model.inputs):
    print(f"  [{i}] {sorted(inp.get_names())} shape={inp.get_partial_shape()}", flush=True)

compiled = core.compile_model(model, "GPU")
has_beam_idx = any("beam_idx" in inp.get_names() for inp in compiled.inputs)
request = compiled.create_infer_request()


def reset_kv(req):
    # For genai-style stateful IR with dynamic KV shape, reset_state() alone is
    # sufficient; trying to set state shape explicitly fails on dynamic dims.
    req.reset_state()


def gen_one():
    reset_kv(request)
    gens = []
    # Prefill
    att_mask = np.ones((1, seq_len), dtype=np.int64)
    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    feed = {"input_ids": input_ids, "attention_mask": att_mask, "position_ids": pos_ids}
    if has_beam_idx:
        feed["beam_idx"] = np.zeros(1, dtype=np.int32)
    request.infer(feed)
    logits = request.get_output_tensor(0).data
    nt = int(np.argmax(logits[0, -1, :]))
    gens.append(nt)
    # Decode
    for i in range(1, MAX_TOKENS):
        ids = np.array([[nt]], dtype=np.int64)
        att_mask = np.ones((1, seq_len + i), dtype=np.int64)
        pos_ids = np.array([[seq_len + i - 1]], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": att_mask, "position_ids": pos_ids}
        if has_beam_idx:
            feed["beam_idx"] = np.zeros(1, dtype=np.int32)
        request.infer(feed)
        logits = request.get_output_tensor(0).data
        nt = int(np.argmax(logits[0, -1, :]))
        gens.append(nt)
    return gens


print("warmup x2...", flush=True)
for _ in range(2):
    gen_one()

tok_ss = []
for i in range(5):
    t0 = time.perf_counter()
    toks = gen_one()
    dt = time.perf_counter() - t0
    ts = len(toks) / dt
    tok_ss.append(ts)
    print(f"  run {i+1}: {len(toks)} tok in {dt:.3f}s -> {ts:.2f} tok/s", flush=True)

print(f"\nmean = {statistics.mean(tok_ss):.3f}  stddev = {statistics.stdev(tok_ss):.3f}", flush=True)
print(f"first 15 tokens: {toks[:15]}", flush=True)
