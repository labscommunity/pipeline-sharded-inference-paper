"""Stress-test v7 at 1024 and 2048 tokens.

Cache grows unboundedly with masked-out entries. This test looks for:
  (a) absolute throughput at realistic long-form generation lengths
  (b) memory impact (via cache_len tracking)
  (c) any degradation pattern from cache growth
"""
import os, time, statistics, numpy as np, openvino as ov
import sys
sys.path.insert(0, r"C:\cascadia")
from cascadia.pipeline.spec_decode import make_masked_req_from_model, spec_decode_greedy
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
K = 3


def feed(req, has_beam, input_ids, pos_start):
    n = input_ids.shape[1]
    att = np.ones((1, pos_start + n), dtype=np.int64)
    pos = np.arange(pos_start, pos_start + n, dtype=np.int64).reshape(1, -1)
    fd = {"input_ids": input_ids, "attention_mask": att, "position_ids": pos}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


def simple_decode(req, has_beam, prompt_ids, max_tokens):
    req.reset_state(); pos = 0; gens = []
    l = feed(req, has_beam, prompt_ids, pos); pos += prompt_ids.shape[1]
    nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    for _ in range(1, max_tokens):
        l = feed(req, has_beam, np.array([[nt]], dtype=np.int64), pos); pos += 1
        nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    return gens


print(f"OV {ov.__version__}  K={K}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
t_c = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
t_req = t_c.create_infer_request()
t_beam = any("beam_idx" in i.get_names() for i in t_c.inputs)
t_m = make_masked_req_from_model(t_c)
d_m = make_masked_req_from_model(d_c)

# Global warmup at 128
simple_decode(t_req, t_beam, input_ids, 128)
spec_decode_greedy(t_m, d_m, input_ids, 128, k=K)

print(f"\n{'max_toks':>9s} {'baseline':>10s} {'spec':>10s} {'accept':>8s} {'speedup':>8s} {'cache_grow':>12s}", flush=True)
print("-" * 70)

for max_toks in [128, 512, 1024, 2048]:
    # Baseline
    t0 = time.perf_counter()
    _ = simple_decode(t_req, t_beam, input_ids, max_toks)
    baseline_tps = max_toks / (time.perf_counter() - t0)

    # Spec
    t0 = time.perf_counter()
    spec_tokens, stats = spec_decode_greedy(t_m, d_m, input_ids, max_toks, k=K)
    spec_tps = max_toks / (time.perf_counter() - t0)

    # Approximate cache growth: logical_pos ≈ max_toks + prompt_len; cache_len is larger
    logical_est = max_toks + input_ids.shape[1]
    cache_ratio = t_m.cache_len / max(logical_est, 1)

    print(f"{max_toks:>9d} {baseline_tps:>8.2f} t/s {spec_tps:>8.2f} t/s {stats.accept_rate:>7.1%} {spec_tps/baseline_tps:>7.2f}x {cache_ratio:>11.2f}x", flush=True)
