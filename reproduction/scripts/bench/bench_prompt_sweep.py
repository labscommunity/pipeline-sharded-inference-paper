"""Multi-prompt robustness sweep for v7 spec decode.

Shows the 1.36x LAN speedup isn't cherry-picked — reports speedup
across diverse prompts (factual Q, code completion, creative writing,
reasoning).
"""
import os, time, statistics, numpy as np, openvino as ov
from transformers import AutoTokenizer
import sys
sys.path.insert(0, r"C:\cascadia")
from cascadia.pipeline.spec_decode import make_masked_req_from_model, spec_decode_greedy

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
MAX_TOKENS = 128
K = 3

PROMPTS = [
    ("short-factual",     "What is the capital of France?"),
    ("reasoning",         "If a train leaves Chicago at 3pm traveling 60 mph and another leaves New York at 4pm traveling 80 mph, when do they meet?"),
    ("code-completion",   "def fibonacci(n):\n    if n <= 1:\n        return n\n    "),
    ("list-enumeration",  "Five important inventions of the 20th century:\n1."),
    ("creative",          "Once upon a time, in a kingdom far away, a young prince discovered"),
    ("technical-expl",    "HTTP/2 multiplexing works by"),
    ("chat-assistant",    "User: Can you help me understand recursion?\nAssistant:"),
    ("translation",       "Translate 'Hello, how are you?' to French:"),
]


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


print(f"OV {ov.__version__}  K={K}  MAX_TOKENS={MAX_TOKENS}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET)

core = ov.Core()
t_c = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
t_req = t_c.create_infer_request()
t_beam = any("beam_idx" in i.get_names() for i in t_c.inputs)
t_m = make_masked_req_from_model(t_c)
d_m = make_masked_req_from_model(d_c)

# Quick per-prompt warmup
# Do one global warmup on the first prompt
input_ids = tok.encode(PROMPTS[0][1], return_tensors="np").astype(np.int64)
simple_decode(t_req, t_beam, input_ids, MAX_TOKENS)
spec_decode_greedy(t_m, d_m, input_ids, MAX_TOKENS, k=K)

print(f"\n{'prompt':20s} {'plen':>5s} {'baseline':>10s} {'spec':>10s} {'accept':>8s} {'speedup':>8s} {'match':>5s}", flush=True)
print(f"{'-'*70}", flush=True)

speedups = []
for name, text in PROMPTS:
    input_ids = tok.encode(text, return_tensors="np").astype(np.int64)
    plen = input_ids.shape[1]

    # Baseline (no warmup per prompt to stay quick; rely on global warmup)
    t0 = time.perf_counter()
    baseline = simple_decode(t_req, t_beam, input_ids, MAX_TOKENS)
    baseline_tps = MAX_TOKENS / (time.perf_counter() - t0)

    # Spec
    t0 = time.perf_counter()
    spec_tokens, stats = spec_decode_greedy(t_m, d_m, input_ids, MAX_TOKENS, k=K)
    spec_tps = MAX_TOKENS / (time.perf_counter() - t0)

    speedup = spec_tps / baseline_tps
    match = spec_tokens[:10] == baseline[:10]
    speedups.append(speedup)
    print(f"{name:20s} {plen:>5d} {baseline_tps:>8.2f} t/s {spec_tps:>8.2f} t/s {stats.accept_rate:>7.1%} {speedup:>7.2f}x {'Y' if match else 'N':>5s}", flush=True)

print(f"\nmean speedup across {len(PROMPTS)} prompts: {statistics.mean(speedups):.2f}x (stddev {statistics.stdev(speedups):.2f})", flush=True)
print(f"min: {min(speedups):.2f}x  max: {max(speedups):.2f}x", flush=True)
