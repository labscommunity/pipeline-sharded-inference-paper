"""Measure Python overhead inside MaskedReq.feed() vs pure infer() time.

Breaks down each feed call into:
  - buffer allocation (np.empty / np.arange for attn_mask, pos_ids)
  - infer call
  - output .copy()

Uses the production cascadia.pipeline.spec_decode module's MaskedReq,
with timing hooks monkey-patched in.
"""
import os, time, numpy as np, openvino as ov
import sys
sys.path.insert(0, r"C:\cascadia")
from cascadia.pipeline.spec_decode import MaskedReq, spec_decode_greedy, make_masked_req_from_model
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128
K = 3


# Monkey-patch MaskedReq.feed with a timed version
_original_feed = MaskedReq.feed
_timings = {"alloc": 0.0, "infer": 0.0, "output": 0.0, "total": 0.0, "count": 0}


def timed_feed(self, input_ids):
    t_start = time.perf_counter()
    n = input_ids.shape[1]
    total = self.cache_len + n

    t_alloc_start = time.perf_counter()
    if total > len(self.valid_mask):
        new_size = max(total * 2, len(self.valid_mask) * 2)
        new_mask = np.ones(new_size, dtype=np.int64)
        new_mask[: len(self.valid_mask)] = self.valid_mask
        self.valid_mask = new_mask
    att = np.empty((1, total), dtype=np.int64)
    att[0, : self.cache_len] = self.valid_mask[: self.cache_len]
    att[0, self.cache_len :] = 1
    pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
    _timings["alloc"] += time.perf_counter() - t_alloc_start

    t_infer_start = time.perf_counter()
    fd = {"input_ids": input_ids, "attention_mask": att, "position_ids": pos}
    if self.has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    self.req.infer(fd)
    out = self.req.get_output_tensor(0).data
    _timings["infer"] += time.perf_counter() - t_infer_start

    t_output_start = time.perf_counter()
    # Mimic what callers typically do: slice and copy
    _ = out[0, -1, :]  # just read, no full copy for drafting — the caller copies
    _timings["output"] += time.perf_counter() - t_output_start

    self.cache_len += n
    self.logical_pos += n
    _timings["total"] += time.perf_counter() - t_start
    _timings["count"] += 1
    return out


MaskedReq.feed = timed_feed


tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
t_c = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
t_m = make_masked_req_from_model(t_c)
d_m = make_masked_req_from_model(d_c)

# Warmup
for _ in range(2):
    spec_decode_greedy(t_m, d_m, input_ids, MAX_TOKENS, k=K)

# Reset timings after warmup
for k in _timings:
    _timings[k] = 0.0 if k != "count" else 0

t_wall = time.perf_counter()
gens, stats = spec_decode_greedy(t_m, d_m, input_ids, MAX_TOKENS, k=K)
wall = time.perf_counter() - t_wall

print(f"\n=== Spec decode K={K}, {MAX_TOKENS} tokens ===")
print(f"Wall time: {wall*1000:.1f} ms  ({MAX_TOKENS/wall:.2f} tok/s)")
print(f"Steps: {stats.n_steps}  Accept rate: {stats.accept_rate:.1%}")
print(f"Total feed() calls: {_timings['count']}")
print(f"  (breakdown: expect {stats.n_steps} target verifies + ~{stats.n_steps*(K-1)} draft drafts + ~{stats.n_steps*2} rewind feeds + 2 prefills + 1 first-feed)")
print()
print(f"Per-feed timing (summed across all {_timings['count']} calls):")
print(f"  alloc:  {_timings['alloc']*1000:6.1f} ms  ({_timings['alloc']/_timings['total']*100:5.1f}% of feed time, {_timings['alloc']/wall*100:5.1f}% of wall)")
print(f"  infer:  {_timings['infer']*1000:6.1f} ms  ({_timings['infer']/_timings['total']*100:5.1f}% of feed time, {_timings['infer']/wall*100:5.1f}% of wall)")
print(f"  output: {_timings['output']*1000:6.1f} ms  ({_timings['output']/_timings['total']*100:5.1f}% of feed time, {_timings['output']/wall*100:5.1f}% of wall)")
print(f"  TOTAL:  {_timings['total']*1000:6.1f} ms  ({_timings['total']/wall*100:5.1f}% of wall)")
print(f"  non-feed wall: {(wall-_timings['total'])*1000:6.1f} ms  ({(wall-_timings['total'])/wall*100:5.1f}% of wall)")
print()
print(f"Per-call averages:")
print(f"  alloc:  {_timings['alloc']/_timings['count']*1e6:5.1f} us/call")
print(f"  infer:  {_timings['infer']/_timings['count']*1e6:5.1f} us/call")
print(f"  output: {_timings['output']/_timings['count']*1e6:5.1f} us/call")
