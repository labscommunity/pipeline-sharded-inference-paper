"""Speculative decoding v7 — mask-based rewind (no trim_kv).

Key innovation: instead of physically trimming KV state (~40ms per call),
use attention_mask=0 for rejected-draft cache positions. Tokens stay in cache
but the model ignores them. Bit-exact equivalent to trim, ~free in cost.

State tracking:
  valid_mask: np.ndarray of length cache_len. 1 = valid, 0 = masked (ignore).
  cache_len: physical cache length (grows monotonically).
  logical_pos: next logical position for position_ids.

Per step:
  1. Draft generates K drafts (cache_len grows K-1 on draft, mask all valid).
  2. Target verify [prev_correction, drafts[0..K-1]] — cache_len += K+1, all new positions valid.
  3. Decide accept count A.
  4. INVALIDATE last (K-A) entries by setting valid_mask[-(K-A):] = 0.
     Decrement logical_pos by (K-A).
  5. Similarly rewind draft valid_mask.

Note: cache grows unboundedly. For long generations, need periodic cleanup.
For 128-token bench it's fine.
"""
import os, sys, time, statistics
import numpy as np
import openvino as ov
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128
K = int(os.environ.get("K", 5))


def feed_raw(req, has_beam, input_ids, attn_mask, pos_ids):
    fd = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pos_ids}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


def feed(req, has_beam, input_ids, pos_start):
    n = input_ids.shape[1]
    att = np.ones((1, pos_start + n), dtype=np.int64)
    pos = np.arange(pos_start, pos_start + n, dtype=np.int64).reshape(1, -1)
    return feed_raw(req, has_beam, input_ids, att, pos)


def simple_decode(req, has_beam, prompt_ids, max_tokens):
    req.reset_state()
    pos = 0
    gens = []
    logits = feed(req, has_beam, prompt_ids, pos); pos += prompt_ids.shape[1]
    nt = int(np.argmax(logits[0, -1, :])); gens.append(nt)
    for _ in range(1, max_tokens):
        logits = feed(req, has_beam, np.array([[nt]], dtype=np.int64), pos); pos += 1
        nt = int(np.argmax(logits[0, -1, :])); gens.append(nt)
    return gens


class MaskedReq:
    """Wraps an InferRequest with mask-based KV tracking. cache_len grows; rewinds just mask out."""
    def __init__(self, req, has_beam):
        self.req = req
        self.has_beam = has_beam
        # Preallocate a generous mask buffer; grow as needed.
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        self.req.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids):
        """Feed n tokens at current logical_pos. All new positions are valid."""
        n = input_ids.shape[1]
        # Ensure mask buffer is large enough for past + new
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[:len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask
        # Build attn_mask: concat past-valid-mask + ones for new tokens
        att = np.empty((1, total), dtype=np.int64)
        att[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        att[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
        out = feed_raw(self.req, self.has_beam, input_ids, att, pos)
        # New positions are valid (already in valid_mask buffer as 1)
        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k_invalid):
        """Mark last k_invalid cache positions as invalid. logical_pos decreases by k_invalid."""
        if k_invalid <= 0:
            return
        self.valid_mask[self.cache_len - k_invalid : self.cache_len] = 0
        self.logical_pos -= k_invalid


def spec_decode_v7(t_mreq, d_mreq, prompt_ids, max_tokens, k):
    t_mreq.reset(); d_mreq.reset()
    prompt_len = prompt_ids.shape[1]

    # Prefill
    t_l = t_mreq.feed(prompt_ids)
    d_l = d_mreq.feed(prompt_ids)

    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]
    prev_correction = first

    # Feed first into draft (so d_last_logit predicts next)
    d_l = d_mreq.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_l[0, -1, :].copy()

    n_steps = 0
    total_drafts = 0
    total_accepted = 0

    while len(gens) < max_tokens:
        n_steps += 1

        # 1. Draft generates K drafts
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_l = d_mreq.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        d_advanced = len(drafts) - 1
        total_drafts += len(drafts)

        # 2. Target verifies [prev_correction, drafts[0..K-1]]
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_l = t_mreq.feed(verify)
        tgt_new_greedy = np.argmax(t_l[0], axis=-1)  # [K+1]

        # 3. Acceptance
        accepted = 0
        for i in range(len(drafts)):
            if int(tgt_new_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        total_accepted += accepted

        # 4. Correction
        if accepted < len(drafts):
            correction = int(tgt_new_greedy[accepted])
        else:
            correction = int(tgt_new_greedy[len(drafts)])

        # 5. Emit
        new_tokens = drafts[:accepted] + [correction]
        for tk in new_tokens:
            if len(gens) >= max_tokens:
                break
            gens.append(tk)

        # 6. Rewind target: keep prev_correction + drafts[0..A-1] from the K+1 fed.
        # Invalidate last (K - A) positions.
        t_mreq.rewind(len(drafts) - accepted)

        # 7. Rewind draft: we fed drafts[0..K-2] (= d_advanced = K-1 tokens).
        # We want valid = drafts[0..A-1] + correction.
        # If A <= K-1 (d_advanced): invalidate last (d_advanced - A) positions, then feed correction.
        # If A == K: all of drafts[0..K-2] are valid. Feed drafts[K-1] then correction.
        if accepted < len(drafts):
            d_mreq.rewind(d_advanced - accepted)
            d_l = d_mreq.feed(np.array([[correction]], dtype=np.int64))
        else:
            d_l = d_mreq.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_l = d_mreq.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_l[0, -1, :].copy()

        prev_correction = correction

    return gens[:max_tokens], n_steps, total_accepted, total_drafts


# ---- main ----
print(f"OV {ov.__version__}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)
print(f"Prompt: {input_ids.shape}", flush=True)

core = ov.Core()
t_compiled = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
t_req = t_compiled.create_infer_request()
t_beam = any("beam_idx" in i.get_names() for i in t_compiled.inputs)

d_compiled = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
d_req = d_compiled.create_infer_request()
d_beam = any("beam_idx" in i.get_names() for i in d_compiled.inputs)

t_mreq = MaskedReq(t_req, t_beam)
d_mreq = MaskedReq(d_req, d_beam)


def timed(fn, n_runs=3, warmup=2):
    for _ in range(warmup):
        fn()
    ts = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        n = len(out[0]) if isinstance(out, tuple) else len(out)
        ts.append(n / dt)
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0, out


print("\n=== A: target only ===", flush=True)
mean_a, sd_a, gens_a = timed(lambda: simple_decode(t_req, t_beam, input_ids, MAX_TOKENS))
print(f"A: {mean_a:.3f} tok/s  sd={sd_a:.3f}", flush=True)
print(f"  first10: {gens_a[:10]}", flush=True)

print(f"\n=== A_sd v7 (masked): K={K} ===", flush=True)
def run():
    return spec_decode_v7(t_mreq, d_mreq, input_ids, MAX_TOKENS, K)
mean_sd, sd_sd, (gens_sd, n_steps, n_acc, n_drafts) = timed(run)
print(f"A_sd: {mean_sd:.3f} tok/s  sd={sd_sd:.3f}", flush=True)
print(f"  steps={n_steps}  drafts={n_drafts}  accepted={n_acc}  rate={n_acc/max(n_drafts,1):.1%}", flush=True)
print(f"  first10: {gens_sd[:10]}", flush=True)
print(f"  match baseline: {gens_sd[:10] == gens_a[:10]}", flush=True)

print(f"\n=== SUMMARY ===", flush=True)
print(f"  A:   {mean_a:.2f} tok/s", flush=True)
print(f"  Asd: {mean_sd:.2f} tok/s", flush=True)
print(f"  Speedup: {mean_sd/mean_a:.2f}x", flush=True)
