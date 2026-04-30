"""Threaded async spec decode: draft next-step's K drafts in parallel with target verify.

Uses Python threads — time.sleep() in the target's sleeps releases GIL, so
draft's infer() on a different OV InferRequest can run concurrently.

Strategy:
  Step N:
    - Start thread T that does target verify for step N (takes T ms including sleeps)
    - On main thread: generate speculative drafts for step N+1, assuming all drafts
      accepted AND correction = target_greedy[K]. Predict correction as draft's
      argmax after drafts[-1].
    - Join thread T, get target's tgt_new_greedy
    - Decide accept count A
    - If A == K AND correction == speculative_correction: use speculative next drafts
    - Else: rewind draft state, redo standard step N+1
"""
import os, time, statistics, threading, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET_STAGES = [
    r"C:\cascadia\shards_3stage_v5_beam\stage_0",
    r"C:\cascadia\shards_3stage_v5_beam\stage_1",
    r"C:\cascadia\shards_3stage_v5_beam\stage_2",
]
DRAFT = r"C:\cascadia\models\llama-3.2-1b-int4"
TARGET_MONO = r"C:\cascadia\models\llama-3.1-8b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128
K = int(os.environ.get("K", 10))
LATENCY_MS = int(os.environ.get("LATENCY_MS", 100))


def feed_raw(req, has_beam, input_ids, attn_mask, pos_ids):
    fd = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pos_ids}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


class MaskedReq:
    def __init__(self, req, has_beam):
        self.req = req; self.has_beam = has_beam
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0; self.logical_pos = 0

    def reset(self):
        self.req.reset_state(); self.valid_mask[:] = 1
        self.cache_len = 0; self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new = np.ones(max(total * 2, len(self.valid_mask) * 2), dtype=np.int64)
            new[:len(self.valid_mask)] = self.valid_mask; self.valid_mask = new
        att = np.empty((1, total), dtype=np.int64)
        att[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        att[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
        out = feed_raw(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n; self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0: return
        self.valid_mask[self.cache_len - k: self.cache_len] = 0
        self.logical_pos -= k


class ShardedMaskedReq:
    def __init__(self, stage_reqs, latency_per_hop_s=0.0):
        self.reqs = stage_reqs
        self.latency_per_hop = latency_per_hop_s
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0; self.logical_pos = 0

    def reset(self):
        for r in self.reqs: r.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0; self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new = np.ones(max(total * 2, len(self.valid_mask) * 2), dtype=np.int64)
            new[:len(self.valid_mask)] = self.valid_mask; self.valid_mask = new
        att = np.empty((1, total), dtype=np.int64)
        att[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        att[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
        beam = np.zeros(1, dtype=np.int32)
        self.reqs[0].infer({"input_ids": input_ids, "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        h = self.reqs[0].get_output_tensor(0).data
        if self.latency_per_hop > 0: time.sleep(self.latency_per_hop)
        self.reqs[1].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        h = self.reqs[1].get_output_tensor(0).data
        if self.latency_per_hop > 0: time.sleep(self.latency_per_hop)
        self.reqs[2].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        out = self.reqs[2].get_output_tensor(0).data
        if self.latency_per_hop > 0: time.sleep(self.latency_per_hop)
        self.cache_len += n; self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0: return
        self.valid_mask[self.cache_len - k: self.cache_len] = 0
        self.logical_pos -= k


def simple_decode(t_m, prompt_ids, max_tokens):
    t_m.reset()
    l = t_m.feed(prompt_ids); nt = int(np.argmax(l[0, -1, :])); gens = [nt]
    for _ in range(1, max_tokens):
        l = t_m.feed(np.array([[nt]], dtype=np.int64))
        nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    return gens


def spec_sync(t_m, d_m, prompt_ids, max_tokens, k):
    t_m.reset(); d_m.reset()
    t_l = t_m.feed(prompt_ids); d_l = d_m.feed(prompt_ids)
    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]; prev_correction = first
    d_l = d_m.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_l[0, -1, :].copy()
    total_acc = total_drafts = 0
    while len(gens) < max_tokens:
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens: break
            d_l = d_m.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        d_advanced = len(drafts) - 1
        total_drafts += len(drafts)
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_l = t_m.feed(verify)
        tgt = np.argmax(t_l[0], axis=-1)
        accepted = 0
        for i in range(len(drafts)):
            if int(tgt[i]) == drafts[i]: accepted += 1
            else: break
        total_acc += accepted
        correction = int(tgt[accepted]) if accepted < len(drafts) else int(tgt[len(drafts)])
        for tk in drafts[:accepted] + [correction]:
            if len(gens) >= max_tokens: break
            gens.append(tk)
        t_m.rewind(len(drafts) - accepted)
        if accepted < len(drafts):
            d_m.rewind(d_advanced - accepted)
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        else:
            d_l = d_m.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_l[0, -1, :].copy()
        prev_correction = correction
    return gens[:max_tokens], total_acc, total_drafts


def spec_threaded(t_m, d_m, prompt_ids, max_tokens, k):
    """Async via Python threads: target verify on thread T, draft CURRENT step
    drafting on main thread SIMULTANEOUSLY.

    Wait — current-step drafts are NEEDED for verify input. So we can't overlap
    verify with current drafts.

    BUT we CAN overlap target verify with NEXT-STEP draft's speculative feed
    of drafts[-1]. The draft's feed(drafts[-1]) produces speculative_d_last
    which is only valid if target accepts all drafts AND correction matches.

    Let's just overlap THAT one feed. It's 16ms on iGPU, fits in the 300ms
    target wait. When target completes and we check: if speculation was right,
    we skip the draft rewind's correction feed (saving 16ms). If wrong, we
    rewind the speculative feed and do proper rewind.
    """
    t_m.reset(); d_m.reset()
    t_l = t_m.feed(prompt_ids); d_l = d_m.feed(prompt_ids)
    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]; prev_correction = first
    d_l = d_m.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_l[0, -1, :].copy()
    total_acc = total_drafts = 0

    while len(gens) < max_tokens:
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens: break
            d_l = d_m.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        d_advanced = len(drafts) - 1
        total_drafts += len(drafts)

        verify = np.array([[prev_correction] + drafts], dtype=np.int64)

        # Start target verify in a thread
        t_result = [None]
        def target_thread():
            t_result[0] = t_m.feed(verify)
        th = threading.Thread(target=target_thread)
        th.start()

        # While target is running, speculatively feed drafts[-1] to draft.
        # This is the "if all accepted" case's prep work.
        spec_d_cache_before = d_m.cache_len
        spec_d_l = d_m.feed(np.array([[drafts[-1]]], dtype=np.int64))
        spec_d_last_logit = spec_d_l[0, -1, :].copy()
        # We could speculatively draft up to K more tokens here — but risk of waste high.
        # Stick with just the one drafts[-1] feed for simplicity.

        th.join()
        t_l = t_result[0]
        tgt = np.argmax(t_l[0], axis=-1)

        accepted = 0
        for i in range(len(drafts)):
            if int(tgt[i]) == drafts[i]: accepted += 1
            else: break
        total_acc += accepted
        correction = int(tgt[accepted]) if accepted < len(drafts) else int(tgt[len(drafts)])

        for tk in drafts[:accepted] + [correction]:
            if len(gens) >= max_tokens: break
            gens.append(tk)
        t_m.rewind(len(drafts) - accepted)

        # Handle draft cache:
        # We fed drafts[-1] speculatively. Draft cache has drafts[0..K-2] (from loop)
        # + drafts[-1] (spec). That's K drafts total.
        # Target state: accepted out of K drafts accepted, correction chosen.
        #
        # Desired draft cache for next step:
        #   prompt + ... + prev_correction + drafts[0..A-1] + correction
        #
        # Actual draft cache now:
        #   prompt + ... + prev_correction + drafts[0..K-1] (all K)
        #
        # Three cases:
        #   (A) accepted == K AND correction == argmax(spec_d_last_logit):
        #       our spec was right! Draft cache is prompt+...+drafts[0..K-1].
        #       We want it to be prompt+...+drafts[0..K-1]+correction.
        #       So feed correction. d_cache += 1. But we already fed drafts[-1]
        #       and now the LOGIT for predicting correction came from that feed.
        #       We can skip the explicit correction feed if we'll soon get a
        #       logit via drafting next step's first draft... no wait we do need
        #       correction in cache because next step's drafts are conditioned on it.
        #
        #       Actually in sync version's all-accept case, we feed drafts[-1]
        #       AND correction (2 feeds). We've already done 1 feed (drafts[-1]).
        #       Just feed correction now. But we know d_last_logit for after
        #       drafts[-1] is spec_d_last_logit. And correction == argmax(spec).
        #       So drafts[0] of next step is argmax(spec_d_last_logit) == correction.
        #       Hmm that means drafts[0]_next = correction. We'd want to skip
        #       the correction feed and use spec_d_last_logit as d_last_logit.
        #       But then drafts[0]_next = correction would be the first draft,
        #       and prev_correction = correction would be in verify. So verify =
        #       [correction, correction, drafts[1..K-1]_next]. The first position
        #       of verify attends to prev_correction (= correction in cache).
        #       Hmm this is getting confusing.
        #
        #       Simpler: just feed correction like normal. Cache ends at
        #       drafts[0..K-1] + correction. d_last_logit = predicted next.
        #       Skip savings: 0 (we did extra spec feed but also needed correction
        #       feed same as sync, so net same).
        #
        #   (B) accepted == K AND correction != argmax(spec_d_last_logit):
        #       Our spec guess was wrong in content but draft cache layout is
        #       same as case A. Feed correction.
        #
        #   (C) accepted < K:
        #       Draft cache has drafts[0..K-1] but we only want drafts[0..A-1].
        #       Rewind K-A entries. Then feed correction.

        if accepted < len(drafts):
            # Case C: rewind speculative + (K-1-A) drafts = K - A
            to_rewind = len(drafts) - accepted  # includes the speculative drafts[-1]
            d_m.rewind(to_rewind)
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        else:
            # Case A or B: spec feed already added drafts[-1]. Feed correction.
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_l[0, -1, :].copy()
        prev_correction = correction

    return gens[:max_tokens], total_acc, total_drafts


print(f"OV {ov.__version__}  K={K}  latency={LATENCY_MS}ms/hop", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET_MONO)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
t_shard_reqs = []
for s in TARGET_STAGES:
    c = core.compile_model(core.read_model(os.path.join(s, "openvino_model.xml")), "GPU")
    t_shard_reqs.append(c.create_infer_request())
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
d_req = d_c.create_infer_request()
d_beam = any("beam_idx" in i.get_names() for i in d_c.inputs)
d_m = MaskedReq(d_req, d_beam)
t_m = ShardedMaskedReq(t_shard_reqs, latency_per_hop_s=LATENCY_MS / 1000.0)


def timed(fn, n_runs=2, warmup=1):
    for _ in range(warmup): fn()
    ts = []
    out = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        n = len(out) if isinstance(out, list) else len(out[0])
        ts.append(n / dt)
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0, out


print("baseline (no spec)...", flush=True)
m_a, s_a, _ = timed(lambda: simple_decode(t_m, input_ids, MAX_TOKENS))
print(f"  baseline: {m_a:.2f} tok/s", flush=True)

print("sync spec...", flush=True)
m_sync, s_sync, (g_sync, _, _) = timed(lambda: spec_sync(t_m, d_m, input_ids, MAX_TOKENS, K))
print(f"  sync:     {m_sync:.2f} tok/s  speedup={m_sync/m_a:.2f}x", flush=True)

print("threaded spec (overlap drafts[-1] feed with target wait)...", flush=True)
m_thr, s_thr, (g_thr, _, _) = timed(lambda: spec_threaded(t_m, d_m, input_ids, MAX_TOKENS, K))
print(f"  threaded: {m_thr:.2f} tok/s  speedup={m_thr/m_a:.2f}x", flush=True)

print(f"\n  match sync output: {g_thr[:10] == g_sync[:10]}", flush=True)
