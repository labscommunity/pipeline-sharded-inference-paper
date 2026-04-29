"""Spec decode across varying injected WAN latency.

Simulates network RTT between sharded target stages by sleep()-ing between
stage forwards. For N stages there are (N-1) inter-stage hops per forward,
so with latency_ms_per_hop we add (N-1)*latency_ms per token of target compute.

Shows the "WAN amortization" effect: spec decode batches K+1 tokens per
target call, so network overhead is paid once per K+1 tokens instead of
per-token. Speedup vs no-spec should grow with latency.
"""
import os, time, statistics, json
import numpy as np
import openvino as ov
from transformers import AutoTokenizer

TARGET_MONO = r"C:\cascadia\models\llama-3.1-8b-int4"
TARGET_STAGES = [
    r"C:\cascadia\shards_3stage_v5_beam\stage_0",
    r"C:\cascadia\shards_3stage_v5_beam\stage_1",
    r"C:\cascadia\shards_3stage_v5_beam\stage_2",
]
DRAFT = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128
K = 3


def feed_raw_mono(req, has_beam, input_ids, attn_mask, pos_ids):
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
        out = feed_raw_mono(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n; self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0: return
        self.valid_mask[self.cache_len - k: self.cache_len] = 0
        self.logical_pos -= k


class ShardedMaskedReq:
    """3-stage target with injected latency between stages."""
    def __init__(self, stage_reqs, latency_per_hop_s=0.0):
        self.reqs = stage_reqs
        self.latency_per_hop = latency_per_hop_s  # seconds per inter-stage hop
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

        if self.latency_per_hop > 0:
            time.sleep(self.latency_per_hop)   # simulate network transfer to stage 1

        self.reqs[1].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        h = self.reqs[1].get_output_tensor(0).data

        if self.latency_per_hop > 0:
            time.sleep(self.latency_per_hop)   # simulate network transfer to stage 2

        self.reqs[2].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        out = self.reqs[2].get_output_tensor(0).data

        if self.latency_per_hop > 0:
            time.sleep(self.latency_per_hop)   # simulate return of logits to coordinator

        self.cache_len += n; self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0: return
        self.valid_mask[self.cache_len - k: self.cache_len] = 0
        self.logical_pos -= k


def simple_decode(t_m, prompt_ids, max_tokens):
    t_m.reset()
    l = t_m.feed(prompt_ids)
    nt = int(np.argmax(l[0, -1, :])); gens = [nt]
    for _ in range(1, max_tokens):
        l = t_m.feed(np.array([[nt]], dtype=np.int64))
        nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    return gens


def spec_decode_v7(t_m, d_m, prompt_ids, max_tokens, k):
    t_m.reset(); d_m.reset()
    t_l = t_m.feed(prompt_ids); d_l = d_m.feed(prompt_ids)
    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]; prev_correction = first
    d_l = d_m.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_l[0, -1, :].copy()
    while len(gens) < max_tokens:
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens: break
            d_l = d_m.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        d_advanced = len(drafts) - 1
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_l = t_m.feed(verify)
        tgt = np.argmax(t_l[0], axis=-1)
        accepted = 0
        for i in range(len(drafts)):
            if int(tgt[i]) == drafts[i]: accepted += 1
            else: break
        if accepted < len(drafts): correction = int(tgt[accepted])
        else: correction = int(tgt[len(drafts)])
        new_tokens = drafts[:accepted] + [correction]
        for tk in new_tokens:
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
    return gens[:max_tokens]


# --- main ---
print(f"OV {ov.__version__}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET_MONO)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
print("compiling 3-stage v5_beam target + draft...", flush=True)
t_shard_reqs = []
for s in TARGET_STAGES:
    c = core.compile_model(core.read_model(os.path.join(s, "openvino_model.xml")), "GPU")
    t_shard_reqs.append(c.create_infer_request())
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
d_req = d_c.create_infer_request()
d_beam = any("beam_idx" in i.get_names() for i in d_c.inputs)
d_m = MaskedReq(d_req, d_beam)


def timed(fn, n_runs=2, warmup=1):
    # fewer runs for this sweep
    for _ in range(warmup): fn()
    ts = []
    for _ in range(n_runs):
        t0 = time.perf_counter(); out = fn()
        dt = time.perf_counter() - t0
        n = len(out) if isinstance(out, list) else len(out[0])
        ts.append(n / dt)
    return statistics.mean(ts), statistics.stdev(ts) if len(ts) > 1 else 0.0


print()
results = []
# hops per target call: 3 (s0->s1, s1->s2, s2->coord)
# RTTs scan: 0, 5, 10, 20, 50 ms per hop → total (3 hops * 2-way? or 1-way)
# Use 1-way latency between stages; coord sends prompt → s0 then gets logits back.
# For simplicity: latency_per_hop is 1-way inter-stage time.
LATENCIES_MS = [0, 5, 10, 20, 50, 100]

for lat_ms in LATENCIES_MS:
    t_shard = ShardedMaskedReq(t_shard_reqs, latency_per_hop_s=lat_ms / 1000.0)
    print(f"--- latency/hop = {lat_ms} ms ---", flush=True)
    m_ts, s_ts = timed(lambda: simple_decode(t_shard, input_ids, MAX_TOKENS))
    m_sd, s_sd = timed(lambda: spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, K))
    speedup = m_sd / m_ts
    print(f"  baseline: {m_ts:.3f} tok/s  spec: {m_sd:.3f} tok/s  speedup: {speedup:.2f}x", flush=True)
    results.append({"latency_ms": lat_ms, "baseline": m_ts, "spec": m_sd, "speedup": speedup})

print("\n=== SUMMARY ===", flush=True)
print(f"{'lat/hop':>8} {'baseline':>10} {'spec':>10} {'speedup':>10}", flush=True)
for r in results:
    print(f"{r['latency_ms']:>6}ms {r['baseline']:>8.2f} t/s {r['spec']:>8.2f} t/s {r['speedup']:>8.2f}x", flush=True)

with open(r"C:\cascadia\spec_wan_results.json", "w") as f:
    json.dump(results, f, indent=2)
