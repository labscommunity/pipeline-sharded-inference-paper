"""Sweep K at high WAN latency — higher K better amortizes network RTT."""
import os, time, statistics, numpy as np, openvino as ov
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
        self.reqs = stage_reqs; self.latency_per_hop = latency_per_hop_s
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


def spec_decode_v7(t_m, d_m, prompt_ids, max_tokens, k):
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


print(f"OV {ov.__version__}", flush=True)
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

print(f"\n=== K sweep at 0 ms/hop (LAN) ===", flush=True)
t_shard = ShardedMaskedReq(t_shard_reqs, latency_per_hop_s=0.000)
simple_decode(t_shard, input_ids, MAX_TOKENS)  # warmup
t0 = time.perf_counter()
simple_decode(t_shard, input_ids, MAX_TOKENS)
baseline = MAX_TOKENS / (time.perf_counter() - t0)
print(f"  baseline (no spec): {baseline:.2f} tok/s", flush=True)
for k in [2, 3, 5, 7, 10]:
    spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)
    t0 = time.perf_counter()
    _, acc, drafts = spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)
    tps = MAX_TOKENS / (time.perf_counter() - t0)
    print(f"  K={k:2d}: {tps:.2f} tok/s  accept={acc/max(drafts,1):.1%}  speedup={tps/baseline:.2f}x", flush=True)


print(f"\n=== K sweep at 50 ms/hop ===", flush=True)
t_shard = ShardedMaskedReq(t_shard_reqs, latency_per_hop_s=0.050)

# Baseline at 50 ms/hop
simple_decode(t_shard, input_ids, MAX_TOKENS)  # warmup
t0 = time.perf_counter()
simple_decode(t_shard, input_ids, MAX_TOKENS)
baseline = MAX_TOKENS / (time.perf_counter() - t0)
print(f"  baseline (no spec): {baseline:.2f} tok/s", flush=True)

for k in [2, 3, 5, 7, 10]:
    spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)  # warmup
    t0 = time.perf_counter()
    _, acc, drafts = spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)
    tps = MAX_TOKENS / (time.perf_counter() - t0)
    print(f"  K={k:2d}: {tps:.2f} tok/s  accept={acc/max(drafts,1):.1%}  speedup={tps/baseline:.2f}x", flush=True)

print(f"\n=== K sweep at 100 ms/hop ===", flush=True)
t_shard = ShardedMaskedReq(t_shard_reqs, latency_per_hop_s=0.100)

simple_decode(t_shard, input_ids, MAX_TOKENS)  # warmup
t0 = time.perf_counter()
simple_decode(t_shard, input_ids, MAX_TOKENS)
baseline = MAX_TOKENS / (time.perf_counter() - t0)
print(f"  baseline (no spec): {baseline:.2f} tok/s", flush=True)

for k in [2, 3, 5, 7, 10]:
    spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)  # warmup
    t0 = time.perf_counter()
    _, acc, drafts = spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, k)
    tps = MAX_TOKENS / (time.perf_counter() - t0)
    print(f"  K={k:2d}: {tps:.2f} tok/s  accept={acc/max(drafts,1):.1%}  speedup={tps/baseline:.2f}x", flush=True)
