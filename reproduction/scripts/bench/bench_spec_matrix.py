"""Bench matrix: {monolithic, sharded} x {target-only, + spec decode}.

Shows spec decode speedup stacks with sharding (i.e., sharded+spec roughly
matches monolithic+spec).
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
        self.req = req
        self.has_beam = has_beam
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        self.req.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[:len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask
        att = np.empty((1, total), dtype=np.int64)
        att[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        att[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
        out = feed_raw_mono(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


class ShardedMaskedReq:
    def __init__(self, stage_reqs):
        self.reqs = stage_reqs
        self.valid_mask = np.ones(4096, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self):
        for r in self.reqs:
            r.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids):
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[:len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask
        att = np.empty((1, total), dtype=np.int64)
        att[0, :self.cache_len] = self.valid_mask[:self.cache_len]
        att[0, self.cache_len:] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)
        beam = np.zeros(1, dtype=np.int32)

        self.reqs[0].infer({"input_ids": input_ids, "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        h = self.reqs[0].get_output_tensor(0).data
        self.reqs[1].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        h = self.reqs[1].get_output_tensor(0).data
        self.reqs[2].infer({"hidden_states": h.astype(np.float32), "attention_mask": att,
                            "position_ids": pos, "beam_idx": beam})
        out = self.reqs[2].get_output_tensor(0).data

        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k):
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


def simple_decode(t_m, prompt_ids, max_tokens):
    t_m.reset()
    l = t_m.feed(prompt_ids)
    nt = int(np.argmax(l[0, -1, :]))
    gens = [nt]
    for _ in range(1, max_tokens):
        l = t_m.feed(np.array([[nt]], dtype=np.int64))
        nt = int(np.argmax(l[0, -1, :]))
        gens.append(nt)
    return gens


def spec_decode_v7(t_m, d_m, prompt_ids, max_tokens, k):
    t_m.reset(); d_m.reset()
    t_l = t_m.feed(prompt_ids)
    d_l = d_m.feed(prompt_ids)
    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]
    prev_correction = first
    d_l = d_m.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_l[0, -1, :].copy()
    n_steps = total_drafts = total_accepted = 0
    while len(gens) < max_tokens:
        n_steps += 1
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_l = d_m.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        d_advanced = len(drafts) - 1
        total_drafts += len(drafts)
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_l = t_m.feed(verify)
        tgt_new_greedy = np.argmax(t_l[0], axis=-1)
        accepted = 0
        for i in range(len(drafts)):
            if int(tgt_new_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        total_accepted += accepted
        if accepted < len(drafts):
            correction = int(tgt_new_greedy[accepted])
        else:
            correction = int(tgt_new_greedy[len(drafts)])
        new_tokens = drafts[:accepted] + [correction]
        for tk in new_tokens:
            if len(gens) >= max_tokens:
                break
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
    return gens[:max_tokens], n_steps, total_accepted, total_drafts


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


print(f"OV {ov.__version__}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET_MONO)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
print("compiling monolithic target...", flush=True)
t_mono_c = core.compile_model(core.read_model(os.path.join(TARGET_MONO, "openvino_model.xml")), "GPU")
t_mono_req = t_mono_c.create_infer_request()
t_mono_beam = any("beam_idx" in i.get_names() for i in t_mono_c.inputs)
t_mono = MaskedReq(t_mono_req, t_mono_beam)

print("compiling 3-stage v5_beam target...", flush=True)
t_shard_reqs = []
for s in TARGET_STAGES:
    c = core.compile_model(core.read_model(os.path.join(s, "openvino_model.xml")), "GPU")
    t_shard_reqs.append(c.create_infer_request())
t_shard = ShardedMaskedReq(t_shard_reqs)

print("compiling draft...", flush=True)
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
d_req = d_c.create_infer_request()
d_beam = any("beam_idx" in i.get_names() for i in d_c.inputs)
d_m = MaskedReq(d_req, d_beam)

results = {}

print("\n=== A: monolithic, target only ===", flush=True)
m, s, g = timed(lambda: simple_decode(t_mono, input_ids, MAX_TOKENS))
print(f"{m:.3f} tok/s  sd={s:.3f}  first10={g[:10]}", flush=True)
results["A_mono"] = {"tok_s": m, "sd": s, "gens": g[:20]}

print("\n=== B: 3-stage v5_beam shards, target only ===", flush=True)
m, s, g = timed(lambda: simple_decode(t_shard, input_ids, MAX_TOKENS))
print(f"{m:.3f} tok/s  sd={s:.3f}  first10={g[:10]}", flush=True)
results["B_shard"] = {"tok_s": m, "sd": s, "gens": g[:20]}

print(f"\n=== C: monolithic + spec decode K={K} ===", flush=True)
m, s, out = timed(lambda: spec_decode_v7(t_mono, d_m, input_ids, MAX_TOKENS, K))
g, ns, na, nd = out
print(f"{m:.3f} tok/s  sd={s:.3f}  steps={ns}  acc={na/max(nd,1):.1%}", flush=True)
print(f"  first10={g[:10]}  match A: {g[:10] == results['A_mono']['gens'][:10]}", flush=True)
results["C_mono_spec"] = {"tok_s": m, "sd": s, "accept": na/max(nd,1), "gens": g[:20]}

print(f"\n=== D: 3-stage shards + spec decode K={K} ===", flush=True)
m, s, out = timed(lambda: spec_decode_v7(t_shard, d_m, input_ids, MAX_TOKENS, K))
g, ns, na, nd = out
print(f"{m:.3f} tok/s  sd={s:.3f}  steps={ns}  acc={na/max(nd,1):.1%}", flush=True)
print(f"  first10={g[:10]}  match B: {g[:10] == results['B_shard']['gens'][:10]}", flush=True)
results["D_shard_spec"] = {"tok_s": m, "sd": s, "accept": na/max(nd,1), "gens": g[:20]}

print("\n=== SUMMARY ===", flush=True)
a = results["A_mono"]["tok_s"]
b = results["B_shard"]["tok_s"]
c = results["C_mono_spec"]["tok_s"]
d = results["D_shard_spec"]["tok_s"]
print(f"  A (mono, target only):      {a:6.2f} tok/s  (baseline)", flush=True)
print(f"  B (shard, target only):     {b:6.2f} tok/s  ({b/a:.2f}x mono baseline)", flush=True)
print(f"  C (mono + spec K={K}):        {c:6.2f} tok/s  ({c/a:.2f}x mono baseline)", flush=True)
print(f"  D (shard + spec K={K}):       {d:6.2f} tok/s  ({d/a:.2f}x mono baseline, {d/b:.2f}x shard baseline)", flush=True)

with open("/tmp/spec_matrix_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results written to /tmp/spec_matrix_results.json", flush=True)
