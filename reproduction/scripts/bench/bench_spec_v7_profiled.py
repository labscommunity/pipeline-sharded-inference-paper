"""Profile v7 per-component time — same structure as v6 profiler but w/o trim_kv."""
import os, time, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
MAX_TOKENS = 128
K = int(os.environ.get("K", 3))


def feed_raw(req, has_beam, input_ids, attn_mask, pos_ids):
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
        out = feed_raw(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k_invalid):
        if k_invalid <= 0:
            return
        self.valid_mask[self.cache_len - k_invalid : self.cache_len] = 0
        self.logical_pos -= k_invalid


tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
t_compiled = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
d_compiled = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
t_req = t_compiled.create_infer_request()
d_req = d_compiled.create_infer_request()
t_beam = any("beam_idx" in i.get_names() for i in t_compiled.inputs)
d_beam = any("beam_idx" in i.get_names() for i in d_compiled.inputs)
t_m = MaskedReq(t_req, t_beam)
d_m = MaskedReq(d_req, d_beam)


def spec_run(max_tokens, K):
    t_m.reset(); d_m.reset()
    times = {"t_prefill": 0, "d_prefill": 0, "t_first": 0, "d_first": 0,
             "t_verify": 0, "d_drafts": 0, "d_rewind": 0, "cpu_mask": 0, "cpu_argmax": 0}

    t0 = time.perf_counter()
    t_l = t_m.feed(input_ids); times["t_prefill"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    d_l = d_m.feed(input_ids); times["d_prefill"] += time.perf_counter() - t0

    first = int(np.argmax(t_l[0, -1, :]))
    gens = [first]
    prev_correction = first

    t0 = time.perf_counter()
    d_l = d_m.feed(np.array([[first]], dtype=np.int64))
    times["d_first"] += time.perf_counter() - t0
    d_last_logit = d_l[0, -1, :].copy()

    total_accepted = 0
    total_drafts = 0
    n_steps = 0

    while len(gens) < max_tokens:
        n_steps += 1

        # Draft drafts
        drafts = [int(np.argmax(d_last_logit))]
        t0 = time.perf_counter()
        for i in range(1, K):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_l = d_m.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_l[0, -1, :])))
        times["d_drafts"] += time.perf_counter() - t0
        total_drafts += len(drafts)

        # Verify
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t0 = time.perf_counter()
        t_l = t_m.feed(verify)
        times["t_verify"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        tgt_new_greedy = np.argmax(t_l[0], axis=-1)
        accepted = 0
        for i in range(len(drafts)):
            if int(tgt_new_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        total_accepted += accepted
        times["cpu_argmax"] += time.perf_counter() - t0

        if accepted < len(drafts):
            correction = int(tgt_new_greedy[accepted])
        else:
            correction = int(tgt_new_greedy[len(drafts)])

        new_tokens = drafts[:accepted] + [correction]
        for tk in new_tokens:
            if len(gens) >= max_tokens:
                break
            gens.append(tk)

        # Rewinds
        t_m.rewind(len(drafts) - accepted)
        d_advanced = len(drafts) - 1

        t0 = time.perf_counter()
        if accepted < len(drafts):
            d_m.rewind(d_advanced - accepted)
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        else:
            d_l = d_m.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_l = d_m.feed(np.array([[correction]], dtype=np.int64))
        times["d_rewind"] += time.perf_counter() - t0
        d_last_logit = d_l[0, -1, :].copy()

        prev_correction = correction

    return gens, n_steps, total_accepted, total_drafts, times


# Warmup
for _ in range(2):
    spec_run(MAX_TOKENS, K)

print(f"=== K={K}  MAX={MAX_TOKENS} ===", flush=True)
t0 = time.perf_counter()
gens, n_steps, n_acc, n_drafts, times = spec_run(MAX_TOKENS, K)
total = time.perf_counter() - t0
print(f"Total: {total:.3f}s  ({len(gens)/total:.2f} tok/s)  steps={n_steps} accept_rate={n_acc/max(n_drafts,1):.1%}")
print()
for k, v in times.items():
    pct = v/total*100
    print(f"  {k:12s}: {v*1000:6.0f}ms  ({pct:5.1f}%)  per-step={v*1000/n_steps:5.1f}ms")
acc = sum(times.values())
print(f"  unaccounted: {(total-acc)*1000:.0f}ms  ({(total-acc)/total*100:.1f}%)")
