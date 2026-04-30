"""Stress-test v7 spec decode at longer generation lengths.

Cache grows unboundedly with masked-out entries → see if tok/s holds up.
"""
import os, time, statistics, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"
DRAFT  = r"C:\cascadia\models\llama-3.2-1b-int4"
PROMPT = "What is the capital of France?"
K = 3


def feed_raw(req, has_beam, input_ids, attn_mask, pos_ids):
    fd = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pos_ids}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


class MaskedReq:
    def __init__(self, req, has_beam):
        self.req = req; self.has_beam = has_beam
        self.valid_mask = np.ones(8192, dtype=np.int64)
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


def feed_simple(req, has_beam, input_ids, pos_start):
    n = input_ids.shape[1]
    att = np.ones((1, pos_start + n), dtype=np.int64)
    pos = np.arange(pos_start, pos_start + n, dtype=np.int64).reshape(1, -1)
    return feed_raw(req, has_beam, input_ids, att, pos)


def simple_decode(req, has_beam, prompt_ids, max_tokens):
    req.reset_state()
    pos = 0; gens = []
    l = feed_simple(req, has_beam, prompt_ids, pos); pos += prompt_ids.shape[1]
    nt = int(np.argmax(l[0, -1, :])); gens.append(nt)
    for _ in range(1, max_tokens):
        l = feed_simple(req, has_beam, np.array([[nt]], dtype=np.int64), pos); pos += 1
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


# ---- main ----
print(f"OV {ov.__version__}", flush=True)
tok = AutoTokenizer.from_pretrained(TARGET)
input_ids = tok.encode(PROMPT, return_tensors="np").astype(np.int64)

core = ov.Core()
t_c = core.compile_model(core.read_model(os.path.join(TARGET, "openvino_model.xml")), "GPU")
d_c = core.compile_model(core.read_model(os.path.join(DRAFT, "openvino_model.xml")), "GPU")
t_req = t_c.create_infer_request()
d_req = d_c.create_infer_request()
t_beam = any("beam_idx" in i.get_names() for i in t_c.inputs)
d_beam = any("beam_idx" in i.get_names() for i in d_c.inputs)
t_m = MaskedReq(t_req, t_beam)
d_m = MaskedReq(d_req, d_beam)


for max_toks in [128, 256, 512]:
    print(f"\n--- MAX_TOKENS={max_toks} ---", flush=True)
    # warmup
    simple_decode(t_req, t_beam, input_ids, max_toks)
    spec_decode_v7(t_m, d_m, input_ids, max_toks, K)

    # baseline
    t0 = time.perf_counter()
    g_a = simple_decode(t_req, t_beam, input_ids, max_toks)
    dt = time.perf_counter() - t0
    tps_a = max_toks / dt
    print(f"  baseline: {tps_a:.2f} tok/s  ({dt:.2f}s)")

    # spec
    t0 = time.perf_counter()
    g_sd, acc, drafts = spec_decode_v7(t_m, d_m, input_ids, max_toks, K)
    dt = time.perf_counter() - t0
    tps_sd = max_toks / dt
    print(f"  spec K={K}: {tps_sd:.2f} tok/s  ({dt:.2f}s)  accept={acc/max(drafts,1):.1%}  speedup={tps_sd/tps_a:.2f}x")
    print(f"  match first10: {g_sd[:10] == g_a[:10]}", flush=True)
