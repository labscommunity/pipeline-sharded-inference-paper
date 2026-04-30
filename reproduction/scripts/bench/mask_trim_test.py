"""Test: can we use attention_mask to mask out 'garbage' cache positions
instead of physically trimming KV state?

Scenario:
  Prefill prompt. Feed [first, draft_garbage]. Mask out draft_garbage positions.
  Feed correction with expanded attention_mask and position_id=prompt_len+1.
  Compare correction's output to TRUE correction (where cache is trimmed before feeding).
"""
import os, numpy as np, openvino as ov
from transformers import AutoTokenizer

TARGET = r"C:\cascadia\models\llama-3.1-8b-int4"


def feed(req, input_ids, attention_mask, position_ids):
    req.infer({
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'beam_idx': np.zeros(1, dtype=np.int32),
    })
    return req.get_output_tensor(0).data


tok = AutoTokenizer.from_pretrained(TARGET)
prompt = tok.encode("What is the capital of France?", return_tensors="np").astype(np.int64)
pl = prompt.shape[1]

core = ov.Core()
c = core.compile_model(os.path.join(TARGET, "openvino_model.xml"), "GPU")

# Path A: correct trim. Feed prompt, first, verify drafts, TRIM, feed correction.
r_a = c.create_infer_request()
r_a.reset_state()
l = feed(r_a, prompt, np.ones((1, pl), dtype=np.int64), np.arange(pl, dtype=np.int64).reshape(1, -1))
first = int(np.argmax(l[0, -1, :]))
# Feed first
l = feed(r_a, np.array([[first]], dtype=np.int64),
         np.ones((1, pl + 1), dtype=np.int64),
         np.array([[pl]], dtype=np.int64))
# Now feed 3 "garbage" drafts: [999, 888, 777]
drafts = [999, 888, 777]
l = feed(r_a, np.array([drafts], dtype=np.int64),
         np.ones((1, pl + 1 + 3), dtype=np.int64),
         np.arange(pl + 1, pl + 4, dtype=np.int64).reshape(1, -1))
# Pretend we accepted 1 draft (999). Trim to pl + 1 + 1. Feed correction.
# First trim:
for sv in r_a.query_state():
    cur = np.asarray(sv.state.data)
    if cur.ndim == 4 and cur.shape[2] > pl + 2:
        sv.state = ov.Tensor(cur[:, :, :pl + 2, :].copy())
# Feed correction (just use 555)
l = feed(r_a, np.array([[555]], dtype=np.int64),
         np.ones((1, pl + 3), dtype=np.int64),
         np.array([[pl + 2]], dtype=np.int64))
out_trim = int(np.argmax(l[0, -1, :]))
out_trim_logits = l[0, -1, :5].copy()

# Path B: NO trim. Mask garbage positions in attention_mask instead.
r_b = c.create_infer_request()
r_b.reset_state()
feed(r_b, prompt, np.ones((1, pl), dtype=np.int64), np.arange(pl, dtype=np.int64).reshape(1, -1))
feed(r_b, np.array([[first]], dtype=np.int64),
     np.ones((1, pl + 1), dtype=np.int64),
     np.array([[pl]], dtype=np.int64))
feed(r_b, np.array([drafts], dtype=np.int64),
     np.ones((1, pl + 1 + 3), dtype=np.int64),
     np.arange(pl + 1, pl + 4, dtype=np.int64).reshape(1, -1))
# Cache now: prompt + first + drafts (all 3). Length pl + 4.
# Feed correction with attention_mask masking out drafts[1], drafts[2] (positions pl+2, pl+3).
# position_ids for correction = pl + 2 (as if it's right after drafts[0]=999 accepted).
# attention_mask: allow prompt + first + drafts[0] + correction; mask drafts[1], drafts[2].
attn_mask = np.ones((1, pl + 4 + 1), dtype=np.int64)  # past + current
attn_mask[:, pl + 2] = 0  # drafts[1]
attn_mask[:, pl + 3] = 0  # drafts[2]
l = feed(r_b, np.array([[555]], dtype=np.int64),
         attn_mask,
         np.array([[pl + 2]], dtype=np.int64))
out_mask = int(np.argmax(l[0, -1, :]))
out_mask_logits = l[0, -1, :5].copy()

print(f'Path A (trim): next={out_trim}  ({tok.decode([out_trim])!r})  logits[:5]={out_trim_logits}')
print(f'Path B (mask): next={out_mask}  ({tok.decode([out_mask])!r})  logits[:5]={out_mask_logits}')
print(f'Match: {out_trim == out_mask}')
print(f'Max diff: {np.max(np.abs(out_trim_logits - out_mask_logits)):.4f}')
