"""Speculative decoding with mask-based KV cache rewind.

Exports two pieces for downstream callers:

  1. `MaskedReq`: wraps an OpenVINO `InferRequest` for a stateful causal LM
     and adds a `rewind(k)` operation that invalidates the last `k` tokens
     of the KV cache by flipping `attention_mask[...] = 0` on future feeds,
     instead of physically trimming state via `query_state()` + `set_state()`.
     See `DISCOVERIES.md#20` for the full rationale; the physical-trim path
     costs ~40 ms per call on Arc iGPU, which is enough to make spec decode
     a net loss.

  2. `spec_decode_greedy()`: a self-contained greedy speculative decoding
     loop. Takes a `MaskedReq` wrapping the target, a `MaskedReq` wrapping
     the draft, a prompt, K, and max_tokens — returns generated token ids
     plus step-level statistics. Output is bit-exact with the target's own
     greedy decode (verified in `scripts/bench_spec_matrix.py`).

Both target and draft must be OpenVINO stateful causal LMs with the
Rainier canonical input convention: `(input_ids, attention_mask, position_ids)`
plus optional `beam_idx` (for GPU `IndirectKVCache` fusion). Shards exported
by `scripts/export_cached_shards_v5.py` work directly; `optimum-intel`
monolithic models do too once wrapped (see `scripts/bench_spec_v7_masked.py`
for the simplest wiring).

This module is the canonical reference implementation for the speedups
documented in `docs/BENCHMARKS.md#speculative-decoding-v7-mask-based-kv-rewind-panther-lake-igpu-ov-20261`:

  - 1.36× over monolithic target (LAN, K=3, 128 tok)
  - 1.55× over monolithic target (LAN, K=3, 512 tok, 90.6% accept)
  - 1.33× over 3-stage sharded target (LAN, K=3, 128 tok)
  - 4.03× over 3-stage sharded target (100 ms/hop WAN, K=10, 128 tok)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import openvino as ov


def _feed_raw(
    req, has_beam: bool, input_ids: np.ndarray, attn_mask: np.ndarray, pos_ids: np.ndarray
) -> np.ndarray:
    fd = {"input_ids": input_ids, "attention_mask": attn_mask, "position_ids": pos_ids}
    if has_beam:
        fd["beam_idx"] = np.zeros(1, dtype=np.int32)
    req.infer(fd)
    return req.get_output_tensor(0).data


class MaskedReq:
    """Wraps an OpenVINO stateful causal LM InferRequest with mask-based rewind.

    Usage:
        req = compiled_model.create_infer_request()
        mr = MaskedReq(req, has_beam=any("beam_idx" in i.get_names() for i in compiled_model.inputs))
        mr.reset()
        logits = mr.feed(prompt_ids)                    # prompt
        logits = mr.feed(np.array([[next_tok]], np.int64))  # one-token extend
        mr.rewind(3)   # drop last 3 emitted tokens from the logical sequence
        logits = mr.feed(np.array([[correction]], np.int64))

    Invariants maintained:
        cache_len: physical length of the underlying KV cache (monotonic, append-only).
        logical_pos: logical sequence length after all rewinds. `position_ids` for the
            next feed starts here.
        valid_mask[i]: 1 if cache position i is "live", 0 if it's a rejected draft
            that has been rewound. Used to build `attention_mask` on every feed.
    """

    __slots__ = ("req", "has_beam", "valid_mask", "cache_len", "logical_pos")

    def __init__(self, req, has_beam: bool, initial_capacity: int = 4096):
        self.req = req
        self.has_beam = has_beam
        self.valid_mask = np.ones(initial_capacity, dtype=np.int64)
        self.cache_len = 0
        self.logical_pos = 0

    def reset(self) -> None:
        """Reset underlying KV state and all tracking counters."""
        self.req.reset_state()
        self.valid_mask[:] = 1
        self.cache_len = 0
        self.logical_pos = 0

    def feed(self, input_ids: np.ndarray) -> np.ndarray:
        """Feed `input_ids` (shape [1, n]) at the current logical position.

        Returns logits of shape [1, n, vocab_size]. All n new cache positions
        are marked valid; cache_len and logical_pos both increase by n.
        """
        n = input_ids.shape[1]
        total = self.cache_len + n
        if total > len(self.valid_mask):
            new_size = max(total * 2, len(self.valid_mask) * 2)
            new_mask = np.ones(new_size, dtype=np.int64)
            new_mask[: len(self.valid_mask)] = self.valid_mask
            self.valid_mask = new_mask

        att = np.empty((1, total), dtype=np.int64)
        att[0, : self.cache_len] = self.valid_mask[: self.cache_len]
        att[0, self.cache_len :] = 1
        pos = np.arange(self.logical_pos, self.logical_pos + n, dtype=np.int64).reshape(1, -1)

        out = _feed_raw(self.req, self.has_beam, input_ids, att, pos)
        self.cache_len += n
        self.logical_pos += n
        return out

    def rewind(self, k: int) -> None:
        """Invalidate the last `k` cache positions (logically rewind by k tokens).

        Physical cache is unchanged; `valid_mask[cache_len-k:cache_len] = 0` so
        future `feed()`s set `attention_mask` to 0 at those indices. `logical_pos`
        decreases by k so subsequent RoPE rotations and position_ids stay coherent.

        Note: memory grows with rewound entries since they stay in physical cache.
        For long generations (> ~2k tokens) a periodic physical compaction may
        be needed; not implemented here.
        """
        if k <= 0:
            return
        self.valid_mask[self.cache_len - k : self.cache_len] = 0
        self.logical_pos -= k


@dataclass
class SpecDecodeStats:
    """Per-run telemetry from `spec_decode_greedy()`."""

    n_steps: int
    total_drafts: int
    total_accepted: int

    @property
    def accept_rate(self) -> float:
        return self.total_accepted / max(self.total_drafts, 1)


def spec_decode_greedy(
    target: MaskedReq,
    draft: MaskedReq,
    prompt_ids: np.ndarray,
    max_tokens: int,
    k: int = 3,
) -> Tuple[List[int], SpecDecodeStats]:
    """Greedy speculative decoding with mask-based rewind.

    Per step (invariant: `target.cache_len` is at the end of all emitted tokens
    EXCEPT the most recent correction, which lives in `prev_correction`; draft
    cache holds the full emitted prefix including that correction):

      1. Draft generates K candidates: drafts[0] from draft's current logit
         (no extra feed), drafts[1..K-1] by feeding drafts[i-1] each step.
         This leaves draft cache advanced by K-1.

      2. Target verifies [prev_correction, drafts[0..K-1]] in one forward pass
         (K+1 tokens). `target.cache_len` grows by K+1.

      3. Compare drafts against target's greedy continuation:
             accepted = longest prefix where drafts[i] == argmax(target_logits[i]).
         Correction = argmax(target_logits[accepted]) if accepted < K else
                      argmax(target_logits[K]).

      4. Emit drafts[:accepted] + [correction] to `gens`.

      5. Rewind target by K-accepted (the rejected drafts). Draft rewind depends
         on whether all K were accepted: if so, feed drafts[K-1] and correction;
         otherwise rewind K-1-accepted and feed correction.

    Returns (gens, stats). Output is bit-exact with the target's own greedy
    decode (argmax of target.feed) given the same prompt — verified in
    `scripts/bench_spec_matrix.py` (max logit diff 0.0000 at each accept
    decision point).
    """
    target.reset()
    draft.reset()

    # Prefill both
    t_logits = target.feed(prompt_ids)
    draft.feed(prompt_ids)

    # Emit first token from target
    first = int(np.argmax(t_logits[0, -1, :]))
    gens: List[int] = [first]
    prev_correction = first

    # Feed first into draft so its next logit is the prediction at prompt_len+1
    d_logits = draft.feed(np.array([[first]], dtype=np.int64))
    d_last_logit = d_logits[0, -1, :].copy()

    stats = SpecDecodeStats(n_steps=0, total_drafts=0, total_accepted=0)

    while len(gens) < max_tokens:
        stats.n_steps += 1

        # 1. Draft K candidates
        drafts = [int(np.argmax(d_last_logit))]
        for i in range(1, k):
            if len(gens) + len(drafts) >= max_tokens:
                break
            d_logits = draft.feed(np.array([[drafts[i - 1]]], dtype=np.int64))
            drafts.append(int(np.argmax(d_logits[0, -1, :])))
        d_advanced = len(drafts) - 1
        stats.total_drafts += len(drafts)

        # 2. Target verify [prev_correction, drafts] in one forward
        verify = np.array([[prev_correction] + drafts], dtype=np.int64)
        t_logits = target.feed(verify)
        t_greedy = np.argmax(t_logits[0], axis=-1)  # shape: [K+1]

        # 3. Acceptance: longest prefix where drafts[i] == t_greedy[i]
        accepted = 0
        for i in range(len(drafts)):
            if int(t_greedy[i]) == drafts[i]:
                accepted += 1
            else:
                break
        stats.total_accepted += accepted

        # 4. Correction = target's preferred continuation at the mismatch point,
        #    or after all drafts if they all matched
        if accepted < len(drafts):
            correction = int(t_greedy[accepted])
        else:
            correction = int(t_greedy[len(drafts)])

        for tk in drafts[:accepted] + [correction]:
            if len(gens) >= max_tokens:
                break
            gens.append(tk)

        # 5. Rewind target: drop the last (K - accepted) cache entries
        target.rewind(len(drafts) - accepted)

        # 5b. Draft rewind depends on whether all drafts were accepted
        if accepted < len(drafts):
            # Rewind draft by (d_advanced - accepted) then feed correction
            draft.rewind(d_advanced - accepted)
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        else:
            # All drafts accepted — draft cache is missing drafts[-1] and correction.
            # Feed drafts[-1] then correction.
            d_logits = draft.feed(np.array([[drafts[-1]]], dtype=np.int64))
            d_logits = draft.feed(np.array([[correction]], dtype=np.int64))
        d_last_logit = d_logits[0, -1, :].copy()
        prev_correction = correction

    return gens[:max_tokens], stats


def make_masked_req_from_model(compiled_model) -> MaskedReq:
    """Convenience: wrap an already-compiled ov.CompiledModel into a MaskedReq."""
    req = compiled_model.create_infer_request()
    has_beam = any("beam_idx" in i.get_names() for i in compiled_model.inputs)
    return MaskedReq(req, has_beam)


__all__ = [
    "MaskedReq",
    "SpecDecodeStats",
    "spec_decode_greedy",
    "make_masked_req_from_model",
]
