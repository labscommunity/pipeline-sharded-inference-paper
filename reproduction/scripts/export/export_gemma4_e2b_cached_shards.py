#!/usr/bin/env python3
"""Export Gemma 4 E2B as stateful KV-cached per-stage OpenVINO IR shards.

Each shard has internal KV cache state (ReadValue/Assign). Caller just
runs infer() and cache accumulates. Call reset_state() between sequences.

Architecture: 35 layers, 8 Q heads, 1 KV head, head_dim=256 (sliding)
or 512 (full). Layers 15-34 share KV from layers 13/14 — we copy the
source weights so every layer computes its own KV independently.

Inputs after make_stateful: (main_input, position_ids)
  - Stage 0: main_input=input_ids [batch, seq] (int64)
  - Stage 1+: main_input=hidden_states_with_pli [batch, seq, dim] (float32)

Usage:
    python export_gemma4_e2b_cached_shards.py --output-dir C:\\cascadia\\shards_e2b_cached --num-stages 2
"""

import argparse, gc, json, math, os, sys, time
import numpy as np


def log(msg):
    print(msg, flush=True)


def apply_patches():
    import openvino.frontend.pytorch.utils as ov_utils
    _orig = ov_utils.torch_tensor_to_ov_const
    def _patched(tensor, shared_memory=False):
        if hasattr(tensor, 'dim') and tensor.dim() == 0:
            tensor = tensor.reshape(1)
        return _orig(tensor, shared_memory)
    ov_utils.torch_tensor_to_ov_const = _patched


def compute_stage_plan(num_layers, num_stages):
    base = num_layers // num_stages
    remainder = num_layers % num_stages
    stages, offset = [], 0
    for i in range(num_stages):
        count = base + (1 if i < remainder else 0)
        stages.append({
            "stage": i, "layer_start": offset, "layer_end": offset + count,
            "has_embed": (i == 0), "has_head": (i == num_stages - 1),
        })
        offset += count
    return stages


# ---------------------------------------------------------------------------
# Rotary embedding (matches Gemma 4's apply_rotary_pos_emb)
#
# We replace HF's `tm.rotary_emb` with a custom traced rotary that produces
# consistent-dtype cos/sin. HF's Gemma3nTextRotaryEmbedding internally mixes
# FP16/FP32 (autocast disabled, then `.to(x.dtype)`), and torch.jit.trace
# captures that mismatch into the OV IR — opset1::MatMul on OV 2026.1 GPU
# rejects mismatched element types (regression vs 2026.0 which auto-promoted).
# Computing inv_freq*pos in FP32 then casting once at the end avoids that.
# ---------------------------------------------------------------------------

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return __import__('torch').cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary on [batch, heads, seq, head_dim] tensors."""
    # cos/sin: [1, seq, head_dim] → unsqueeze at dim 1 for heads
    cos = cos.unsqueeze(1)  # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(1)
    return (q * cos + rotate_half(q) * sin,
            k * cos + rotate_half(k) * sin)


class GemmaTracedRotaryEmbedding(__import__('torch').nn.Module):
    """Trace-friendly rotary for one Gemma 4 layer type (local or global).

    Holds inv_freq as a buffer; forward computes cos/sin in FP32 and casts to
    target_dtype at the end so the entire output graph is consistent dtype.

    Supports the two rope_types Gemma 4 uses:
      - "default": full head_dim rotated (sliding_attention)
      - "proportional": only first partial_rotary_factor*head_dim dims rotated,
        rest pass through (full_attention; matches HF's
        _compute_proportional_rope_parameters which pads inv_freq with zeros for
        the un-rotated tail).
    """
    def __init__(self, head_dim, rope_theta, partial_rotary_factor=1.0):
        import torch
        super().__init__()
        self.head_dim = head_dim
        if partial_rotary_factor < 1.0:
            rope_angles = int(partial_rotary_factor * head_dim // 2)
            inv_freq_rotated = 1.0 / (rope_theta ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / head_dim))
            nope_angles = head_dim // 2 - rope_angles
            if nope_angles > 0:
                inv_freq = torch.cat(
                    [inv_freq_rotated, torch.zeros(nope_angles, dtype=torch.float32)],
                    dim=0,
                )
            else:
                inv_freq = inv_freq_rotated
        else:
            inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, target_dtype):
        import torch
        # position_ids: [bsz, seq_len] int64
        bsz, seq_len = position_ids.shape
        inv_freq_expanded = self.inv_freq[None, None, :].expand(bsz, seq_len, -1)
        freqs = position_ids[:, :, None].float() * inv_freq_expanded
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(target_dtype)
        sin = emb.sin().to(target_dtype)
        return cos, sin


# ---------------------------------------------------------------------------
# Cached layer forward for Gemma 4 E2B
# ---------------------------------------------------------------------------

def cached_gemma4_layer_forward(layer, hidden_states, cos, sin,
                                past_key, past_value,
                                num_heads, num_kv_heads, head_dim,
                                per_layer_input):
    """Manual cached forward for one Gemma 4 E2B decoder layer.

    Handles: input_layernorm, Q/K/V with norms + rotary, KV cache concat,
    GQA expansion (8:1), attention (scaling=1.0), o_proj, post_attention_layernorm,
    pre/post_feedforward_layernorm, MLP, PLI modulation, layer_scalar.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    # 1. Input layernorm + attention
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    # Q, K, V projections with norms
    q = layer.self_attn.q_proj(hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim)
    q = layer.self_attn.q_norm(q)
    q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]

    k = layer.self_attn.k_proj(hidden_states)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    k = layer.self_attn.k_norm(k)
    k = k.transpose(1, 2)

    v = layer.self_attn.v_proj(hidden_states)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim)
    v = layer.self_attn.v_norm(v)
    v = v.transpose(1, 2)

    # Rotary (applied after norms, before KV concat)
    q, k = apply_rope(q, k, cos, sin)

    # KV cache concat
    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)

    # GQA expansion: 1 KV head → 8 Q heads
    k_exp = k.unsqueeze(2).expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v.unsqueeze(2).expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    # Attention (scaling=1.0 for Gemma 4 — Q/K norms handle magnitude)
    attn_weights = torch.matmul(q, k_exp.transpose(2, 3))

    # Causal mask
    full_seq_len = k.shape[2]
    causal_mask = torch.triu(
        torch.full((seq_len, full_seq_len), float("-inf"),
                   device=q.device, dtype=q.dtype),
        diagonal=full_seq_len - seq_len + 1,
    )
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_exp)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    # 2. Post-attention residual
    hidden_states = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    # 3. MLP
    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    # 4. Per-layer input modulation
    if per_layer_input is not None and layer.hidden_size_per_layer_input:
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    # 5. Layer scalar
    hidden_states = hidden_states * layer.layer_scalar

    return hidden_states, k, v


def cached_gemma4_shared_layer_forward(layer, hidden_states, cos, sin,
                                       source_key, source_value,
                                       num_heads, num_kv_heads, head_dim,
                                       per_layer_input):
    """Forward for KV-shared layers. Uses source layer's K/V instead of own.

    Only computes Q projection. K and V come from the source layer's present cache.
    """
    import torch
    import torch.nn.functional as F

    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    # Only Q projection (K/V from source)
    q = layer.self_attn.q_proj(hidden_states)
    q = q.view(bsz, seq_len, num_heads, head_dim)
    q = layer.self_attn.q_norm(q)
    q = q.transpose(1, 2)
    # Apply rotary to Q only
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)
    q = q * cos_u + rotate_half(q) * sin_u

    # Use source layer's accumulated K/V (no concat — already full)
    k = source_key
    v = source_value

    # GQA expansion
    k_exp = k.unsqueeze(2).expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v.unsqueeze(2).expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    # Attention (scaling=1.0)
    attn_weights = torch.matmul(q, k_exp.transpose(2, 3))
    full_seq_len = k.shape[2]
    causal_mask = torch.triu(
        torch.full((seq_len, full_seq_len), float("-inf"),
                   device=q.device, dtype=q.dtype),
        diagonal=full_seq_len - seq_len + 1,
    )
    attn_weights = attn_weights + causal_mask.unsqueeze(0).unsqueeze(0)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v_exp)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    # Post-attention + MLP + PLI (identical to non-shared)
    hidden_states = layer.post_attention_layernorm(attn_output)
    hidden_states = residual + hidden_states

    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    hidden_states = residual + hidden_states

    if per_layer_input is not None and layer.hidden_size_per_layer_input:
        residual = hidden_states
        hidden_states = layer.per_layer_input_gate(hidden_states)
        hidden_states = layer.act_fn(hidden_states)
        hidden_states = hidden_states * per_layer_input
        hidden_states = layer.per_layer_projection(hidden_states)
        hidden_states = layer.post_per_layer_input_norm(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = hidden_states * layer.layer_scalar
    return hidden_states


# ---------------------------------------------------------------------------
# Cached stage wrappers
# ---------------------------------------------------------------------------

def build_cached_wrapper(model, text_config, stage_plan):
    """Build a CachedStageWrapper for one stage."""
    import torch
    import torch.nn as nn

    ls, le = stage_plan["layer_start"], stage_plan["layer_end"]
    has_embed, has_head = stage_plan["has_embed"], stage_plan["has_head"]
    num_total = text_config.num_hidden_layers
    hidden_dim = text_config.hidden_size
    pli_dim = text_config.hidden_size_per_layer_input
    num_layers = le - ls

    tm = model.model.language_model
    stage_layers = list(tm.layers)[ls:le]
    stage_layer_types = text_config.layer_types[ls:le]

    # Map shared layers to their source layer's local index
    first_non_shared = num_total - text_config.num_kv_shared_layers
    prev_types = text_config.layer_types[:first_non_shared]
    is_shared = []
    source_local = []  # local idx of source, or -1 if source is in another stage
    for i in range(num_layers):
        global_idx = ls + i
        if global_idx >= first_non_shared:
            lt = text_config.layer_types[global_idx]
            src_global = len(prev_types) - 1 - prev_types[::-1].index(lt)
            src_local = src_global - ls
            is_shared.append(True)
            source_local.append(src_local if 0 <= src_local < num_layers else -1)
        else:
            is_shared.append(False)
            source_local.append(-1)

    # Identify cross-stage shared sources: source layers in THIS stage whose KV
    # is needed by downstream stages. These stay as non-stateful I/O.
    cross_stage_sources = set()
    for gi in range(le, num_total):
        if gi >= first_non_shared:
            lt = text_config.layer_types[gi]
            src_g = len(prev_types) - 1 - prev_types[::-1].index(lt)
            if ls <= src_g < le:
                cross_stage_sources.add(src_g - ls)  # local index
    cross_stage_sources = sorted(cross_stage_sources)

    # External shared KV: sources from a PREVIOUS stage that this stage needs
    external_shared_sources = []  # list of (layer_type, head_dim) for KV received from upstream
    for i in range(num_layers):
        if is_shared[i] and source_local[i] == -1:
            lt = text_config.layer_types[ls + i]
            hd = text_config.global_head_dim if lt == "full_attention" else text_config.head_dim
            src_global = len(prev_types) - 1 - prev_types[::-1].index(lt)
            key = (src_global, lt, hd)
            if key not in [(s[0], s[1], s[2]) for s in external_shared_sources]:
                external_shared_sources.append(key)
            # Update source_local to point to the external source index (negative)
            ext_idx = next(j for j, s in enumerate(external_shared_sources) if s[0] == src_global)
            source_local[i] = -(ext_idx + 1)  # negative = external

    n_shared = sum(is_shared)
    n_own_kv = sum(not s for s in is_shared)
    log(f"  KV sharing: {n_own_kv} own + {n_shared} shared, "
        f"{len(cross_stage_sources)} cross-stage sources, "
        f"{len(external_shared_sources)} external sources")

    # Head dims per layer in this stage
    head_dims = []
    for lt in stage_layer_types:
        hd = text_config.global_head_dim if lt == "full_attention" else text_config.head_dim
        head_dims.append(hd)

    num_heads = text_config.num_attention_heads
    num_kv_heads = text_config.num_key_value_heads
    downstream_pli_count = num_total - le if not has_head else 0

    # Gemma 4 E2B rotary parameters, read from text_config.rope_parameters dict:
    #   sliding_attention: head_dim, rope_type="default"
    #   full_attention:    global_head_dim, rope_type="proportional",
    #                      partial_rotary_factor=0.25 (only first 25% rotated)
    head_dim_local = text_config.head_dim
    head_dim_global = getattr(text_config, "global_head_dim", head_dim_local)
    rope_params = getattr(text_config, "rope_parameters", None)
    if rope_params:
        rp_local = rope_params.get("sliding_attention", {})
        rp_global = rope_params.get("full_attention", {})
        rope_theta_local = rp_local.get("rope_theta", getattr(text_config, "default_theta", 10000.0))
        rope_theta_global = rp_global.get("rope_theta", rope_theta_local)
        prf_local = rp_local.get("partial_rotary_factor", 1.0)
        prf_global = rp_global.get("partial_rotary_factor", 1.0)
    else:
        rope_theta_global = getattr(text_config, "rope_theta", 1_000_000.0)
        rope_theta_local = getattr(text_config, "rope_local_base_freq", rope_theta_global)
        prf_local = prf_global = 1.0
    log(f"  Rotary: local hd={head_dim_local} theta={rope_theta_local} prf={prf_local} | "
        f"global hd={head_dim_global} theta={rope_theta_global} prf={prf_global}")

    class CachedStageWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            if has_embed:
                self.embed = tm.embed_tokens
                self.embed_pli = tm.embed_tokens_per_layer
                self.pli_proj = tm.per_layer_model_projection
                self.pli_sv = tm.per_layer_model_projection_scale
                self.pli_norm = tm.per_layer_projection_norm
                self.pli_is = tm.per_layer_input_scale
            self.layers = nn.ModuleList(stage_layers)
            self.rotary_local = GemmaTracedRotaryEmbedding(
                head_dim_local, rope_theta_local, partial_rotary_factor=prf_local)
            self.rotary_global = GemmaTracedRotaryEmbedding(
                head_dim_global, rope_theta_global, partial_rotary_factor=prf_global)
            if has_head:
                self.norm = tm.norm
                self.lm_head = model.lm_head
            # Store as attributes for tracing
            self._head_dims = head_dims
            self._layer_types = list(stage_layer_types)
            self._unique_types = list(set(stage_layer_types))
            self._sliding_window = text_config.sliding_window
            self._is_shared = is_shared
            self._source_local = source_local
            self._cross_stage_sources = cross_stage_sources
            self._n_external = len(external_shared_sources)

        def forward(self, main_input, position_ids, *args):
            # args layout: [*external_shared_kv, *own_past_kv]
            # external_shared_kv: pairs of (key, value) for sources in previous stage
            n_ext = self._n_external
            ext_kv = args[:n_ext * 2] if n_ext > 0 else ()
            past_kv = args[n_ext * 2:]
            if has_embed:
                h = self.embed(main_input)
                # PLI for all layers
                raw_pli = self.embed_pli(main_input).reshape(
                    main_input.shape[0], main_input.shape[1], num_total, pli_dim)
                pj = self.pli_proj(h) * self.pli_sv
                pj = pj.reshape(h.shape[0], h.shape[1], num_total, pli_dim)
                pj = self.pli_norm(pj)
                all_pli = (pj + raw_pli) * self.pli_is
                stage_pli = all_pli[:, :, ls:le, :]
                if downstream_pli_count > 0:
                    downstream_pli = all_pli[:, :, le:, :]
            else:
                # Split hidden_states and PLI from concatenated input
                h = main_input[:, :, :hidden_dim]
                pli_flat = main_input[:, :, hidden_dim:]
                n_pli = num_layers + downstream_pli_count
                pli_all = pli_flat.reshape(h.shape[0], h.shape[1], n_pli, pli_dim)
                stage_pli = pli_all[:, :, :num_layers, :]
                if downstream_pli_count > 0:
                    downstream_pli = pli_all[:, :, num_layers:, :]

            # Precompute rotary per unique layer type. Custom traced rotary
            # produces consistent-dtype cos/sin (FP32 trig, then cast to h.dtype),
            # avoiding the FP16/FP32 mismatch that HF's Gemma3nTextRotaryEmbedding
            # bakes into the OV IR.
            rotary_cache = {}
            for lt in self._unique_types:
                if lt == "full_attention":
                    rotary_cache[lt] = self.rotary_global(position_ids, h.dtype)
                else:  # "sliding_attention" or any future variant
                    rotary_cache[lt] = self.rotary_local(position_ids, h.dtype)

            present_kv = []  # only for non-shared layers
            present_kv_by_local = {}  # local_idx → (k, v) for sharing
            kv_input_idx = 0  # index into past_kv (only non-shared)
            for i, layer in enumerate(self.layers):
                lt = self._layer_types[i]
                hd = self._head_dims[i]
                cos, sin = rotary_cache[lt]

                if self._is_shared[i]:
                    src = self._source_local[i]
                    if src >= 0:
                        # Source is in this stage
                        sk, sv = present_kv_by_local[src]
                    else:
                        # Source is in a previous stage (external)
                        ext_idx = -(src + 1)
                        sk, sv = ext_kv[ext_idx * 2], ext_kv[ext_idx * 2 + 1]
                    h = cached_gemma4_shared_layer_forward(
                        layer, h, cos, sin, sk, sv,
                        num_heads, num_kv_heads, hd,
                        per_layer_input=stage_pli[:, :, i, :],
                    )
                else:
                    h, pk, pv = cached_gemma4_layer_forward(
                        layer, h, cos, sin,
                        past_kv[kv_input_idx * 2],
                        past_kv[kv_input_idx * 2 + 1],
                        num_heads, num_kv_heads, hd,
                        per_layer_input=stage_pli[:, :, i, :],
                    )
                    present_kv.extend([pk, pv])
                    present_kv_by_local[i] = (pk, pv)
                    kv_input_idx += 1

            # Collect cross-stage source KV to pass to downstream stages
            cross_kv = []
            for src_local in self._cross_stage_sources:
                sk, sv = present_kv_by_local[src_local]
                cross_kv.extend([sk, sv])

            if has_head:
                h = self.lm_head(self.norm(h))
                return (h, *cross_kv, *present_kv)
            else:
                if downstream_pli_count > 0:
                    pli_out = downstream_pli.reshape(h.shape[0], h.shape[1], -1)
                    out = __import__('torch').cat([h, pli_out], dim=-1)
                else:
                    out = h
                return (out, *cross_kv, *present_kv)

    wrapper = CachedStageWrapper()
    wrapper.eval()
    return wrapper, head_dims, is_shared, cross_stage_sources, external_shared_sources


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def export_stage(model, text_config, stage_plan, output_dir, device_test="CPU"):
    import torch
    import openvino as ov

    idx = stage_plan["stage"]
    ls, le = stage_plan["layer_start"], stage_plan["layer_end"]
    has_embed, has_head = stage_plan["has_embed"], stage_plan["has_head"]
    num_layers = le - ls
    num_total = text_config.num_hidden_layers
    hidden_dim = text_config.hidden_size
    pli_dim = text_config.hidden_size_per_layer_input
    num_kv_heads = text_config.num_key_value_heads
    downstream_pli_count = num_total - le if not has_head else 0

    log(f"\n{'='*60}")
    log(f"STAGE {idx}: layers [{ls}, {le}) | embed={has_embed} | head={has_head}")
    log(f"{'='*60}")

    (wrapper, head_dims, is_shared_list,
     cross_stage_sources, external_shared_sources) = build_cached_wrapper(
        model, text_config, stage_plan)
    log("  Wrapper built")

    # Non-shared layers that have own KV (will become stateful)
    own_kv_head_dims = [hd for hd, sh in zip(head_dims, is_shared_list) if not sh]
    # Exclude cross-stage sources from stateful (they stay as regular I/O)
    non_shared_local_indices = [i for i, sh in enumerate(is_shared_list) if not sh]
    stateful_kv_dims = []
    cross_stage_kv_dims = []
    stateful_kv_map = []  # maps stateful index → own_kv index
    cross_kv_map = []     # maps cross-stage index → own_kv index
    for own_idx, local_idx in enumerate(non_shared_local_indices):
        if local_idx in cross_stage_sources:
            cross_stage_kv_dims.append(own_kv_head_dims[own_idx])
            cross_kv_map.append(own_idx)
        else:
            stateful_kv_dims.append(own_kv_head_dims[own_idx])
            stateful_kv_map.append(own_idx)
    n_stateful = len(stateful_kv_dims)
    n_cross = len(cross_stage_kv_dims)
    n_external = len(external_shared_sources)
    log(f"  KV: {n_stateful} stateful, {n_cross} cross-stage out, {n_external} external in")

    # Create example inputs: (main_input, position_ids, *ext_kv, *own_past_kv)
    seq_len = 4
    past_seq = 1

    if has_embed:
        main_input = torch.randint(0, text_config.vocab_size, (1, seq_len))
    else:
        input_dim = hidden_dim + (num_layers + downstream_pli_count) * pli_dim
        main_input = torch.randn(1, seq_len, input_dim, dtype=torch.float32)

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)

    ext_kv = []
    for _, _, hd in external_shared_sources:
        ext_kv.append(torch.randn(1, num_kv_heads, past_seq, hd))
        ext_kv.append(torch.randn(1, num_kv_heads, past_seq, hd))

    own_past_kv = []
    for hd in own_kv_head_dims:
        own_past_kv.append(torch.randn(1, num_kv_heads, past_seq, hd))
        own_past_kv.append(torch.randn(1, num_kv_heads, past_seq, hd))

    example_inputs = (main_input, position_ids, *ext_kv, *own_past_kv)

    # Trace
    log("  Tracing...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs, check_trace=False)
    log("  Trace OK")
    del wrapper; gc.collect()

    # Convert to OV
    log("  Converting to OV...")
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    del traced; gc.collect()

    log(f"  OV: {len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs")

    # Inputs: (main, pos, *ext_kv, *own_past_kv)
    n_own_kv = len(own_kv_head_dims)
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        if i == 0:
            name = "input_ids" if has_embed else "hidden_states"
            if len(shape) >= 2: shape[1] = -1
        elif i == 1:
            name = "position_ids"
            if len(shape) >= 2: shape[1] = -1
        elif i < 2 + n_external * 2:
            ei = (i - 2) // 2
            kt = "value" if (i - 2) % 2 else "key"
            name = f"external_kv.{ei}.{kt}"
            if len(shape) >= 3: shape[2] = -1
        else:
            oi = (i - 2 - n_external * 2) // 2
            kt = "value" if (i - 2 - n_external * 2) % 2 else "key"
            name = f"past_key_values.{oi}.{kt}"
            if len(shape) >= 3: shape[2] = -1
        inp.node.set_partial_shape(shape)
        inp.set_names({name})

    # Outputs: (main_out, *cross_kv, *own_present_kv)
    for i, out in enumerate(ov_model.outputs):
        if i == 0:
            name = "logits" if has_head else "hidden_states"
        elif i < 1 + n_cross * 2:
            ci = (i - 1) // 2
            kt = "value" if (i - 1) % 2 else "key"
            name = f"cross_kv.{ci}.{kt}"
        else:
            oi = (i - 1 - n_cross * 2) // 2
            kt = "value" if (i - 1 - n_cross * 2) % 2 else "key"
            name = f"present.{oi}.{kt}"
        out.set_names({name})
    ov_model.validate_nodes_and_infer_types()

    # Make stateful: pair past_key_values.N ↔ present.N for ALL own KV
    log(f"  apply_make_stateful_transformation ({n_own_kv} KV pairs)...")
    from openvino._offline_transformations import apply_make_stateful_transformation
    kv_pairs = {}
    for oi in range(n_own_kv):
        for kt in ["key", "value"]:
            kv_pairs[f"past_key_values.{oi}.{kt}"] = f"present.{oi}.{kt}"
    if kv_pairs:
        apply_make_stateful_transformation(ov_model, kv_pairs)
    # After this: past_key_values/present pairs become ReadValue/Assign state.
    # cross_kv outputs survive (not paired with any input).
    # external_kv inputs survive (not paired with any output).
    ov_model.validate_nodes_and_infer_types()
    log(f"  Stateful: {len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs")

    # Save (FP32 — INT4 doesn't work for per-stage graphs)
    stage_dir = os.path.join(output_dir, f"stage_{idx}")
    os.makedirs(stage_dir, exist_ok=True)
    xml_path = os.path.join(stage_dir, "openvino_model.xml")
    ov.save_model(ov_model, xml_path)
    bin_path = os.path.join(stage_dir, "openvino_model.bin")
    size_mb = os.path.getsize(bin_path) / (1024**2) if os.path.exists(bin_path) else 0
    log(f"  Saved: {stage_dir} ({size_mb:.0f} MB)")

    # Metadata
    meta = {
        "stage": idx, "layer_start": ls, "layer_end": le,
        "has_embed": has_embed, "has_head": has_head,
        "num_layers_total": num_total,
        "hidden_size": hidden_dim,
        "vocab_size": text_config.vocab_size,
        "num_kv_heads": num_kv_heads,
        "num_attention_heads": text_config.num_attention_heads,
        "head_dims": own_kv_head_dims,
        "model_type": "gemma4_e2b",
        "stateful": True,
        "rotary_internal": True,
        "pli_dim": pli_dim,
        "downstream_pli_count": downstream_pli_count,
        "n_cross_stage_kv": n_cross,
        "cross_stage_kv_dims": cross_stage_kv_dims,
        "n_external_kv": n_external,
        "external_kv_dims": [hd for _, _, hd in external_shared_sources],
    }
    with open(os.path.join(stage_dir, "stage_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Verify
    log(f"  Verifying on {device_test}...")
    try:
        core = ov.Core()
        comp = core.compile_model(ov_model, device_test)
        req = comp.create_infer_request()

        # Reset KV state using actual shapes
        req.reset_state()
        for sv in req.query_state():
            shape = list(sv.state.shape)
            shape[0] = 1; shape[2] = 0
            sv.state = ov.Tensor(np.zeros(shape, dtype=np.float32))

        # Build input dict with external KV if needed
        pf_seq = 3
        inputs = {}
        inp_idx = 0
        if has_embed:
            inputs[inp_idx] = np.array([[1, 2, 3]], dtype=np.int64)
        else:
            input_dim = hidden_dim + (num_layers + downstream_pli_count) * pli_dim
            inputs[inp_idx] = np.random.randn(1, pf_seq, input_dim).astype(np.float32)
        inp_idx += 1
        inputs[inp_idx] = np.array([[0, 1, 2]], dtype=np.int64)
        inp_idx += 1
        # External KV (empty for prefill)
        for _, _, hd in external_shared_sources:
            inputs[inp_idx] = np.zeros((1, num_kv_heads, pf_seq, hd), dtype=np.float32)
            inp_idx += 1
            inputs[inp_idx] = np.zeros((1, num_kv_heads, pf_seq, hd), dtype=np.float32)
            inp_idx += 1
        result = req.infer(inputs)
        out = result[comp.output(0)]
        log(f"  Prefill OK: shape={out.shape}, n_outputs={len(result)}")

        # Decode
        inputs2 = {}
        inp_idx = 0
        if has_embed:
            inputs2[inp_idx] = np.array([[4]], dtype=np.int64)
        else:
            inputs2[inp_idx] = np.random.randn(1, 1, input_dim).astype(np.float32)
        inp_idx += 1
        inputs2[inp_idx] = np.array([[3]], dtype=np.int64)
        inp_idx += 1
        for _, _, hd in external_shared_sources:
            inputs2[inp_idx] = np.zeros((1, num_kv_heads, pf_seq + 1, hd), dtype=np.float32)
            inp_idx += 1
            inputs2[inp_idx] = np.zeros((1, num_kv_heads, pf_seq + 1, hd), dtype=np.float32)
            inp_idx += 1
        result2 = req.infer(inputs2)
        out2 = result2[comp.output(0)]
        log(f"  Decode OK: shape={out2.shape}")
        log(f"  Stage {idx} PASSED")
    except Exception as e:
        log(f"  Verification FAILED: {str(e)[:300]}")

    del ov_model; gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--log-file", default=None)
    args = parser.parse_args()

    if args.log_file:
        f = open(args.log_file, "w", buffering=1, encoding="utf-8")
        sys.stdout = f
        sys.stderr = f

    apply_patches()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    model_id = "google/gemma-4-E2B-it"
    os.environ["HF_TOKEN"] = os.environ["HF_TOKEN"]

    log("Loading config...")
    tc = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
    log(f"  {tc.num_hidden_layers} layers, hidden={tc.hidden_size}, "
        f"heads={tc.num_attention_heads}, kv_heads={tc.num_key_value_heads}")

    plan = compute_stage_plan(tc.num_hidden_layers, args.num_stages)
    for p in plan:
        log(f"  Stage {p['stage']}: layers [{p['layer_start']}, {p['layer_end']})")

    log("Loading model (float32)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32,
        trust_remote_code=True, low_cpu_mem_usage=True,
        attn_implementation="eager")
    model.eval()
    fixed = 0
    for n, b in model.named_buffers():
        if b.dim() == 0:
            parts = n.split('.')
            obj = model
            for p in parts[:-1]:
                obj = getattr(obj, p)
            setattr(obj, parts[-1], b.reshape(1))
            fixed += 1
    log(f"  Loaded + patched {fixed} scalars in {time.time()-t0:.0f}s")

    # Save tokenizer
    tok_dir = os.path.join(args.output_dir, "tokenizer")
    if not os.path.exists(tok_dir):
        os.makedirs(tok_dir, exist_ok=True)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tok.save_pretrained(tok_dir)
        tc_path = os.path.join(tok_dir, "tokenizer_config.json")
        with open(tc_path) as ff:
            tc_json = json.load(ff)
        if isinstance(tc_json.get("extra_special_tokens"), list):
            tc_json["extra_special_tokens"] = {}
            with open(tc_path, "w") as ff:
                json.dump(tc_json, ff, indent=2)
        log(f"  Tokenizer saved: {tok_dir}")

    if args.stage is not None:
        sp = [p for p in plan if p["stage"] == args.stage][0]
        export_stage(model, tc, sp, args.output_dir, args.device)
    else:
        for sp in plan:
            export_stage(model, tc, sp, args.output_dir, args.device)

    log("\nDone!")
    if args.log_file:
        sys.stdout.close()


if __name__ == "__main__":
    main()
