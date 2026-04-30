#!/usr/bin/env python3
"""Export Llama 3.1 8B INT4 KV-cached OV shards — v5: canonical genai convention.

v5 changes vs v3:
  1. Graph input convention matches optimum-intel's: (input_ids, attention_mask,
     position_ids) instead of (input_ids, cos, sin). cos/sin computed internally
     from position_ids.
  2. SDPA uses an explicit boolean-like attention mask built from attention_mask
     + the in-graph KV length (not a hand-crafted torch.triu).
  3. After apply_make_stateful_transformation, inject a `beam_idx` Parameter and
     Gather(ReadValue, beam_idx, axis=0) on each KV ReadValue. This is what
     optimum-intel's `fuse_cache_reorder` does — it unlocks OV GPU plugin's
     `IndirectKVCache` transformation (added in OV 2025.1).
  4. Optionally apply `ov::pass::SDPAToPagedAttention` to swap generic SDPA for
     OV's fused PagedAttention primitive (the big ~40% kernel gain on GPU since
     OV 2025.1).

Graph inputs (per stage, post-make-stateful):
  - embed/full: (input_ids, attention_mask, position_ids, beam_idx)
  - middle/head: (hidden_states, attention_mask, position_ids, beam_idx)
  (KV is internal state via ReadValue/Assign.)
"""

import argparse
import gc
import glob
import json
import math
import os
import sys
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_stage_plan(num_layers, num_stages):
    base = num_layers // num_stages
    remainder = num_layers % num_stages
    stages = []
    offset = 0
    for i in range(num_stages):
        count = base + (1 if i < remainder else 0)
        stages.append({
            "stage": i, "layer_start": offset, "layer_end": offset + count,
            "has_embed": (i == 0), "has_head": (i == num_stages - 1),
        })
        offset += count
    return stages


# ---------------------------------------------------------------------------
# Internal rotary — computed from position_ids (v5: canonical convention)
# ---------------------------------------------------------------------------

class TracedRotaryEmbedding(nn.Module):
    """Rotary from position_ids. FP32-accurate trig, cast to hidden's dtype.
    Traces into a compact graph that OV's RoPEFusion can recognize."""
    def __init__(self, head_dim, rope_theta=500000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids, target_dtype):
        # position_ids: [bsz, seq_len] int64
        bsz, seq_len = position_ids.shape
        inv_freq_expanded = self.inv_freq[None, None, :].expand(bsz, seq_len, -1)
        freqs = position_ids[:, :, None].float() * inv_freq_expanded
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(target_dtype)
        sin = emb.sin().to(target_dtype)
        return cos, sin


def apply_rotary(q, k, cos, sin):
    """Apply rotary to q, k. cos, sin: [bsz, seq_len, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    half = q.shape[-1] // 2

    def rotate_half(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# SDPA-based layer forward with internal rotary + graph-provided causal mask
# ---------------------------------------------------------------------------

def cached_layer_forward_sdpa(layer, hidden_states, cos, sin, causal_mask,
                              past_key, past_value,
                              num_heads, num_kv_heads, head_dim):
    """One decoder layer, SDPA attention, KV-cached.
    causal_mask: [1, 1, seq_len, past_seq_len + seq_len] with 0 allowed / -inf blocked."""
    bsz, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)

    q = layer.self_attn.q_proj(hidden_states)
    k = layer.self_attn.k_proj(hidden_states)
    v = layer.self_attn.v_proj(hidden_states)

    q = q.view(bsz, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    q, k = apply_rotary(q, k, cos, sin)

    # Append to cache
    k = torch.cat([past_key, k], dim=2)
    v = torch.cat([past_value, v], dim=2)

    # Expand KV for GQA (num_kv_heads -> num_heads via num_kv_groups)
    k_exp = k[:, :, None, :, :].expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    k_exp = k_exp.reshape(bsz, num_heads, -1, head_dim)
    v_exp = v[:, :, None, :, :].expand(bsz, num_kv_heads, num_kv_groups, -1, head_dim)
    v_exp = v_exp.reshape(bsz, num_heads, -1, head_dim)

    # SDPA with externally-provided causal_mask (built once per forward pass
    # in the wrapper, not here, so OV can see a single Add+Compare chain).
    # OV converts this pattern into a fused Attention op.
    attn_output = F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=causal_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
    )

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seq_len, -1)
    attn_output = layer.self_attn.o_proj(attn_output)

    hidden_states = residual + attn_output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, k, v


# ---------------------------------------------------------------------------
# Stage wrappers with internal rotary
# ---------------------------------------------------------------------------

class _BaseStage(nn.Module):
    def __init__(self, layers, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary = TracedRotaryEmbedding(head_dim, rope_theta)

    def _build_causal_mask(self, attention_mask, seq_len, past_kv_len, dtype):
        """attention_mask: [bsz, past_kv_len + seq_len] with 1 for allowed, 0 for masked.
        Returns [1, 1, seq_len, past_kv_len + seq_len] float mask with 0.0/-inf."""
        full_seq_len = past_kv_len + seq_len
        # Causal triangle: for query q (abs pos = past_kv_len + q), keys up to that position allowed.
        q_pos = torch.arange(seq_len, device=attention_mask.device).unsqueeze(-1) + past_kv_len
        k_pos = torch.arange(full_seq_len, device=attention_mask.device).unsqueeze(0)
        causal_allow = (k_pos <= q_pos).to(dtype)  # [seq_len, full_seq_len]
        # Combine with attention_mask (padding): 1 allowed, 0 masked
        pad_allow = attention_mask.unsqueeze(1).to(dtype)  # [bsz, 1, full_seq_len]
        allow = causal_allow.unsqueeze(0) * pad_allow  # [bsz, seq_len, full_seq_len]
        # Convert to additive mask
        mask = (1.0 - allow) * torch.finfo(dtype).min
        return mask.unsqueeze(1)  # [bsz, 1, seq_len, full_seq_len]

    def _run_layers(self, hidden_states, attention_mask, position_ids, past_kv):
        cos, sin = self.rotary(position_ids, hidden_states.dtype)
        bsz, seq_len = position_ids.shape
        past_kv_len = past_kv[0].shape[2]  # first past_key, dim 2 is seq
        causal_mask = self._build_causal_mask(attention_mask, seq_len, past_kv_len, hidden_states.dtype)
        present_kv = []
        for idx, layer in enumerate(self.layers):
            hidden_states, pk, pv = cached_layer_forward_sdpa(
                layer, hidden_states, cos, sin, causal_mask,
                past_kv[idx * 2], past_kv[idx * 2 + 1],
                self.num_heads, self.num_kv_heads, self.head_dim,
            )
            present_kv.extend([pk, pv])
        return hidden_states, present_kv


class CachedEmbedStageWrapper(_BaseStage):
    def __init__(self, embed_tokens, layers, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
        self.embed_tokens = embed_tokens

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_kv = self._run_layers(hidden_states, attention_mask, position_ids, past_kv)
        return (hidden_states, *present_kv)


class CachedMiddleStageWrapper(_BaseStage):
    def forward(self, hidden_states, attention_mask, position_ids, *past_kv):
        hidden_states, present_kv = self._run_layers(hidden_states, attention_mask, position_ids, past_kv)
        return (hidden_states, *present_kv)


class CachedHeadStageWrapper(_BaseStage):
    def __init__(self, layers, norm, lm_head, num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states, attention_mask, position_ids, *past_kv):
        hidden_states, present_kv = self._run_layers(hidden_states, attention_mask, position_ids, past_kv)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return (logits, *present_kv)


class CachedFullStageWrapper(_BaseStage):
    def __init__(self, embed_tokens, layers, norm, lm_head,
                 num_heads, num_kv_heads, head_dim, rope_theta):
        super().__init__(layers, num_heads, num_kv_heads, head_dim, rope_theta)
        self.embed_tokens = embed_tokens
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, position_ids, *past_kv):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states, present_kv = self._run_layers(hidden_states, attention_mask, position_ids, past_kv)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return (logits, *present_kv)


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_stage_weights(model_dir, layer_start, layer_end, has_embed, has_head):
    from safetensors import safe_open
    needed = []
    for i in range(layer_start, layer_end):
        needed.append(f"model.layers.{i}.")
    if has_embed:
        needed.append("model.embed_tokens.")
    if has_head:
        needed.append("model.norm.")
        needed.append("lm_head.")
    state_dict = {}
    for sf in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(sf, framework="pt", device="cpu") as f:
            for key in f.keys():
                if any(key.startswith(p) for p in needed):
                    state_dict[key] = f.get_tensor(key)
    return state_dict


def build_wrapper(config, state_dict, layer_start, layer_end, has_embed, has_head, rope_theta):
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer, LlamaRMSNorm,
    )
    config._attn_implementation = "eager"  # we pull q_proj/k_proj/v_proj/o_proj/mlp etc. manually

    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    layers = []
    for i in range(layer_start, layer_end):
        layer = LlamaDecoderLayer(config, layer_idx=i)
        prefix = f"model.layers.{i}."
        layer_keys = [k for k in list(state_dict.keys()) if k.startswith(prefix)]
        layer_sd = {k.removeprefix(prefix): state_dict[k] for k in layer_keys}
        layer.load_state_dict(layer_sd, strict=False)
        layer.eval()
        for k in layer_keys:
            del state_dict[k]
        del layer_sd
        gc.collect()
        layers.append(layer)
        if (i - layer_start) % 4 == 3:
            print(f"    built layer {i} (+{(i-layer_start+1)}/{layer_end-layer_start})", flush=True)

    if has_embed and has_head:
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        embed.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})
        del state_dict["model.embed_tokens.weight"]
        norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": state_dict["model.norm.weight"]})
        del state_dict["model.norm.weight"]
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})
        del state_dict["lm_head.weight"]
        wrapper = CachedFullStageWrapper(embed, layers, norm, lm_head,
                                         num_heads, num_kv_heads, head_dim, rope_theta)
    elif has_embed:
        embed = nn.Embedding(config.vocab_size, config.hidden_size)
        embed.load_state_dict({"weight": state_dict["model.embed_tokens.weight"]})
        del state_dict["model.embed_tokens.weight"]
        wrapper = CachedEmbedStageWrapper(embed, layers,
                                          num_heads, num_kv_heads, head_dim, rope_theta)
    elif has_head:
        norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.load_state_dict({"weight": state_dict["model.norm.weight"]})
        del state_dict["model.norm.weight"]
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict({"weight": state_dict["lm_head.weight"]})
        del state_dict["lm_head.weight"]
        wrapper = CachedHeadStageWrapper(layers, norm, lm_head,
                                         num_heads, num_kv_heads, head_dim, rope_theta)
    else:
        wrapper = CachedMiddleStageWrapper(layers, num_heads, num_kv_heads, head_dim, rope_theta)

    wrapper.eval()
    gc.collect()
    return wrapper


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def export_single_stage(model_dir, output_dir, stage_plan, config, quantization, rope_theta):
    import openvino as ov

    stage_idx = stage_plan["stage"]
    layer_start = stage_plan["layer_start"]
    layer_end = stage_plan["layer_end"]
    has_embed = stage_plan["has_embed"]
    has_head = stage_plan["has_head"]
    num_layers = layer_end - layer_start
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    print(f"\n{'=' * 60}", flush=True)
    print(f"STAGE {stage_idx}: layers [{layer_start}, {layer_end})"
          f" | embed={has_embed} | head={has_head}", flush=True)
    print(f"{'=' * 60}", flush=True)

    # 1. Load weights
    print("  Loading weights...", flush=True)
    state_dict = load_stage_weights(
        model_dir, layer_start, layer_end, has_embed, has_head)
    weight_mb = sum(t.nbytes for t in state_dict.values()) / 1e6
    print(f"  {len(state_dict)} tensors ({weight_mb:.0f} MB)", flush=True)

    # 2. Build wrapper
    print("  Building wrapper...", flush=True)
    wrapper = build_wrapper(
        config, state_dict, layer_start, layer_end, has_embed, has_head, rope_theta)
    del state_dict
    gc.collect()

    # 3. Example inputs (past_seq=1 to avoid zero-dim). v5 uses canonical inputs:
    #    (input_ids/hidden, attention_mask, position_ids, *past_kv)
    seq_len = 4
    past_seq = 1
    full_seq_len = past_seq + seq_len

    if has_embed:
        main_input = torch.randint(0, config.vocab_size, (1, seq_len))
    else:
        main_input = torch.randn(1, seq_len, config.hidden_size)

    # attention_mask covers past + current, all ones for trace (no padding)
    attention_mask = torch.ones(1, full_seq_len, dtype=torch.long)
    # position_ids for the current (non-past) tokens
    position_ids = torch.arange(past_seq, past_seq + seq_len, dtype=torch.long).unsqueeze(0)

    past_kv_tensors = []
    for _ in range(num_layers):
        past_kv_tensors.append(torch.randn(1, num_kv_heads, past_seq, head_dim))
        past_kv_tensors.append(torch.randn(1, num_kv_heads, past_seq, head_dim))

    example_inputs = (main_input, attention_mask, position_ids, *past_kv_tensors)

    # 4. Trace
    print("  torch.jit.trace...", flush=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example_inputs)
    print("  Trace OK", flush=True)

    with torch.no_grad():
        traced_out = traced(*example_inputs)
    if isinstance(traced_out, tuple):
        print(f"  Traced outputs: {len(traced_out)} "
              f"(hidden={traced_out[0].shape}, + {len(traced_out)-1} KV tensors)", flush=True)

    del wrapper
    gc.collect()

    # 5. Convert to OpenVINO
    print("  ov.convert_model...", flush=True)
    ov_model = ov.convert_model(traced, example_input=example_inputs)
    print(f"  OV model: {len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs", flush=True)

    del traced
    gc.collect()

    # 6. Name inputs/outputs and set dynamic shapes. v5 ordering:
    #    [0] input_ids/hidden_states, [1] attention_mask, [2] position_ids, [3..] past_kv pairs
    kv_input_names = {}
    for i, inp in enumerate(ov_model.inputs):
        shape = inp.partial_shape
        if i == 0:
            name = "input_ids" if has_embed else "hidden_states"
            if len(shape) >= 2:
                shape[1] = -1
        elif i == 1:
            name = "attention_mask"
            if len(shape) >= 2:
                shape[1] = -1
        elif i == 2:
            name = "position_ids"
            if len(shape) >= 2:
                shape[1] = -1
        else:
            kv_idx = i - 3
            layer_local = kv_idx // 2
            is_value = kv_idx % 2 == 1
            kv_type = "value" if is_value else "key"
            name = f"past_key_values.{layer_local}.{kv_type}"
            kv_input_names[name] = i
            if len(shape) >= 3:
                shape[2] = -1
        inp.node.set_partial_shape(shape)
        inp.set_names({name})

    kv_output_names = {}
    for i, out in enumerate(ov_model.outputs):
        if i == 0:
            name = "logits" if has_head else "hidden_states"
        else:
            kv_idx = i - 1
            layer_local = kv_idx // 2
            is_value = kv_idx % 2 == 1
            kv_type = "value" if is_value else "key"
            name = f"present.{layer_local}.{kv_type}"
            kv_output_names[name] = i
        out.set_names({name})

    # 7. Make stateful
    print("  apply_make_stateful_transformation (%d KV pairs)..." % num_layers, flush=True)
    from openvino.passes import Manager
    pairs = []
    for layer_local in range(num_layers):
        kin = f"past_key_values.{layer_local}.key"
        vin = f"past_key_values.{layer_local}.value"
        kout = f"present.{layer_local}.key"
        vout = f"present.{layer_local}.value"
        pairs.append((kin, kout))
        pairs.append((vin, vout))
    pair_map = dict(pairs)

    from openvino._offline_transformations import apply_make_stateful_transformation

    # V5_MODE: "none" | "beam_idx" | "paged_attention"
    # paged_attention_transformation expects a model with explicit past_kv inputs
    # (pre-stateful); beam_idx path expects a stateful model so we can intercept
    # each ReadValue with a Gather. Order is mode-dependent.
    mode = os.environ.get("V5_MODE", "beam_idx").lower()
    apply_beam_idx = (mode == "beam_idx")
    apply_paged = (mode == "paged_attention")
    print(f"  V5_MODE={mode} (beam_idx={apply_beam_idx}, paged={apply_paged})", flush=True)

    if not apply_paged:
        apply_make_stateful_transformation(ov_model, pair_map)
        print(f"  Stateful: {len(ov_model.inputs)} inputs, {len(ov_model.outputs)} outputs", flush=True)

    # 7b. fuse_cache_reorder: add beam_idx Parameter + Gather(ReadValue, beam_idx).
    # Required for OV GPU plugin's IndirectKVCache transformation (2025.1+).
    if apply_beam_idx:
        try:
            import openvino.opset13 as opset
            from openvino import PartialShape, Type
            beam_idx_param = opset.parameter(PartialShape([-1]), Type.i32, name="beam_idx")
            beam_idx_param.set_friendly_name("beam_idx")
            beam_idx_param.output(0).set_names({"beam_idx"})
            read_values = [n for n in ov_model.get_ops() if n.get_type_name() == "ReadValue"]
            axis_const = opset.constant(0, Type.i32)
            rv_count = 0
            for rv in read_values:
                rv_out = rv.output(0)
                # opset.gather returns a Node in 2026.1 — use its output(0)
                gather_node = opset.gather(rv_out, beam_idx_param, axis_const)
                gather_out = gather_node.output(0)
                for target_input in list(rv_out.get_target_inputs()):
                    if target_input.get_node() is gather_node:
                        continue
                    target_input.replace_source_output(gather_out)
                rv_count += 1
            # Add beam_idx Parameter BEFORE validate
            ov_model.add_parameters([beam_idx_param])
            ov_model.validate_nodes_and_infer_types()
            print(f"  fuse_cache_reorder: beam_idx Parameter + Gather on {rv_count} ReadValue ops", flush=True)
        except Exception as e:
            print(f"  fuse_cache_reorder FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # 7c. paged_attention_transformation — swaps SDPA for OV's PagedAttention op.
    # This is the same transform openvino_genai applies internally. Lives in
    # openvino._offline_transformations (NOT in openvino.passes on 2026.1).
    #
    # IMPORTANT: the transformation adds new Parameters (past_lens,
    # subsequence_begins, block_indices, block_indices_begins, max_context_len,
    # and possibly key/value_cache.N.*) but does NOT register them with the
    # Model. We have to scan for unregistered Parameter ops and add them.
    if apply_paged:
        try:
            from openvino._offline_transformations import paged_attention_transformation
            n_before = len(ov_model.inputs)
            registered = {id(p) for p in ov_model.get_parameters()}
            paged_attention_transformation(ov_model)
            # Find new Parameter ops in the post-transform graph
            new_params = []
            for node in ov_model.get_ops():
                if node.get_type_name() == "Parameter" and id(node) not in registered:
                    new_params.append(node)
            if new_params:
                ov_model.add_parameters(new_params)
                ov_model.validate_nodes_and_infer_types()
                print(f"  paged_attention: registered {len(new_params)} new Parameters", flush=True)
            n_after = len(ov_model.inputs)
            print(f"  paged_attention_transformation applied. Inputs: {n_before} -> {n_after}", flush=True)
            for inp in ov_model.inputs:
                print(f"    input: {sorted(inp.get_names())} shape={inp.get_partial_shape()}", flush=True)
        except Exception as e:
            print(f"  paged_attention_transformation FAILED: {e}", flush=True)
            import traceback; traceback.print_exc()

    # 8. INT4 compression
    if quantization in ("int4", "int4_asym"):
        print(f"  nncf {quantization} compression...", flush=True)
        import nncf
        mode = nncf.CompressWeightsMode.INT4_SYM if quantization == "int4" else nncf.CompressWeightsMode.INT4_ASYM
        ov_model = nncf.compress_weights(
            ov_model,
            mode=mode,
            group_size=128,
            ratio=1.0,
            all_layers=True,
        )
        print("  Quantization OK", flush=True)
    elif quantization == "int8":
        import nncf
        ov_model = nncf.compress_weights(ov_model, mode=nncf.CompressWeightsMode.INT8_ASYM)

    # 9. Save
    stage_dir = os.path.join(output_dir, f"stage_{stage_idx}")
    os.makedirs(stage_dir, exist_ok=True)
    xml_path = os.path.join(stage_dir, "openvino_model.xml")
    import openvino as ov
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    size_mb = os.path.getsize(xml_path.replace(".xml", ".bin")) / 1e6
    print(f"  Saved: {stage_dir} ({size_mb:.0f} MB)", flush=True)

    # 10. Metadata
    meta = {
        "stage": stage_idx,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "has_embed": has_embed,
        "has_head": has_head,
        "quantization": quantization,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "num_layers_total": config.num_hidden_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "stateful": True,
        "rope_theta": rope_theta,
        "inputs": "input_ids/hidden_states, attention_mask, position_ids, beam_idx (KV cache is internal state)",
        "export_version": "v5_canonical_inputs_paged_attention",
    }
    with open(os.path.join(stage_dir, "stage_config.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 11. Verify (shape-only sanity check via CPU)
    print(f"  Verifying stage {stage_idx}...", flush=True)
    try:
        core = ov.Core()
        reloaded = core.read_model(xml_path)
        compiled = core.compile_model(reloaded, "CPU")
        request = compiled.create_infer_request()

        # Build cos/sin of shape [1, seq_len, head_dim]
        def _cs(seq_len_):
            inv = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
            pos = np.arange(seq_len_, dtype=np.float32)
            freqs = np.outer(pos, inv)
            emb = np.concatenate([freqs, freqs], axis=-1)
            return np.cos(emb)[None].astype(np.float32), np.sin(emb)[None].astype(np.float32)

        prefill_len = 4
        ids = np.random.randint(0, config.vocab_size, (1, prefill_len)).astype(np.int64) if has_embed else np.random.randn(1, prefill_len, config.hidden_size).astype(np.float32)
        cos_p, sin_p = _cs(prefill_len)
        request.infer({0: ids, 1: cos_p, 2: sin_p})
        out = request.get_output_tensor(0).data
        print(f"  Prefill OK: shape={out.shape}", flush=True)
        print(f"  Stage {stage_idx} verification PASSED", flush=True)
    except Exception as ve:
        print(f"  Verification step errored (non-fatal — shard is saved): {ve}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export INT4 KV-cached shards v2 (SDPA + internal rotary)")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument("--quantization", default="int4",
                        choices=["fp16", "int4", "int4_asym", "int8"])
    parser.add_argument("--layer-split", type=int, default=None)
    parser.add_argument("--default-dtype", default="fp16", choices=["fp16", "fp32"])
    args = parser.parse_args()

    if args.default_dtype == "fp16":
        torch.set_default_dtype(torch.float16)

    from transformers import AutoConfig, AutoTokenizer

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model_dir = args.model_dir
    config = AutoConfig.from_pretrained(model_dir)
    rope_theta = float(getattr(config, "rope_theta", 500000.0))
    print(f"Model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"kv_heads={config.num_key_value_heads}, rope_theta={rope_theta}", flush=True)

    if args.layer_split and args.num_stages == 2:
        n = config.num_hidden_layers
        plan = [
            {"stage": 0, "layer_start": 0, "layer_end": args.layer_split,
             "has_embed": True, "has_head": False},
            {"stage": 1, "layer_start": args.layer_split, "layer_end": n,
             "has_embed": False, "has_head": True},
        ]
    else:
        plan = compute_stage_plan(config.num_hidden_layers, args.num_stages)
    print(f"\nStage plan ({args.num_stages} stages):", flush=True)
    for s in plan:
        parts = []
        if s["has_embed"]:
            parts.append("embed")
        parts.append(f"layers {s['layer_start']}-{s['layer_end'] - 1}")
        if s["has_head"]:
            parts.append("norm+head")
        print(f"  Stage {s['stage']}: {' + '.join(parts)}", flush=True)

    # Tokenizer
    print("\nSaving tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

    pipeline_meta = {
        "model_id": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "num_stages": args.num_stages,
        "num_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "vocab_size": config.vocab_size,
        "quantization": args.quantization,
        "stateful": True,
        "rope_theta": rope_theta,
        "export_version": "v5_canonical_inputs_paged_attention",
        "stages": plan,
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w") as f:
        json.dump(pipeline_meta, f, indent=2)

    stages_to_export = [plan[args.stage]] if args.stage is not None else plan

    for stage_plan in stages_to_export:
        try:
            export_single_stage(
                model_dir, output_dir, stage_plan, config, args.quantization, rope_theta)
        except Exception as e:
            print(f"\n  ERROR stage {stage_plan['stage']}: {e}", flush=True)
            traceback.print_exc()
            print("  Continuing...", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print("EXPORT COMPLETE", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)
    total_mb = 0
    for s in plan:
        stage_dir = os.path.join(output_dir, f"stage_{s['stage']}")
        bin_path = os.path.join(stage_dir, "openvino_model.bin")
        if os.path.exists(bin_path):
            mb = os.path.getsize(bin_path) / 1e6
            total_mb += mb
            print(f"  Stage {s['stage']}: {mb:.0f} MB", flush=True)
        else:
            print(f"  Stage {s['stage']}: NOT EXPORTED", flush=True)
    print(f"  Total: {total_mb:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
