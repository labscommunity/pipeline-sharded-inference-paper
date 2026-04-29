"""Load a model shard for pipeline-parallel inference.

Supports two loading modes:

1. **Selective PyTorch loading** (from HuggingFace safetensors):
   Only loads the assigned layers' weights from safetensors files.
   Memory usage is proportional to the layer count, not the full model.
   Uses the Petals-style selective loading approach.

2. **Full OpenVINO loading** (from OpenVINO IR):
   Loads the full compiled model. Useful for single-node benchmarks.
   Uses mmap so unaccessed weights don't consume physical RAM.

The default is mode 1 (selective PyTorch) for pipeline-parallel inference.
Mode 2 is used for monolithic single-node benchmarks.

Export a model for use with this loader:
    # For selective loading (recommended for pipeline):
    huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir /path/to/model

    # For full OV loading (single-node benchmark):
    optimum-cli export openvino --model meta-llama/Llama-3.1-8B-Instruct \
        --weight-format int4 --output /path/to/model

IMPORTANT: This module is the primary target for optimization. Future work
includes true OpenVINO IR splitting for per-node shards.
"""

import gc
import glob
import json
import re
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn


# Maps HuggingFace model_type to the attribute paths for key components.
# Most modern decoder models follow the same pattern.
_MODEL_STRUCTURE = {
    # model_type: (layers_attr, embed_attr, norm_attr, head_attr, rotary_attr)
    "llama":   ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "mistral": ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "qwen2":   ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "phi3":    ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "gemma":   ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "gemma2":  ("model.layers", "model.embed_tokens", "model.norm", "lm_head", "model.rotary_emb"),
    "phi":     ("model.layers", "model.embed_tokens", "model.final_layernorm", "lm_head", None),
    "starcoder2": ("model.layers", "model.embed_tokens", "model.norm", "lm_head", None),
    # Hybrid Transformer-Mamba architectures (multimodal with nested text decoder)
    "qwen3_5":   ("model.language_model.layers", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head", None),
    "qwen3_5_text": ("model.layers", "model.embed_tokens", "model.norm", "lm_head", None),
    # Kimi K2.5 — multimodal wrapper around DeepSeek-V3 language backbone.
    # Uses trust_remote_code; classes in modeling_deepseek.py on HF Hub.
    # Layer 0 is dense, layers 1-60 are MoE (384 experts, 8 active per token).
    # Expert weights are INT4 QAT (weight_packed + weight_scale + weight_shape).
    "kimi_k25": ("language_model.model.layers", "language_model.model.embed_tokens", "language_model.model.norm", "language_model.lm_head", None),
}


class INT4Linear(torch.nn.Module):
    """Linear layer that stores INT4 packed weights and dequantizes on the fly.

    Memory: ~8 MB per projection (vs 59 MB for BF16 nn.Linear).
    Forward: dequantizes to BF16 → matmul → discard dequanted.
    """

    def __init__(self, weight_packed: torch.Tensor, weight_scale: torch.Tensor):
        super().__init__()
        self.register_buffer("weight_packed", weight_packed)
        self.register_buffer("weight_scale", weight_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # K2.5 compressed-tensors INT4 format (pack-quantized, num_bits=4,
        # group_size=32, symmetric). Unpack with STRIDED layout: nibble i of
        # each packed int32 occupies columns [i::8]. Subtract 8 for the
        # symmetric zero-point mapping (stored as unsigned [0..15] → signed
        # [-8..7]).
        dev = x.device
        wp = self.weight_packed.to(dev) if self.weight_packed.device != dev else self.weight_packed
        ws = self.weight_scale.to(dev) if self.weight_scale.device != dev else self.weight_scale
        rows, cols_packed = wp.shape
        cols = cols_packed * 8
        unpacked = torch.empty((rows, cols), dtype=torch.int32, device=dev)
        for i in range(8):
            unpacked[:, i::8] = (wp >> (4 * i)) & 0xF
        # Symmetric signed: subtract 8 to map unsigned [0..15] → signed [-8..7].
        signed = unpacked.to(torch.float32) - 8.0
        # Group-wise scale: reshape [rows, groups, group_size], multiply by scale.
        group_size = cols // ws.shape[1]
        signed = signed.view(rows, ws.shape[1], group_size)
        weight = (signed * ws.to(torch.float32).unsqueeze(-1)).view(rows, cols).to(x.dtype)
        return x @ weight.T


def _dequant_int4_state_dict(sd: dict) -> dict:
    """Dequantize INT4 packed weights in a state dict.

    Converts {key}.weight_packed (int32) + {key}.weight_scale (bf16) →
    {key}.weight (bf16). Non-packed tensors are passed through as-is.
    This is needed for Kimi K2.5's INT4 QAT format where expert weights
    are stored as compressed-tensors style packed int4.
    """
    out = {}
    packed_bases = set()
    for k in sd:
        if k.endswith(".weight_packed"):
            packed_bases.add(k.removesuffix(".weight_packed"))

    for k, v in sd.items():
        if k.endswith(".weight_packed"):
            base = k.removesuffix(".weight_packed")
            scale_key = f"{base}.weight_scale"
            if scale_key not in sd:
                continue
            packed, scale = v, sd[scale_key]
            # K2.5 uses compressed-tensors pack-quantized format with STRIDED
            # nibble layout: nibble i of each int32 occupies columns [i::8].
            # Symmetric signed int4 is stored as unsigned [0..15] with
            # zero-point=8: value = raw_nibble - 8.
            rows, cols_packed = packed.shape
            cols = cols_packed * 8
            unpacked = torch.empty((rows, cols), dtype=torch.int32)
            for i in range(8):
                unpacked[:, i::8] = (packed >> (4 * i)) & 0xF
            signed = unpacked.to(torch.float32) - 8.0  # symmetric zero-point
            group_size = cols // scale.shape[1]
            signed = signed.view(rows, scale.shape[1], group_size)
            weight = (signed * scale.to(torch.float32).unsqueeze(-1)) \
                .view(rows, cols).to(torch.bfloat16)
            out[f"{base}.weight"] = weight
        elif k.endswith(".weight_scale") or k.endswith(".weight_shape"):
            continue  # consumed by packed handler
        else:
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
    return out


def _resolve_attr(obj, dotted_path):
    """Resolve a dotted attribute path like 'model.layers' on an object."""
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve_attr_safe(obj, dotted_path):
    """Like _resolve_attr but returns None if any part is missing."""
    try:
        return _resolve_attr(obj, dotted_path)
    except AttributeError:
        return None


class ModelShard:
    """A model shard — a subset of transformer layers.

    Loads only the assigned layer range from safetensors files, so memory
    usage is proportional to the number of layers, not the full model size.
    """

    def __init__(
        self,
        model_path: str,
        layer_start: int,
        layer_end: int,
        device: str = "CPU",
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ):
        self.model_path = model_path
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.device = device
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

        self.tokenizer = None
        self._loaded = False

        # Model components (populated during load)
        self._embed = None       # Embedding layer (first stage only)
        self._layers = None      # nn.ModuleList of decoder layers
        self._norm = None        # Final layer norm (last stage only)
        self._head = None        # LM head (last stage only)
        self._rotary_emb = None  # Rotary embedding (if applicable)
        self._config = None      # HuggingFace model config
        self._model_structure = None  # Tuple of attribute paths

    def load(self):
        """Load only the assigned layers from safetensors."""
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        from safetensors import safe_open

        model_dir = Path(self.model_path)
        print(f"Loading shard from {model_dir} "
              f"(layers {self.layer_start}-{self.layer_end - 1}, "
              f"first={self.is_first_stage}, last={self.is_last_stage})...")

        # Load config and tokenizer. trust_remote_code is needed for
        # models like Kimi K2.5 that define custom modeling code on HF Hub.
        self._config = AutoConfig.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), trust_remote_code=True,
        )

        # Determine model structure from model_type
        model_type = getattr(self._config, "model_type", "llama")
        self._model_structure = _MODEL_STRUCTURE.get(model_type)
        if self._model_structure is None:
            print(f"  Warning: unknown model_type '{model_type}', "
                  f"falling back to llama structure")
            self._model_structure = _MODEL_STRUCTURE["llama"]

        layers_attr, embed_attr, norm_attr, head_attr, rotary_attr = \
            self._model_structure

        # For multimodal wrappers (e.g., kimi_k25), unwrap to the text config
        # so layer constructors see hidden_size, num_attention_heads, etc.
        if hasattr(self._config, "text_config"):
            self._text_config = self._config.text_config
        else:
            self._text_config = self._config

        # Force eager attention for compatibility
        self._config._attn_implementation = "eager"
        self._text_config._attn_implementation = "eager"

        # Determine which weight prefixes we need
        needed_prefixes = []
        for i in range(self.layer_start, self.layer_end):
            needed_prefixes.append(f"{layers_attr.replace('.', '.')}.{i}.")
        if self.is_first_stage:
            needed_prefixes.append(f"{embed_attr}.")
        if self.is_last_stage:
            needed_prefixes.append(f"{norm_attr}.")
            needed_prefixes.append(f"{head_attr}.")

        # Convert attribute paths to weight prefixes
        # e.g., "model.layers" -> "model.layers.0.", "model.embed_tokens" -> "model.embed_tokens."
        weight_prefixes = []
        for i in range(self.layer_start, self.layer_end):
            weight_prefixes.append(f"{layers_attr}.{i}.")
        if self.is_first_stage:
            weight_prefixes.append(f"{embed_attr}.")
        if self.is_last_stage:
            weight_prefixes.append(f"{norm_attr}.")
            weight_prefixes.append(f"{head_attr}.")

        # Selectively load tensors from safetensors
        safetensor_files = sorted(glob.glob(str(model_dir / "*.safetensors")))
        if not safetensor_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_dir}. "
                f"Download the model with: huggingface-cli download <model_id> "
                f"--local-dir {model_dir}"
            )

        state_dict = {}
        for sf_path in safetensor_files:
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if any(key.startswith(p) for p in weight_prefixes):
                        state_dict[key] = f.get_tensor(key)

        weight_mb = sum(t.nbytes for t in state_dict.values()) / 1e6
        print(f"  Loaded {len(state_dict)} tensors ({weight_mb:.0f} MB)")

        # Ensure custom modeling code (e.g., modeling_deepseek.py for K2.5)
        # is importable. Trust-remote-code models ship .py files alongside
        # safetensors; add the model dir to sys.path so _get_component_classes
        # can find them.
        import sys as _sys
        model_dir_str = str(model_dir)
        if model_dir_str not in _sys.path:
            _sys.path.insert(0, model_dir_str)

        # Build just the components we need (no full model creation)
        self._build_components(state_dict)

        del state_dict
        gc.collect()

        self._loaded = True
        print(f"  Shard ready. Components loaded: "
              f"embed={self._embed is not None}, "
              f"layers={len(self._layers)}, "
              f"norm={self._norm is not None}, "
              f"head={self._head is not None}")

    def _build_components(self, state_dict):
        """Build model components from config and loaded weights.

        Creates each component directly from the model's module classes
        rather than extracting from a full model, to ensure correct
        initialization (especially attention implementation).
        """
        config = self._config
        layer_config = self._text_config
        layers_attr, embed_attr, norm_attr, head_attr, rotary_attr = \
            self._model_structure

        # Dynamically import the model's decoder layer and norm classes
        # based on model_type. These are needed for direct construction.
        layer_class, norm_class, rotary_class = \
            self._get_component_classes(config.model_type)

        # Build decoder layers directly (not via full model).
        # For models with custom quantization (e.g., kimi_k25 INT4 QAT),
        # skip .half() to avoid corrupting packed weight tensors.
        # For models with huge MoE layers (e.g., K2.5: 384 experts per layer),
        # constructing parameters in float32 then overwriting is too expensive.
        # Use accelerate's init_empty_weights to create on meta device first,
        # then populate with actual (possibly INT4-packed) weights.
        skip_half = config.model_type in ("kimi_k25",)
        use_meta_init = config.model_type in ("kimi_k25",)

        self._layers = nn.ModuleList()
        for i in range(self.layer_start, self.layer_end):
            prefix = f"{layers_attr}.{i}."
            layer_sd = {
                k.removeprefix(prefix): v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }

            if use_meta_init:
                from accelerate import init_empty_weights
                from accelerate.utils import set_module_tensor_to_device
                with init_empty_weights():
                    layer = layer_class(layer_config, layer_idx=i)

                target_device = self.device
                use_int4_linear = target_device in ("xpu", "cuda")

                packed_bases = {
                    k.removesuffix(".weight_packed")
                    for k in layer_sd if k.endswith(".weight_packed")
                }

                if use_int4_linear:
                    # GPU path: INT4Linear keeps packed weights on device,
                    # dequants on-the-fly during forward. ~10 GB per layer.
                    for base in packed_bases:
                        pk = layer_sd[f"{base}.weight_packed"]
                        sc = layer_sd[f"{base}.weight_scale"]
                        q_linear = INT4Linear(
                            pk.to(target_device), sc.to(target_device),
                        )
                        parts = base.split(".")
                        parent = layer
                        for p in parts[:-1]:
                            parent = getattr(parent, p) if not p.isdigit() \
                                else parent[int(p)]
                        setattr(parent, parts[-1], q_linear)
                else:
                    # CPU path: pre-dequant INT4 → BF16 for fast matmul.
                    # Uses more memory (~75 GB/layer) but 40x faster forward.
                    dequanted = _dequant_int4_state_dict(layer_sd)
                    for key, tensor in dequanted.items():
                        try:
                            set_module_tensor_to_device(
                                layer, key, "cpu", value=tensor,
                            )
                        except (AttributeError, ValueError):
                            pass

                # Load non-packed tensors (attention, norms, gate, etc.)
                for key, tensor in layer_sd.items():
                    if any(key.startswith(b + ".weight_")
                           for b in packed_bases):
                        continue
                    try:
                        val = tensor.to(torch.bfloat16) \
                            if tensor.is_floating_point() else tensor
                        set_module_tensor_to_device(
                            layer, key, target_device, value=val,
                        )
                    except (AttributeError, ValueError):
                        pass
            else:
                layer = layer_class(layer_config, layer_idx=i)
                layer.load_state_dict(layer_sd, strict=False)

            layer.eval()
            if not skip_half:
                layer.half()
            self._layers.append(layer)

        # Embedding (first stage only)
        if self.is_first_stage:
            if use_meta_init:
                from accelerate import init_empty_weights
                from accelerate.utils import set_module_tensor_to_device
                with init_empty_weights():
                    self._embed = nn.Embedding(
                        layer_config.vocab_size, layer_config.hidden_size,
                    )
                embed_sd = {
                    k.removeprefix(f"{embed_attr}."): v
                    for k, v in state_dict.items()
                    if k.startswith(f"{embed_attr}.")
                }
                for key, tensor in embed_sd.items():
                    set_module_tensor_to_device(
                        self._embed, key, "cpu", value=tensor,
                    )
            else:
                self._embed = nn.Embedding(
                    layer_config.vocab_size, layer_config.hidden_size,
                )
                embed_sd = {
                    k.removeprefix(f"{embed_attr}."): v
                    for k, v in state_dict.items()
                    if k.startswith(f"{embed_attr}.")
                }
                self._embed.load_state_dict(embed_sd, strict=False)
            self._embed.eval()
            if not skip_half:
                self._embed.half()

        # Final norm + LM head (last stage only)
        if self.is_last_stage:
            if norm_class is not None:
                self._norm = norm_class(
                    layer_config.hidden_size,
                    eps=getattr(layer_config, "rms_norm_eps",
                                getattr(layer_config, "layer_norm_eps", 1e-6)),
                )
            else:
                self._norm = nn.LayerNorm(layer_config.hidden_size)

            norm_sd = {
                k.removeprefix(f"{norm_attr}."): v
                for k, v in state_dict.items()
                if k.startswith(f"{norm_attr}.")
            }
            self._norm.load_state_dict(norm_sd, strict=False)
            self._norm.eval()
            if not skip_half:
                self._norm.half()

            self._head = nn.Linear(
                layer_config.hidden_size, layer_config.vocab_size, bias=False
            )
            head_sd = {
                k.removeprefix(f"{head_attr}."): v
                for k, v in state_dict.items()
                if k.startswith(f"{head_attr}.")
            }
            self._head.load_state_dict(head_sd, strict=False)
            self._head.eval()
            if not skip_half:
                self._head.half()

        # Rotary embedding (no learned weights, config-derived only)
        if rotary_class is not None:
            self._rotary_emb = rotary_class(config=config)

    @staticmethod
    def _get_component_classes(model_type):
        """Get the decoder layer, norm, and rotary classes for a model type."""
        # Import the correct classes based on model architecture
        try:
            if model_type in ("llama", "mistral"):
                from transformers.models.llama.modeling_llama import (
                    LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
                )
                return LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding
            elif model_type == "qwen2":
                from transformers.models.qwen2.modeling_qwen2 import (
                    Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding,
                )
                return Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding
            elif model_type in ("gemma", "gemma2"):
                from transformers.models.gemma.modeling_gemma import (
                    GemmaDecoderLayer, GemmaRMSNorm, GemmaRotaryEmbedding,
                )
                return GemmaDecoderLayer, GemmaRMSNorm, GemmaRotaryEmbedding
            elif model_type == "phi3":
                from transformers.models.phi3.modeling_phi3 import (
                    Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding,
                )
                return Phi3DecoderLayer, Phi3RMSNorm, Phi3RotaryEmbedding
            elif model_type == "kimi_k25":
                # K2.5's modeling_deepseek.py uses relative imports, so we
                # need to import via the transformers_modules package that
                # HF auto-creates. Try the hyphenated cache dir first.
                try:
                    import modeling_deepseek as mod
                    return mod.DeepseekV3DecoderLayer, mod.DeepseekV3RMSNorm, None
                except ImportError:
                    import sys as _sys
                    _hf_modules = "/home/ubuntu/.cache/huggingface/modules"
                    if Path(_hf_modules).exists() and _hf_modules not in _sys.path:
                        _sys.path.insert(0, _hf_modules)
                    from transformers_modules.kimi_hyphen_k25_hyphen_hf import modeling_deepseek as mod
                    return mod.DeepseekV3DecoderLayer, mod.DeepseekV3RMSNorm, None
        except ImportError:
            pass

        # Fallback: try llama classes (most common)
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding,
        )
        return LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """Run embedding layer. Only called on stage 0.

        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]

        Returns:
            Hidden states after embedding, shape [batch_size, seq_len, hidden_size]
        """
        assert self.is_first_stage, "embed() should only be called on stage 0"
        assert self._embed is not None, "Embedding not loaded"
        with torch.no_grad():
            inputs_tensor = torch.tensor(input_ids, dtype=torch.long)
            hidden_states = self._embed(inputs_tensor)
            return hidden_states.numpy()

    def forward_layers(
        self, hidden_states: np.ndarray, position_ids: np.ndarray = None
    ) -> np.ndarray:
        """Run forward pass through this shard's assigned layers.

        Args:
            hidden_states: Input activations, shape [batch_size, seq_len, hidden_size]
            position_ids: Position IDs for rotary embeddings

        Returns:
            Output activations, same shape as input
        """
        assert self._loaded, "Model not loaded. Call load() first."

        dev = self.device
        with torch.no_grad():
            hs = torch.tensor(hidden_states, device=dev)
            if position_ids is not None:
                pos = torch.tensor(position_ids, device=dev)
            else:
                pos = torch.arange(hs.shape[1], device=dev).unsqueeze(0)

            position_embeddings = None
            if self._rotary_emb is not None:
                position_embeddings = self._rotary_emb(hs, pos)

            seq_len = hs.shape[1]
            causal_mask = torch.full(
                (1, 1, seq_len, seq_len), float("-inf"),
                dtype=hs.dtype, device=dev,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)

            for layer in self._layers:
                if position_embeddings is not None:
                    layer_output = layer(
                        hs, position_embeddings=position_embeddings,
                        attention_mask=causal_mask,
                    )
                else:
                    layer_output = layer(
                        hs, position_ids=pos,
                        attention_mask=causal_mask,
                    )
                hs = layer_output[0]
                if hs.dim() == 2:
                    hs = hs.unsqueeze(0)

            return hs.cpu().numpy()

    def forward_layers_cached(
        self,
        hidden_states: np.ndarray,
        position_ids: np.ndarray = None,
        past_key_values: list = None,
    ) -> tuple:
        """Forward pass with KV cache support.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len] — must reflect absolute position
                          (e.g., [0,1,...,N-1] for prefill, [N] for decode)
            past_key_values: list of per-layer cache objects (None on first call)

        Returns:
            (output_hidden_states as np.ndarray,
             new_past_key_values list)
        """
        assert self._loaded, "Model not loaded. Call load() first."

        dev = self.device
        with torch.no_grad():
            hs = torch.tensor(hidden_states, device=dev)
            if position_ids is not None:
                pos = torch.tensor(position_ids, device=dev)
            else:
                pos = torch.arange(hs.shape[1], device=dev).unsqueeze(0)

            seq_len = hs.shape[1]
            past_len = 0
            if past_key_values is not None:
                if hasattr(past_key_values, "get_seq_length"):
                    # DynamicCache indexes by layer_idx. Use the first
                    # layer's actual index for the lookup.
                    first_layer_idx = self._layers[0].self_attn.layer_idx \
                        if hasattr(self._layers[0], "self_attn") and \
                           hasattr(self._layers[0].self_attn, "layer_idx") \
                        else 0
                    past_len = past_key_values.get_seq_length(first_layer_idx)
                elif isinstance(past_key_values, (list, tuple)) and \
                     len(past_key_values) > 0 and past_key_values[0] is not None:
                    pkv = past_key_values[0]
                    past_len = pkv[0].shape[2] if isinstance(pkv, tuple) else 0

            total_len = past_len + seq_len
            causal_mask = torch.full(
                (1, 1, seq_len, total_len), float("-inf"),
                dtype=hs.dtype, device=dev,
            )
            causal_mask[:, :, :, :total_len] = 0
            for i in range(seq_len):
                causal_mask[:, :, i, past_len + i + 1:] = float("-inf")

            from transformers.cache_utils import DynamicCache

            if past_key_values is None:
                past_key_values = DynamicCache()

            new_past = past_key_values
            for idx, layer in enumerate(self._layers):
                layer_output = layer(
                    hs,
                    position_ids=pos,
                    attention_mask=causal_mask,
                    past_key_value=new_past,
                    use_cache=True,
                )
                hs = layer_output[0]
                if hs.dim() == 2:
                    hs = hs.unsqueeze(0)
                # DynamicCache updates in-place; layer_output[1] is
                # the same cache object.

            return hs.cpu().numpy(), new_past

    def lm_head(self, hidden_states: np.ndarray) -> np.ndarray:
        """Run the LM head to get logits. Only called on the last stage.

        Args:
            hidden_states: Final hidden states, shape [batch_size, seq_len, hidden_size]

        Returns:
            Logits, shape [batch_size, seq_len, vocab_size]
        """
        assert self.is_last_stage, "lm_head() should only be called on the last stage"
        assert self._norm is not None, "Norm not loaded"
        assert self._head is not None, "LM head not loaded"
        with torch.no_grad():
            hs = torch.tensor(hidden_states)
            hs = self._norm(hs)
            logits = self._head(hs)
            return logits.numpy()

    def get_hidden_size(self) -> int:
        if self._config:
            return self._config.hidden_size
        return None

    def get_num_layers(self) -> int:
        if self._config:
            return self._config.num_hidden_layers
        return None
