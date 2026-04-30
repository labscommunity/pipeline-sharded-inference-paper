#!/usr/bin/env python3
"""Self-contained distributed pipeline node.

Runs as either coordinator (stage 0) or worker (stage 1+).
Supports both stateful (KV-cached) and non-stateful inference.
Embeds TCP relay protocol and OV shard inference — no external package deps
beyond openvino, transformers, numpy.

Usage:
    # Worker (stage 1, middle, stateful):
    python distributed_node.py worker \
        --stage-index 1 --shard-dir C:\cascadia\shards_cached\stage_1 \
        --listen-port 9100 --downstream-host 192.168.86.35 --downstream-port 9100 \
        --device GPU --stateful

    # Worker (stage 2, last, stateful):
    python distributed_node.py worker \
        --stage-index 2 --shard-dir C:\cascadia\shards_cached\stage_2 \
        --listen-port 9100 --device GPU --last-stage --stateful

    # Coordinator (stage 0, stateful):
    python distributed_node.py coordinator \
        --shard-dir C:\cascadia\shards_cached\stage_0 \
        --tokenizer-dir C:\cascadia\shards_cached\tokenizer \
        --downstream-host 192.168.86.32 --downstream-port 9100 \
        --device GPU --stateful --prompt "What is the capital of France?" --max-tokens 20
"""

import argparse
import json
import os
import socket
import struct
import time

import numpy as np
import openvino as ov


# ── TCP Activation Relay Protocol ─────────────────────────────────────────
# 24-byte header: payload_len, dtype, dim0, dim1, dim2, stream_id (all uint32 big-endian)
# stream_id enables multi-stream pipeline micro-batching.

DTYPE_MAP = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.int32}
DTYPE_REVERSE = {v: k for k, v in DTYPE_MAP.items()}
HEADER_SIZE = 24
RECV_BUFFER = 65536


def _recv_exact(sock, num_bytes):
    """Receive exactly num_bytes. Uses pre-allocated bytearray for zero-copy."""
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    received = 0
    while received < num_bytes:
        n = sock.recv_into(view[received:], min(RECV_BUFFER, num_bytes - received))
        if n == 0:
            raise ConnectionError("Socket closed during recv")
        received += n
    return bytes(buf)


_SPARSE_K = 0  # Top-K sparse activation (0 = disabled)


def _set_sparse_k(k):
    global _SPARSE_K
    _SPARSE_K = k


def sparsify_activation(tensor, k):
    """Keep only top-K values by magnitude, zero the rest. In-place for speed."""
    flat = tensor.reshape(-1)
    if k >= len(flat) or k <= 0:
        return tensor
    # Find the K-th largest absolute value as threshold
    abs_vals = np.abs(flat)
    threshold = np.partition(abs_vals, -k)[-k]
    mask = abs_vals >= threshold
    sparse = np.where(mask, flat, 0.0).reshape(tensor.shape).astype(np.float32)
    return sparse


def compress_activation(tensor, method):
    """Compress activation tensor before sending. Returns (compressed, original_dtype)."""
    if method == "fp16":
        return tensor.astype(np.float16), tensor.dtype
    elif method == "int8":
        # Per-tensor symmetric quantization
        absmax = np.abs(tensor).max()
        scale = absmax / 127.0 if absmax > 0 else 1.0
        quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
        # Pack scale as 4-byte float prepended to data
        return quantized, scale
    else:
        return tensor, None


def decompress_activation(tensor, method, scale=None):
    """Decompress activation tensor after receiving."""
    if method == "fp16":
        return tensor.astype(np.float32)
    elif method == "int8":
        return tensor.astype(np.float32) * scale
    else:
        return tensor


# Global config — set by CLI args
_COMPRESS_METHOD = "none"
_ADD_LATENCY_MS = 0.0  # Simulated one-way network latency (RTT = 2x)


def _set_compress_method(method):
    global _COMPRESS_METHOD
    _COMPRESS_METHOD = method


def _set_latency(ms):
    global _ADD_LATENCY_MS
    _ADD_LATENCY_MS = ms


def send_tensor(sock, tensor, stream_id=0):
    # Apply sparse top-K (only to float hidden_states, not token ids or control msgs)
    if _SPARSE_K > 0 and tensor.dtype in (np.float32, np.float16) and tensor.size > _SPARSE_K:
        tensor = sparsify_activation(tensor, _SPARSE_K)

    # Apply activation compression (only to float hidden_states, not token ids)
    scale_bytes = b""
    if _COMPRESS_METHOD != "none" and tensor.dtype in (np.float32, np.float16):
        if _COMPRESS_METHOD == "int8":
            tensor, scale = compress_activation(tensor, "int8")
            scale_bytes = struct.pack(">f", scale)
        else:
            tensor, _ = compress_activation(tensor, _COMPRESS_METHOD)

    tensor = np.ascontiguousarray(tensor)
    dtype_code = DTYPE_REVERSE.get(tensor.dtype.type, 0)
    shape = list(tensor.shape)
    while len(shape) < 3:
        shape.insert(0, 1)
    nbytes = tensor.nbytes
    header = struct.pack(">IIIIII", nbytes + len(scale_bytes), dtype_code,
                         shape[0], shape[1], shape[2], stream_id)
    # Simulate WAN latency
    if _ADD_LATENCY_MS > 0:
        time.sleep(_ADD_LATENCY_MS / 1000.0)
    t0 = time.perf_counter()
    sock.sendall(header)
    if scale_bytes:
        sock.sendall(scale_bytes)
    sock.sendall(tensor.data)
    return (time.perf_counter() - t0) * 1000 + _ADD_LATENCY_MS


def recv_tensor(sock):
    t0 = time.perf_counter()
    header = _recv_exact(sock, HEADER_SIZE)
    payload_len, dtype_code, d0, d1, d2, stream_id = struct.unpack(">IIIIII", header)
    payload = _recv_exact(sock, payload_len)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    dtype = DTYPE_MAP.get(dtype_code, np.float32)

    # Handle int8 decompression (scale is prepended as 4-byte float)
    if _COMPRESS_METHOD == "int8" and dtype == np.int8:
        scale = struct.unpack(">f", payload[:4])[0]
        tensor = np.frombuffer(payload[4:], dtype=np.int8).reshape(d0, d1, d2)
        tensor = decompress_activation(tensor, "int8", scale)
    elif _COMPRESS_METHOD == "fp16" and dtype == np.float16:
        tensor = np.frombuffer(payload, dtype=np.float16).reshape(d0, d1, d2)
        tensor = decompress_activation(tensor, "fp16")
    else:
        tensor = np.frombuffer(payload, dtype=dtype).reshape(d0, d1, d2)

    return tensor, elapsed_ms, stream_id


# ── Rotary Embedding (pure numpy) ─────────────────────────────────────────

def precompute_cos_sin(head_dim, seq_len, rope_theta=500000.0):
    """Compute rotary cos/sin for positions [0, seq_len). Pure numpy."""
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)  # [seq_len, head_dim/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [seq_len, head_dim]
    cos = np.cos(emb)[np.newaxis, :, :].astype(np.float32)  # [1, seq_len, head_dim]
    sin = np.sin(emb)[np.newaxis, :, :].astype(np.float32)
    return cos, sin


def precompute_cos_sin_at(head_dim, position, rope_theta=500000.0):
    """Compute rotary cos/sin for a single position. For KV-cache decode steps."""
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    freqs = position * inv_freq  # [head_dim/2]
    emb = np.concatenate([freqs, freqs])  # [head_dim]
    cos = np.cos(emb).reshape(1, 1, -1).astype(np.float32)  # [1, 1, head_dim]
    sin = np.sin(emb).reshape(1, 1, -1).astype(np.float32)
    return cos, sin


# ── Early Exit Head ────────────────────────────────────────────────────────

class ExitHead:
    """Lightweight norm + lm_head projection for early exit predictions."""

    def __init__(self, path, eps=1e-5):
        data = np.load(path)
        self.norm_weight = data["norm_weight"].astype(np.float32)  # [hidden_size]
        self.lm_head_weight = data["lm_head_weight"].astype(np.float32)  # [vocab, hidden]
        self.eps = eps
        print(f"  Exit head loaded: norm={self.norm_weight.shape}, "
              f"lm_head={self.lm_head_weight.shape}")

    def predict(self, hidden_states):
        """Apply RMSNorm + lm_head to hidden states. Returns (token_id, confidence).

        hidden_states: [1, seq, hidden_size] or [1, 1, hidden_size]
        """
        h = hidden_states[0, -1, :]  # Last position [hidden_size]
        # RMSNorm
        rms = np.sqrt(np.mean(h ** 2) + self.eps)
        h_normed = (h / rms) * self.norm_weight
        # Linear projection
        logits = self.lm_head_weight @ h_normed  # [vocab_size]
        # Softmax confidence (numerically stable)
        logits_shifted = logits - logits.max()
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum()
        token_id = int(np.argmax(probs))
        confidence = float(probs[token_id])
        return token_id, confidence


# ── OV Shard Inference ─────────────────────────────────────────────────────

def load_stage_meta(shard_dir):
    """Load stage metadata from stage_config.json."""
    meta_path = os.path.join(shard_dir, "stage_config.json")
    with open(meta_path) as f:
        return json.load(f)


def load_shard(shard_dir, device, stateful=False):
    """Load and compile an OV IR shard. Returns (compiled, request, meta).

    If stateful=True, creates an InferRequest for stateful KV-cache inference.
    """
    xml_path = os.path.join(shard_dir, "openvino_model.xml")
    core = ov.Core()
    model = core.read_model(xml_path)
    compiled = core.compile_model(model, device)
    meta = load_stage_meta(shard_dir)

    request = None
    if stateful:
        request = compiled.create_infer_request()

    print(f"  Shard loaded on {device} (stateful={stateful}): {xml_path}")
    return compiled, request, meta


def reset_kv_cache(request, meta):
    """Reset KV cache state to empty for a new generation."""
    request.reset_state()
    for state_var in request.query_state():
        # Use the state's own shape to determine dims (handles variable head_dim)
        shape = list(state_var.state.shape)
        # Set batch=1, seq=0, keep kv_heads and head_dim from the state shape
        shape[0] = 1  # batch
        shape[2] = 0  # seq (empty cache)
        state_var.state = ov.Tensor(np.zeros(shape, dtype=np.float32))


def truncate_kv_cache(request, target_len):
    """Truncate KV cache to target_len positions (for speculation rollback)."""
    for state_var in request.query_state():
        current = np.array(state_var.state)
        # KV cache can be [1, num_heads, seq, head_dim] or [1, seq, num_heads, head_dim]
        if len(current.shape) == 4:
            # Find the sequence dimension (not 0=batch, not the smallest two)
            seq_dim = 2  # Default: [batch, heads, seq, head_dim]
            if current.shape[1] > current.shape[2]:
                seq_dim = 2
            elif current.shape[2] > current.shape[1]:
                seq_dim = 1 if current.shape[1] < current.shape[3] else 2
            if current.shape[seq_dim] > target_len:
                slices = [slice(None)] * 4
                slices[seq_dim] = slice(None, target_len)
                truncated = np.ascontiguousarray(current[tuple(slices)])
                state_var.state = ov.Tensor(truncated)
        elif len(current.shape) == 3 and current.shape[1] > target_len:
            truncated = np.ascontiguousarray(current[:, :target_len, :])
            state_var.state = ov.Tensor(truncated)


# ── Truncation Protocol ───────────────────────────────────────────────────
# A tensor with dtype=int32, shape [1,1,2], values [-1, target_pos]
# signals KV cache truncation. Workers detect and relay this.

def is_truncation_command(tensor):
    return (tensor.dtype == np.int32 and tensor.size == 2
            and tensor.flat[0] == -1)


def make_truncation_command(target_pos):
    return np.array([[[-1, target_pos]]], dtype=np.int32)


def run_shard_stateless(compiled, meta, input_data, is_first_stage):
    """Non-cached: full sequence each time."""
    if is_first_stage:
        input_data = input_data.astype(np.int64)
    else:
        input_data = input_data.astype(np.float32)
    # Shards with internal rotary (e.g. Gemma 4 MoE) have 1 input;
    # shards with external rotary (e.g. Llama) have 3 inputs (data, cos, sin).
    if len(compiled.inputs) == 1:
        result = compiled({0: input_data})
    else:
        seq_len = input_data.shape[1]
        cos, sin = precompute_cos_sin(meta["head_dim"], seq_len)
        result = compiled({0: input_data, 1: cos, 2: sin})
    return result[compiled.output(0)]


def run_shard_stateful(request, compiled, meta, input_data, is_first_stage, position):
    """Stateful KV-cached inference.

    For prefill (position=None): input_data is full sequence, cos/sin for all positions.
    For decode (position=int): input_data is single token/hidden, cos/sin for that position.

    Input dispatch by compiled.inputs count:
      1 input  → internal rotary, no KV (Gemma 4 stateless)
      2 inputs → internal rotary + position_ids (Gemma 4 cached)
      3 inputs → data + cos + sin (Llama cached)
    """
    if is_first_stage:
        input_data = input_data.astype(np.int64)
    else:
        input_data = input_data.astype(np.float32)

    n_inputs = len(compiled.inputs)
    if n_inputs == 1:
        result = request.infer({0: input_data})
    elif n_inputs == 2:
        # Gemma 4 cached: (main_input, position_ids)
        if position is None:
            seq_len = input_data.shape[1]
            pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        else:
            pos_ids = np.array([[position]], dtype=np.int64)
        result = request.infer({0: input_data, 1: pos_ids})
    else:
        # Llama cached: (data, cos, sin)
        if position is None:
            seq_len = input_data.shape[1]
            cos, sin = precompute_cos_sin(meta["head_dim"], seq_len)
        else:
            cos, sin = precompute_cos_sin_at(meta["head_dim"], position)
        result = request.infer({0: input_data, 1: cos, 2: sin})
    return result[compiled.output(0)]


# ── Worker Mode ────────────────────────────────────────────────────────────

def run_worker(args):
    is_last = args.last_stage
    is_first = (args.stage_index == 0)
    stateful = args.stateful
    num_streams = getattr(args, 'num_streams', 1)

    print(f"[Stage {args.stage_index}] Loading shard from {args.shard_dir}...")
    compiled, request, meta = load_shard(args.shard_dir, args.device, stateful=stateful)

    # Create additional InferRequests for multi-stream
    requests = {0: request}
    if stateful and num_streams > 1:
        for sid in range(1, num_streams):
            requests[sid] = compiled.create_infer_request()
        print(f"[Stage {args.stage_index}] Created {num_streams} streams")

    # Warmup (stateful: run a dummy prefill+decode to trigger GPU compilation)
    n_inputs = len(compiled.inputs)
    if stateful:
        print(f"[Stage {args.stage_index}] Warming up (n_inputs={n_inputs})...")
        for sid, req in requests.items():
            reset_kv_cache(req, meta)
            if is_first:
                dummy = np.array([[1, 2]], dtype=np.int64)
            else:
                hidden_size = meta.get("hidden_size", 1536)
                # For Gemma4 cached, input includes PLI dims
                n_layers = meta.get("layer_end", 0) - meta.get("layer_start", 0)
                pli_dim = meta.get("pli_dim", 0)
                ds_pli = meta.get("downstream_pli_count", 0)
                extra = (n_layers + ds_pli) * pli_dim if meta.get("stateful") and pli_dim else 0
                dummy = np.zeros((1, 2, hidden_size + extra), dtype=np.float32)
            n_ext = meta.get("n_external_kv", 0)
            ext_dims = meta.get("external_kv_dims", [])
            if n_inputs == 1:
                req.infer({0: dummy})
            elif n_inputs == 2:
                req.infer({0: dummy, 1: np.array([[0, 1]], dtype=np.int64)})
            elif n_ext > 0:
                # Gemma4 with external KV: (main, pos, ext_k0, ext_v0, ...)
                feed = {0: dummy, 1: np.array([[0, 1]], dtype=np.int64)}
                for ei, hd in enumerate(ext_dims):
                    feed[2 + ei*2] = np.zeros((1, 1, 2, hd), dtype=np.float32)
                    feed[2 + ei*2 + 1] = np.zeros((1, 1, 2, hd), dtype=np.float32)
                req.infer(feed)
            else:
                cos, sin = precompute_cos_sin(meta["head_dim"], 2)
                req.infer({0: dummy, 1: cos, 2: sin})
            # Decode step
            if is_first:
                dummy_d = np.array([[3]], dtype=np.int64)
            else:
                dummy_d = np.zeros((1, 1, hidden_size + extra), dtype=np.float32)
            if n_inputs == 1:
                req.infer({0: dummy_d})
            elif n_inputs == 2:
                req.infer({0: dummy_d, 1: np.array([[2]], dtype=np.int64)})
            elif n_ext > 0:
                feed = {0: dummy_d, 1: np.array([[2]], dtype=np.int64)}
                for ei, hd in enumerate(ext_dims):
                    feed[2 + ei*2] = np.zeros((1, 1, 3, hd), dtype=np.float32)
                    feed[2 + ei*2 + 1] = np.zeros((1, 1, 3, hd), dtype=np.float32)
                req.infer(feed)
            else:
                cos_d, sin_d = precompute_cos_sin_at(meta["head_dim"], 2)
                req.infer({0: dummy_d, 1: cos_d, 2: sin_d})
            reset_kv_cache(req, meta)
        print(f"[Stage {args.stage_index}] Warmup done")

    # Listen for upstream connections
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    srv.bind(("0.0.0.0", args.listen_port))
    srv.listen(1)
    print(f"[Stage {args.stage_index}] Listening on port {args.listen_port}...")
    upstream_sock, addr = srv.accept()
    upstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Stage {args.stage_index}] Upstream connected from {addr}")

    # Connect to downstream (if not last stage)
    downstream_sock = None
    if not is_last:
        downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[Stage {args.stage_index}] Connecting to downstream {args.downstream_host}:{args.downstream_port}...")
        for _ in range(60):
            try:
                downstream_sock.connect((args.downstream_host, args.downstream_port))
                break
            except ConnectionRefusedError:
                time.sleep(0.5)
        else:
            raise TimeoutError("Could not connect to downstream")
        print(f"[Stage {args.stage_index}] Downstream connected")

    # Main loop — per-stream state
    tokens = 0
    total_compute_ms = 0
    total_network_ms = 0
    stream_positions = {sid: 0 for sid in requests}  # seq_position per stream
    stream_tokens = {sid: 0 for sid in requests}     # token count per stream
    print(f"[Stage {args.stage_index}] Ready. Processing tokens (streams={num_streams})...")

    # External KV inputs: cross-stage KV received from upstream
    n_ext_kv = meta.get("n_external_kv", 0)

    try:
        while True:
            # Receive main activation from upstream
            activation, recv_ms, stream_id = recv_tensor(upstream_sock)
            total_network_ms += recv_ms

            # Receive cross-stage KV tensors (if any), reshape 3D→4D
            ext_kv_data = []
            ext_dims = meta.get("external_kv_dims", [])
            for ki in range(n_ext_kv * 2):
                ekv, ekv_ms, _ = recv_tensor(upstream_sock)
                # Reshape [batch, seq, kv_heads*head_dim] → [batch, kv_heads, seq, head_dim]
                hd = ext_dims[ki // 2] if ki // 2 < len(ext_dims) else ekv.shape[-1]
                num_kv = meta.get("num_kv_heads", 1)
                if ekv.ndim == 3:
                    b, s, _ = ekv.shape
                    ekv = ekv.reshape(b, num_kv, s, hd)
                ext_kv_data.append(ekv)
                total_network_ms += ekv_ms

            cur_request = requests.get(stream_id, request)
            cur_pos = stream_positions.get(stream_id, 0)
            cur_tokens = stream_tokens.get(stream_id, 0)

            # Truncation command
            if stateful and is_truncation_command(activation):
                target_pos = int(activation.flat[1])
                truncate_kv_cache(cur_request, target_pos)
                stream_positions[stream_id] = target_pos
                if not is_last and downstream_sock:
                    send_tensor(downstream_sock, activation, stream_id=stream_id)
                    recv_tensor(downstream_sock)
                send_tensor(upstream_sock, np.array([0], dtype=np.int32), stream_id=stream_id)
                continue

            # Forward pass
            t0 = time.perf_counter()
            if n_ext_kv > 0 and len(ext_kv_data) > 0:
                # Gemma4 E2B with external KV: pass activation + pos + ext_kv
                activation_f = activation.astype(np.float32)
                if cur_tokens == 0 or activation.shape[1] > 1:
                    pos = np.arange(activation.shape[1], dtype=np.int64).reshape(1, -1)
                    stream_positions[stream_id] = (cur_pos or 0) + activation.shape[1]
                else:
                    pos = np.array([[cur_pos]], dtype=np.int64)
                    stream_positions[stream_id] = cur_pos + 1
                feed = {0: activation_f, 1: pos}
                for ki, ekv in enumerate(ext_kv_data):
                    feed[2 + ki] = ekv.astype(np.float32)
                result = compiled(feed)
                output = result[compiled.output(0)]
            elif stateful:
                seq_len = activation.shape[1]
                if cur_tokens == 0 or seq_len > 1:
                    stream_positions[stream_id] = (cur_pos or 0) + seq_len
                    output = run_shard_stateful(
                        cur_request, compiled, meta, activation,
                        is_first_stage=is_first, position=None)
                else:
                    output = run_shard_stateful(
                        cur_request, compiled, meta, activation,
                        is_first_stage=is_first, position=cur_pos)
                    stream_positions[stream_id] = cur_pos + 1
            else:
                output = run_shard_stateless(compiled, meta, activation, is_first_stage=is_first)
            compute_ms = (time.perf_counter() - t0) * 1000
            total_compute_ms += compute_ms
            tokens += 1
            stream_tokens[stream_id] = cur_tokens + 1

            if is_last:
                # For multi-token input, return ALL logits (speculation verification)
                if activation.shape[1] > 1:
                    all_tokens = np.argmax(output[0], axis=-1).astype(np.int32)
                    send_ms = send_tensor(upstream_sock, all_tokens.reshape(1, 1, -1), stream_id=stream_id)
                else:
                    token_id = int(np.argmax(output[0, -1, :]))
                    token_array = np.array([token_id], dtype=np.int32)
                    send_ms = send_tensor(upstream_sock, token_array, stream_id=stream_id)
                total_network_ms += send_ms
                print(f"  token {tokens} [s{stream_id}]: compute={compute_ms:.0f}ms")
            else:
                # Forward to downstream
                send_ms = send_tensor(downstream_sock, output, stream_id=stream_id)
                total_network_ms += send_ms

                # Relay result from downstream back upstream
                token_array, relay_recv_ms, _ = recv_tensor(downstream_sock)
                relay_send_ms = send_tensor(upstream_sock, token_array, stream_id=stream_id)
                total_network_ms += relay_recv_ms + relay_send_ms

    except (ConnectionError, BrokenPipeError):
        pass

    print(f"\n[Stage {args.stage_index}] Done. {tokens} tokens, "
          f"compute={total_compute_ms:.0f}ms, network={total_network_ms:.0f}ms")
    upstream_sock.close()
    if downstream_sock:
        downstream_sock.close()
    srv.close()


# ── Coordinator Mode ───────────────────────────────────────────────────────

def run_coordinator(args):
    from transformers import AutoTokenizer

    stateful = args.stateful

    print(f"[Coordinator] Loading shard from {args.shard_dir}...")
    compiled, request, meta = load_shard(args.shard_dir, args.device, stateful=stateful)

    print(f"[Coordinator] Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # Load early exit head if configured
    exit_head = None
    exit_threshold = getattr(args, 'early_exit_threshold', 0.0)
    if exit_threshold > 0 and hasattr(args, 'exit_head_path') and args.exit_head_path:
        print(f"[Coordinator] Loading exit head from {args.exit_head_path}...")
        exit_head = ExitHead(args.exit_head_path)
        print(f"[Coordinator] Early exit threshold: {exit_threshold}")

    # Warmup (trigger GPU compilation before timing)
    n_inputs = len(compiled.inputs)
    if stateful:
        print(f"[Coordinator] Warming up (n_inputs={n_inputs})...")
        reset_kv_cache(request, meta)
        dummy = np.array([[1, 2]], dtype=np.int64)
        if n_inputs == 1:
            request.infer({0: dummy})
        elif n_inputs == 2:
            request.infer({0: dummy, 1: np.array([[0, 1]], dtype=np.int64)})
        else:
            cos, sin = precompute_cos_sin(meta["head_dim"], 2)
            request.infer({0: dummy, 1: cos, 2: sin})
        dummy_d = np.array([[3]], dtype=np.int64)
        if n_inputs == 1:
            request.infer({0: dummy_d})
        elif n_inputs == 2:
            request.infer({0: dummy_d, 1: np.array([[2]], dtype=np.int64)})
        else:
            cos_d, sin_d = precompute_cos_sin_at(meta["head_dim"], 2)
            request.infer({0: dummy_d, 1: cos_d, 2: sin_d})
        reset_kv_cache(request, meta)
        print("[Coordinator] Warmup done")

    # Connect to stage 1
    downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Coordinator] Connecting to stage 1 at {args.downstream_host}:{args.downstream_port}...")
    for _ in range(60):
        try:
            downstream_sock.connect((args.downstream_host, args.downstream_port))
            break
        except ConnectionRefusedError:
            time.sleep(0.5)
    else:
        raise TimeoutError("Could not connect to stage 1")
    print("[Coordinator] Connected to stage 1")

    # Tokenize (with optional chat template)
    chat_tpl = getattr(args, 'chat_template', 'none')
    if chat_tpl == 'gemma4':
        formatted = f"<bos><|turn>user\n{args.prompt}<turn|>\n<|turn>model\n"
        input_ids = tokenizer.encode(formatted, return_tensors="np", add_special_tokens=False)
    elif chat_tpl == 'auto':
        msgs = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(formatted, return_tensors="np", add_special_tokens=False)
    else:
        input_ids = tokenizer.encode(args.prompt, return_tensors="np", add_special_tokens=False)
    prompt_len = input_ids.shape[1]
    print(f"[Coordinator] Prompt ({chat_tpl}): '{args.prompt}' ({prompt_len} tokens)")

    # Reset KV cache for new generation
    if stateful:
        reset_kv_cache(request, meta)

    generated_ids = []
    per_token_ms = []
    gen_start = time.perf_counter()
    total_compute_ms = 0
    total_network_ms = 0
    early_exits = 0
    overlap_hits = 0
    overlap_misses = 0
    overlap_enabled = getattr(args, 'overlap_compute', False) and stateful

    # Cross-stage KV: stage 0 may output shared KV for downstream stages
    n_cross_kv = meta.get("n_cross_stage_kv", 0)
    cross_kv_tensors = []  # managed by coordinator, sent to worker each step

    for step in range(args.max_tokens):
        token_start = time.perf_counter()

        # Stage 0: run shard
        t0 = time.perf_counter()
        if stateful:
            if step == 0:
                output = run_shard_stateful(
                    request, compiled, meta, input_ids,
                    is_first_stage=True, position=None)
            else:
                new_token_ids = np.array([[generated_ids[-1]]], dtype=np.int64)
                output = run_shard_stateful(
                    request, compiled, meta, new_token_ids,
                    is_first_stage=True, position=prompt_len + step - 1)
            # Extract cross-stage KV from multi-output model
            if n_cross_kv > 0:
                cross_kv_tensors = []
                for ci in range(n_cross_kv * 2):
                    ot = request.get_output_tensor(1 + ci)
                    cross_kv_tensors.append(np.array(ot.data).reshape(ot.shape))
        else:
            if step == 0:
                current_ids = input_ids.copy()
            else:
                current_ids = np.append(current_ids, [[generated_ids[-1]]], axis=1)
            output = run_shard_stateless(compiled, meta, current_ids, is_first_stage=True)
        compute_ms = (time.perf_counter() - t0) * 1000
        total_compute_ms += compute_ms

        # Early exit check: predict locally if confidence is high enough
        exited_early = False
        if exit_head and step > 0:  # Skip prefill for early exit
            ee_t0 = time.perf_counter()
            ee_token, ee_confidence = exit_head.predict(output)
            ee_ms = (time.perf_counter() - ee_t0) * 1000
            total_compute_ms += ee_ms
            if ee_confidence >= exit_threshold:
                next_token = ee_token
                exited_early = True

        if not exited_early:
            # Send hidden+PLI to stage 1
            send_ms = send_tensor(downstream_sock, output)
            total_network_ms += send_ms
            # Send cross-stage KV tensors (if any), reshaping 4D→3D for TCP
            for ckv in cross_kv_tensors:
                if ckv.ndim == 4:
                    # [batch, kv_heads, seq, head_dim] → [batch, seq, kv_heads*head_dim]
                    b, h, s, d = ckv.shape
                    ckv = ckv.reshape(b, s, h * d)
                send_ms = send_tensor(downstream_sock, ckv)
                total_network_ms += send_ms

            # Receive token back
            token_array, recv_ms, _ = recv_tensor(downstream_sock)
            total_network_ms += recv_ms
            next_token = int(token_array.flat[-1])

        token_ms = (time.perf_counter() - token_start) * 1000
        per_token_ms.append(token_ms)
        token_text = tokenizer.decode([next_token])
        if exited_early:
            early_exits += 1
        ee_tag = " [EXIT]" if exited_early else ""
        print(f"  token {step+1}: '{token_text}' (id={next_token}, {token_ms:.0f}ms, compute={compute_ms:.0f}ms){ee_tag}")

        generated_ids.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    total_ms = (time.perf_counter() - gen_start) * 1000
    num_tokens = len(generated_ids)

    # Emit the Tokens:/Tok/s: summary FIRST — demo_server.py's
    # _coordinator_reader triggers the WebSocket `done` event on the
    # Tok/s: line, and the frontend stops its elapsed timer only when
    # `done` arrives. Any work we do before Tok/s: (detokenizing the
    # full output, formatting extra stats) measurably delays `done`
    # and drags the visible tok/s number down.
    # All flushes explicit so no stdio buffering (SSH pipe, Windows
    # line-buffering on subprocess stdout, etc.) can stall the signal.
    print(f"Tokens: {num_tokens}", flush=True)
    print(f"Total: {total_ms:.0f} ms", flush=True)
    print(f"Tok/s: {num_tokens / (total_ms / 1000):.2f}", flush=True)

    # Everything below is diagnostic/human-readable and runs AFTER the
    # frontend has already registered completion.
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    ttft_ms = per_token_ms[0] if per_token_ms else 0
    decode_ms = per_token_ms[1:] if len(per_token_ms) > 1 else []
    decode_mean = sum(decode_ms) / len(decode_ms) if decode_ms else 0

    print(f"\n{'='*60}", flush=True)
    print(f"Output: {output_text}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Mode: {'stateful (KV-cached)' if stateful else 'stateless (full recompute)'}", flush=True)
    if overlap_enabled:
        total_spec = overlap_hits + overlap_misses
        hit_rate = overlap_hits / total_spec if total_spec > 0 else 0
        print(f"Overlap: hits={overlap_hits}, misses={overlap_misses}, rate={hit_rate:.1%}", flush=True)
    print(f"TTFT: {ttft_ms:.0f} ms", flush=True)
    print(f"Decode mean: {decode_mean:.0f} ms", flush=True)
    print(f"Compute: {total_compute_ms:.0f} ms", flush=True)
    print(f"Network: {total_network_ms:.0f} ms", flush=True)
    if exit_head:
        ee_rate = early_exits / max(num_tokens - 1, 1)  # exclude prefill
        print(f"Early exits: {early_exits}/{num_tokens-1} ({ee_rate:.1%})", flush=True)

    downstream_sock.close()


def run_coordinator_microbatch(args):
    """Coordinator with micro-batching: overlap stage 0 and stage 1 via N streams."""
    from transformers import AutoTokenizer

    stateful = args.stateful
    num_streams = getattr(args, 'num_streams', 2)

    print(f"[Coordinator] Loading shard from {args.shard_dir}...")
    compiled, request, meta = load_shard(args.shard_dir, args.device, stateful=stateful)

    # Create additional InferRequests for multi-stream
    requests = {0: request}
    for sid in range(1, num_streams):
        requests[sid] = compiled.create_infer_request()
    print(f"[Coordinator] Created {num_streams} streams")

    print(f"[Coordinator] Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    n_cross_kv = meta.get("n_cross_stage_kv", 0)
    n_inputs = len(compiled.inputs)

    # Warmup all requests
    if stateful:
        for sid, req in requests.items():
            reset_kv_cache(req, meta)
            dummy = np.array([[1, 2]], dtype=np.int64)
            if n_inputs == 2:
                req.infer({0: dummy, 1: np.array([[0, 1]], dtype=np.int64)})
            reset_kv_cache(req, meta)
        print("[Coordinator] Warmup done")

    # Connect to downstream
    downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Coordinator] Connecting to {args.downstream_host}:{args.downstream_port}...")
    for _ in range(60):
        try:
            downstream_sock.connect((args.downstream_host, args.downstream_port))
            break
        except ConnectionRefusedError:
            time.sleep(0.5)
    else:
        raise TimeoutError("Could not connect")
    print("[Coordinator] Connected")

    # Chat template
    chat_tpl = getattr(args, 'chat_template', 'none')

    # Use same prompt for all streams (throughput measurement)
    if chat_tpl == 'gemma4':
        formatted = f"<bos><|turn>user\n{args.prompt}<turn|>\n<|turn>model\n"
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    elif chat_tpl == 'auto':
        msgs = [{"role": "user", "content": args.prompt}]
        formatted = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    def send_with_kv(sock, req, output, stream_id):
        """Send hidden+PLI and cross-stage KV for a stream."""
        send_tensor(sock, output, stream_id=stream_id)
        if n_cross_kv > 0:
            for ci in range(n_cross_kv * 2):
                ot = req.get_output_tensor(1 + ci)
                ckv = np.array(ot.data).reshape(ot.shape)
                if ckv.ndim == 4:
                    b, h, s, d = ckv.shape
                    ckv = ckv.reshape(b, s, h * d)
                send_tensor(sock, ckv, stream_id=stream_id)

    max_tokens = args.max_tokens

    # Initialize streams
    streams = []
    for sid in range(num_streams):
        streams.append({
            "request": requests[sid],
            "generated": [],
            "step": 0,
            "done": False,
            "sid": sid,
        })

    # Reset and prefill all streams
    for s in streams:
        reset_kv_cache(s["request"], meta)
        ids_np = np.array([prompt_ids], dtype=np.int64)
        pos_np = np.arange(prompt_len, dtype=np.int64).reshape(1, -1)
        result = s["request"].infer({0: ids_np, 1: pos_np})
        output = result[compiled.output(0)]
        send_with_kv(downstream_sock, s["request"], output, s["sid"])
        token_array, _, _ = recv_tensor(downstream_sock)
        first_token = int(token_array.flat[-1])
        s["generated"].append(first_token)
        s["step"] = 1

    print(f"[Coordinator] Prefilled {num_streams} streams ({prompt_len} tokens each)")
    gen_start = time.perf_counter()
    total_tokens = 0

    # Pipelined decode: overlap stage 0 compute with stage 1 worker
    active = [s for s in streams if not s["done"] and s["step"] < max_tokens]
    while active:
        for idx in range(len(active)):
            s = active[idx]
            if s["done"] or s["step"] >= max_tokens:
                continue

            # Compute stage 0 for this stream
            tok = np.array([[s["generated"][-1]]], dtype=np.int64)
            pos = np.array([[prompt_len + s["step"] - 1]], dtype=np.int64)
            result = s["request"].infer({0: tok, 1: pos})
            output = result[compiled.output(0)]

            # Send to worker
            send_with_kv(downstream_sock, s["request"], output, s["sid"])

            # While worker processes this stream, pre-compute NEXT stream's stage 0
            next_idx = (idx + 1) % len(active)
            ns = active[next_idx]
            pre_output = None
            if next_idx != idx and not ns["done"] and ns["step"] < max_tokens:
                nt = np.array([[ns["generated"][-1]]], dtype=np.int64)
                np_ = np.array([[prompt_len + ns["step"] - 1]], dtype=np.int64)
                pre_result = ns["request"].infer({0: nt, 1: np_})
                pre_output = pre_result[compiled.output(0)]

            # Receive current stream's token
            token_array, _, _ = recv_tensor(downstream_sock)
            next_token = int(token_array.flat[-1])
            s["generated"].append(next_token)
            s["step"] += 1
            if next_token == tokenizer.eos_token_id or s["step"] >= max_tokens:
                s["done"] = True
                total_tokens += len(s["generated"])

            # Send pre-computed next stream
            if pre_output is not None and not ns["done"] and ns["step"] < max_tokens:
                send_with_kv(downstream_sock, ns["request"], pre_output, ns["sid"])
                token_array, _, _ = recv_tensor(downstream_sock)
                nt2 = int(token_array.flat[-1])
                ns["generated"].append(nt2)
                ns["step"] += 1
                if nt2 == tokenizer.eos_token_id or ns["step"] >= max_tokens:
                    ns["done"] = True
                    total_tokens += len(ns["generated"])

        active = [s for s in streams if not s["done"] and s["step"] < max_tokens]

    elapsed = time.perf_counter() - gen_start
    # Count remaining tokens from active-but-done streams
    for s in streams:
        if s["done"] and s not in []:
            pass  # already counted
        elif not s["done"]:
            total_tokens += len(s["generated"])

    print(f"\n{'='*60}")
    for s in streams:
        text = tokenizer.decode(s["generated"], skip_special_tokens=True)
        print(f"  [s{s['sid']}] {len(s['generated'])} tokens: {text[:80]}")
    print(f"{'='*60}")
    print(f"Streams: {num_streams}")
    print(f"Total tokens: {total_tokens}")
    print(f"Elapsed: {elapsed*1000:.0f} ms")
    print(f"Throughput: {total_tokens / elapsed:.2f} tok/s")
    per_request = total_tokens / num_streams
    print(f"Per-request: {per_request:.0f} tokens, {per_request / elapsed:.2f} tok/s")

    downstream_sock.close()


def run_coordinator_speculative(args):
    """Coordinator with speculative execution using a draft model."""
    import torch
    from transformers import AutoTokenizer
    from optimum.intel import OVModelForCausalLM

    print(f"[Coordinator] Loading pipeline shard from {args.shard_dir}...")
    compiled, request, meta = load_shard(args.shard_dir, args.device, stateful=True)

    print(f"[Coordinator] Loading draft model from {args.draft_model_dir}...")
    draft_model = OVModelForCausalLM.from_pretrained(
        args.draft_model_dir, device=args.device)

    print(f"[Coordinator] Loading tokenizer from {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    # Use pipeline tokenizer for draft model too (same family = same tokenizer)

    K = args.speculation_k
    print(f"[Coordinator] Speculation depth K={K}")

    # Warmup pipeline shard
    rotary_internal = len(compiled.inputs) == 1
    print(f"[Coordinator] Warming up pipeline shard (rotary_internal={rotary_internal})...")
    reset_kv_cache(request, meta)
    dummy = np.array([[1, 2]], dtype=np.int64)
    if rotary_internal:
        request.infer({0: dummy})
    else:
        cos, sin = precompute_cos_sin(meta["head_dim"], 2)
        request.infer({0: dummy, 1: cos, 2: sin})
    dummy_d = np.array([[3]], dtype=np.int64)
    if rotary_internal:
        request.infer({0: dummy_d})
    else:
        cos_d, sin_d = precompute_cos_sin_at(meta["head_dim"], 2)
        request.infer({0: dummy_d, 1: cos_d, 2: sin_d})
    reset_kv_cache(request, meta)

    # Warmup draft model
    print("[Coordinator] Warming up draft model...")
    draft_model.generate(torch.tensor([[1, 2]]), max_new_tokens=2, do_sample=False)
    print("[Coordinator] Warmup done")

    # Connect downstream
    downstream_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    downstream_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print(f"[Coordinator] Connecting to stage 1 at {args.downstream_host}:{args.downstream_port}...")
    for _ in range(60):
        try:
            downstream_sock.connect((args.downstream_host, args.downstream_port))
            break
        except ConnectionRefusedError:
            time.sleep(0.5)
    else:
        raise TimeoutError("Could not connect to stage 1")
    print("[Coordinator] Connected")

    # Tokenize with pipeline tokenizer
    input_ids = tokenizer.encode(args.prompt, return_tensors="np")
    prompt_len = input_ids.shape[1]
    print(f"[Coordinator] Prompt: '{args.prompt}' ({prompt_len} tokens)")

    # Reset pipeline KV cache
    reset_kv_cache(request, meta)

    # Prefill prompt through pipeline
    output = run_shard_stateful(request, compiled, meta, input_ids,
                                is_first_stage=True, position=None)
    send_tensor(downstream_sock, output)
    result, _, _ = recv_tensor(downstream_sock)
    first_token = int(result.flat[-1])

    generated_ids = [first_token]
    seq_pos = prompt_len + 1
    total_accepted = 0
    total_drafted = 0
    total_rounds = 0

    gen_start = time.perf_counter()
    total_compute_ms = 0
    total_network_ms = 0

    print(f"  prefill token: '{tokenizer.decode([first_token])}' (id={first_token})")

    while len(generated_ids) < args.max_tokens:
        round_start = time.perf_counter()
        total_rounds += 1

        # Phase 1: Draft K tokens with draft model
        # Use same tokenizer — pass token IDs directly (no decode/re-encode)
        t0 = time.perf_counter()
        all_ids = list(input_ids[0]) + generated_ids
        draft_input = torch.tensor([all_ids])
        with torch.no_grad():
            draft_output = draft_model.generate(
                draft_input, max_new_tokens=K, do_sample=False)
        drafted = draft_output[0, len(all_ids):].tolist()
        if len(drafted) == 0:
            drafted = [generated_ids[-1]]
        draft_ms = (time.perf_counter() - t0) * 1000
        total_compute_ms += draft_ms

        # Phase 2: Verify through the pipeline
        # Prepend the last confirmed token — it needs to be fed into the KV cache.
        # Verification batch: [last_confirmed, D1, D2, ..., DK-1]
        # Logits at position i predict what should follow the token at position i.
        t0 = time.perf_counter()
        K_actual = len(drafted)
        last_confirmed = generated_ids[-1]
        verify_ids = [last_confirmed] + drafted[:K_actual - 1]  # K tokens total
        verify_np = np.array([verify_ids], dtype=np.int64)  # [1, K]
        # cos/sin for positions [seq_pos-1, seq_pos, ..., seq_pos+K-2]
        cos_batch, sin_batch = precompute_cos_sin(
            meta["head_dim"], seq_pos + K_actual - 2)
        cos_slice = cos_batch[:, seq_pos-1:seq_pos-1+K_actual, :]
        sin_slice = sin_batch[:, seq_pos-1:seq_pos-1+K_actual, :]
        verify_result = request.infer({0: verify_np, 1: cos_slice, 2: sin_slice})
        verify_output = verify_result[compiled.output(0)]
        compute_ms = (time.perf_counter() - t0) * 1000
        total_compute_ms += compute_ms

        # Send K hidden states downstream
        send_ms = send_tensor(downstream_sock, verify_output)
        total_network_ms += send_ms

        # Receive K verified tokens (one per position in verify batch)
        verified_array, recv_ms, _ = recv_tensor(downstream_sock)
        total_network_ms += recv_ms
        verified_tokens = verified_array.flatten().astype(np.int32)

        # Phase 3: Find longest matching prefix
        # verified_tokens[i] = pipeline's prediction at position i of verify batch
        # verified_tokens[0] = what comes after last_confirmed (should match drafted[0])
        # verified_tokens[1] = what comes after drafted[0] (should match drafted[1])
        accepted = 0
        for i in range(min(K_actual, len(verified_tokens))):
            if drafted[i] == int(verified_tokens[i]):
                accepted += 1
            else:
                break

        total_drafted += K_actual
        total_accepted += accepted

        # Accept: accepted drafts + 1 corrected token from pipeline
        new_tokens = list(drafted[:accepted])
        if accepted < len(verified_tokens):
            new_tokens.append(int(verified_tokens[accepted]))

        for t in new_tokens:
            if len(generated_ids) >= args.max_tokens:
                break
            generated_ids.append(t)

        round_ms = (time.perf_counter() - round_start) * 1000
        token_strs = tokenizer.decode(new_tokens)
        print(f"  round {total_rounds}: drafted={K_actual}, accepted={accepted}, "
              f"got {len(new_tokens)} tokens: {token_strs!r} ({round_ms:.0f}ms)")

        # Phase 4: KV cache management
        # The verify batch was [last_confirmed, D1, ..., DK-1].
        # KV cache now has entries for: previous + last_confirmed + K-1 drafts.
        # We accepted `accepted` drafts. The correction token at verified[accepted]
        # is NOT in the KV cache (it wasn't in the verify input).
        # Keep: previous + last_confirmed (1) + accepted drafts (accepted) = +1+accepted
        # Then we need to process the correction token separately.
        n_to_keep = seq_pos + accepted  # seq_pos-1 was the old cache end, +1 for last_confirmed, +accepted
        if accepted < K_actual - 1:
            # Truncate excess KV entries from rejected drafts
            truncate_kv_cache(request, n_to_keep)
            trunc_cmd = make_truncation_command(n_to_keep)
            send_tensor(downstream_sock, trunc_cmd)
            recv_tensor(downstream_sock)  # ACK

        # Process the correction token to add its KV entry
        if new_tokens and new_tokens[-1] != drafted[accepted - 1] if accepted > 0 else True:
            corr_id = new_tokens[-1]
            corr_np = np.array([[corr_id]], dtype=np.int64)
            corr_cos, corr_sin = precompute_cos_sin_at(meta["head_dim"], n_to_keep)
            corr_result = request.infer({0: corr_np, 1: corr_cos, 2: corr_sin})
            corr_hidden = corr_result[compiled.output(0)]
            send_tensor(downstream_sock, corr_hidden)
            recv_tensor(downstream_sock)  # discard

        seq_pos += len(new_tokens)

        if generated_ids[-1] == tokenizer.eos_token_id:
            break

    total_ms = (time.perf_counter() - gen_start) * 1000
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    num_tokens = len(generated_ids)
    acceptance_rate = total_accepted / total_drafted if total_drafted > 0 else 0

    print(f"\n{'='*60}")
    print(f"Output: {output_text}")
    print(f"{'='*60}")
    print(f"Mode: speculative (K={K}, draft=TinyLlama)")
    print(f"Tokens: {num_tokens}")
    print(f"Total: {total_ms:.0f} ms")
    print(f"Tok/s: {num_tokens / (total_ms / 1000):.2f}")
    print(f"Rounds: {total_rounds}")
    print(f"Acceptance: {acceptance_rate:.1%} ({total_accepted}/{total_drafted})")
    print(f"Tokens/round: {num_tokens / total_rounds:.1f}")
    print(f"Compute: {total_compute_ms:.0f} ms")
    print(f"Network: {total_network_ms:.0f} ms")

    downstream_sock.close()


# ── CLI ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed pipeline node")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Worker
    wp = sub.add_parser("worker")
    wp.add_argument("--stage-index", type=int, required=True)
    wp.add_argument("--shard-dir", required=True)
    wp.add_argument("--listen-port", type=int, default=9100)
    wp.add_argument("--downstream-host", default=None)
    wp.add_argument("--downstream-port", type=int, default=None)
    wp.add_argument("--device", default="GPU")
    wp.add_argument("--last-stage", action="store_true")
    wp.add_argument("--stateful", action="store_true", help="Use stateful KV-cached inference")
    wp.add_argument("--activation-compress", default="none", choices=["none", "fp16", "int8"],
                    help="Compress hidden_states between stages")
    wp.add_argument("--add-latency-ms", type=float, default=0,
                    help="Simulated one-way network latency in ms (RTT = 2x)")
    wp.add_argument("--num-streams", type=int, default=1,
                    help="Number of concurrent inference streams (micro-batching)")
    wp.add_argument("--sparse-k", type=int, default=0,
                    help="Top-K sparse activation (0=disabled, 512/1024/2048=keep top-K dims)")

    # Coordinator
    cp = sub.add_parser("coordinator")
    cp.add_argument("--shard-dir", required=True)
    cp.add_argument("--tokenizer-dir", required=True)
    cp.add_argument("--downstream-host", required=True)
    cp.add_argument("--downstream-port", type=int, default=9100)
    cp.add_argument("--device", default="GPU")
    cp.add_argument("--prompt", default="What is the capital of France?")
    cp.add_argument("--max-tokens", type=int, default=20)
    cp.add_argument("--chat-template", default="none", choices=["none", "gemma4", "auto"],
                    help="Apply chat template before tokenizing")
    cp.add_argument("--stateful", action="store_true", help="Use stateful KV-cached inference")
    cp.add_argument("--activation-compress", default="none", choices=["none", "fp16", "int8"],
                    help="Compress hidden_states between stages")
    cp.add_argument("--add-latency-ms", type=float, default=0,
                    help="Simulated one-way network latency in ms (RTT = 2x)")
    cp.add_argument("--draft-model-dir", default=None,
                    help="Path to draft model OV IR for speculative execution")
    cp.add_argument("--speculation-k", type=int, default=4,
                    help="Number of tokens to draft per speculation round")
    cp.add_argument("--exit-head-path", default=None,
                    help="Path to exit_head.npz for early exit")
    cp.add_argument("--early-exit-threshold", type=float, default=0.0,
                    help="Min softmax confidence for early exit (0=disabled)")
    cp.add_argument("--overlap-compute", action="store_true",
                    help="Overlap stage 0 compute with downstream wait (repeat-last speculation)")

    cp.add_argument("--sparse-k", type=int, default=0,
                    help="Top-K sparse activation (0=disabled, 512/1024/2048=keep top-K dims)")
    cp.add_argument("--num-streams", type=int, default=1,
                    help="Number of concurrent inference streams (micro-batching)")

    args = parser.parse_args()
    _set_compress_method(args.activation_compress)
    _set_latency(args.add_latency_ms)
    _set_sparse_k(getattr(args, 'sparse_k', 0))

    if args.mode == "worker":
        run_worker(args)
    elif args.draft_model_dir:
        run_coordinator_speculative(args)
    elif getattr(args, 'num_streams', 1) > 1:
        run_coordinator_microbatch(args)
    else:
        run_coordinator(args)
