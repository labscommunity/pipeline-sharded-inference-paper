"""Rigorous methodology benchmark.

Compares four configurations with identical input token IDs and identical
measurement loop structure (for A', B, C). Workload A uses openvino_genai's
own C++ decode loop for comparison with our hand-rolled Python loop.

Configurations:
  A  - Monolithic INT4 via openvino_genai.LLMPipeline.generate()
  A' - Monolithic INT4 via ov.Core().compile_model() + hand-rolled Python decode loop
       (same path B and C use — isolates openvino_genai C++ loop vs our Python loop)
  B  - 1-stage INT4 shard (our export pipeline, FP16) + hand-rolled loop
  C  - 2-stage INT4 shards (our export pipeline, FP32) + hand-rolled loop

For each config:
  - Warmup 2 full generations (discarded)
  - Measure N full generations of MAX_TOKENS, all seeing the same prompt_ids
  - Record elapsed per run and generated token_ids

Output: JSON to stdout with per-run results, mean, stddev, and sanity check
that all runs produced identical token sequences (greedy decoding).
"""
import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

MONOLITHIC_PATH = r"C:\cascadia\models\llama-3.1-8b-int4"
STAGE1_SHARD_DIR = r"C:\cascadia\shards_1stage\stage_0"
STAGE1_TOK = r"C:\cascadia\shards_1stage\tokenizer"
STAGE2_DIRS = (r"C:\cascadia\shards_2stage\stage_0", r"C:\cascadia\shards_2stage\stage_1")
STAGE2_TOK = r"C:\cascadia\shards_2stage\tokenizer"
STAGE1_V2_SHARD_DIR = r"C:\cascadia\shards_1stage_v2\stage_0"
STAGE2_V2_DIRS = (r"C:\cascadia\shards_2stage_v2\stage_0", r"C:\cascadia\shards_2stage_v2\stage_1")
STAGE1_V3_SHARD_DIR = r"C:\cascadia\shards_1stage_v3\stage_0"
STAGE2_V3_DIRS = (r"C:\cascadia\shards_2stage_v3\stage_0", r"C:\cascadia\shards_2stage_v3\stage_1")
STAGE1_V3_FP32_SHARD_DIR = r"C:\cascadia\shards_1stage_v3_fp32\stage_0"
STAGE2_V3_FP32_DIRS = (r"C:\cascadia\shards_2stage_v3_fp32\stage_0", r"C:\cascadia\shards_2stage_v3_fp32\stage_1")
STAGE1_V4_SHARD_DIR = r"C:\cascadia\shards_1stage_v4\stage_0"
STAGE1_V5_BEAM_SHARD_DIR = r"C:\cascadia\shards_1stage_v5_beam\stage_0"
STAGE1_V5_PAGED_SHARD_DIR = r"C:\cascadia\shards_1stage_v5_paged\stage_0"

PROMPT = "What is the capital of France?"
MAX_TOKENS = 50
HEAD_DIM = 128
ROPE_THETA = 500000.0
DEVICE = "GPU"

# ─────────────────────── rotary + KV helpers ──────────────────────────────

def precompute_cos_sin(seq_len):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    positions = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(positions, inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(emb)[np.newaxis, :, :].astype(np.float32)
    sin = np.sin(emb)[np.newaxis, :, :].astype(np.float32)
    return cos, sin


def precompute_cos_sin_at(position):
    inv_freq = 1.0 / (ROPE_THETA ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
    freqs = position * inv_freq
    emb = np.concatenate([freqs, freqs])
    cos = np.cos(emb).reshape(1, 1, -1).astype(np.float32)
    sin = np.sin(emb).reshape(1, 1, -1).astype(np.float32)
    return cos, sin


def reset_kv(request):
    import openvino as ov
    request.reset_state()
    for state_var in request.query_state():
        shape = list(state_var.state.shape)
        shape[0] = 1
        shape[2] = 0
        state_var.state = ov.Tensor(np.zeros(shape, dtype=np.float32))


# ─────────────────────── tokenization ──────────────────────────────────────

def tokenize_prompt_shared(apply_chat_template: bool):
    """Use HuggingFace AutoTokenizer with the monolithic model's tokenizer.
    Returns int64 np array, shape [1, seq_len].

    NOTE: For Llama-3.1-Instruct, the openvino_genai default when given a
    string applies the chat template. To match, we apply it here.
    """
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MONOLITHIC_PATH)
    if apply_chat_template:
        msgs = [{"role": "user", "content": PROMPT}]
        formatted = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.encode(formatted, return_tensors="np", add_special_tokens=False)
    else:
        ids = tok.encode(PROMPT, return_tensors="np", add_special_tokens=True)
    return ids.astype(np.int64), tok


# ─────────────────────── Workload A (openvino_genai) ──────────────────────

def bench_a(input_ids, n_runs, max_tokens):
    import openvino as ov
    import openvino_genai as ov_genai

    print(f"[A] Loading openvino_genai.LLMPipeline on {DEVICE}...", flush=True)
    pipe = ov_genai.LLMPipeline(MONOLITHIC_PATH, DEVICE)

    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_tokens
    gen_cfg.min_new_tokens = max_tokens
    gen_cfg.ignore_eos = True
    gen_cfg.apply_chat_template = False  # we already applied it

    # Wrap as ov.Tensor
    tensor = ov.Tensor(input_ids)

    # Warmup
    print("[A] warmup x2...", flush=True)
    for _ in range(2):
        pipe.generate(tensor, gen_cfg)

    # Measure
    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = pipe.generate(tensor, gen_cfg)
        dt = time.perf_counter() - t0
        # result is EncodedResults when input is Tensor
        toks = list(result.tokens[0])
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[A] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del pipe
    gc.collect()
    return runs, token_ids


# ─────────────────────── hand-rolled decode loop ──────────────────────────

def handrolled_generate(compiled, request, input_ids, max_tokens):
    """Greedy decode for single-stage model (v1: external cos/sin). Returns generated token ids list."""
    generated = []
    seq_len = input_ids.shape[1]
    reset_kv(request)

    # Prefill
    cos, sin = precompute_cos_sin(seq_len)
    request.infer({0: input_ids, 1: cos, 2: sin})
    logits = request.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    generated.append(next_token)

    # Decode
    for i in range(1, max_tokens):
        pos = seq_len + i - 1
        ids = np.array([[next_token]], dtype=np.int64)
        c, s = precompute_cos_sin_at(pos)
        request.infer({0: ids, 1: c, 2: s})
        logits = request.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
    return generated


def handrolled_generate_v2(compiled, request, input_ids, max_tokens):
    """Greedy decode for v2 shard: (input_ids, position_ids) inputs."""
    generated = []
    seq_len = input_ids.shape[1]
    reset_kv(request)

    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    request.infer({0: input_ids, 1: pos_ids})
    logits = request.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    generated.append(next_token)

    for i in range(1, max_tokens):
        pos_id = seq_len + i - 1
        ids = np.array([[next_token]], dtype=np.int64)
        pos = np.array([[pos_id]], dtype=np.int64)
        request.infer({0: ids, 1: pos})
        logits = request.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
    return generated


def handrolled_generate_2stage_v2(compiled0, req0, compiled1, req1, input_ids, max_tokens):
    """v2 2-stage (position_ids input)."""
    generated = []
    seq_len = input_ids.shape[1]
    reset_kv(req0)
    reset_kv(req1)

    pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
    req0.infer({0: input_ids, 1: pos_ids})
    hidden = req0.get_output_tensor(0).data
    req1.infer({0: hidden.astype(np.float32), 1: pos_ids})
    logits = req1.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    generated.append(next_token)

    for i in range(1, max_tokens):
        pos_id = seq_len + i - 1
        ids = np.array([[next_token]], dtype=np.int64)
        pos = np.array([[pos_id]], dtype=np.int64)
        req0.infer({0: ids, 1: pos})
        hidden = req0.get_output_tensor(0).data
        req1.infer({0: hidden.astype(np.float32), 1: pos})
        logits = req1.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
    return generated


def handrolled_generate_2stage(compiled0, req0, compiled1, req1, input_ids, max_tokens):
    """Two-stage chained decode, hand-rolled, same as B but 2 compiled graphs."""
    generated = []
    seq_len = input_ids.shape[1]
    reset_kv(req0)
    reset_kv(req1)

    # Prefill
    cos, sin = precompute_cos_sin(seq_len)
    req0.infer({0: input_ids, 1: cos, 2: sin})
    hidden = req0.get_output_tensor(0).data
    req1.infer({0: hidden.astype(np.float32), 1: cos, 2: sin})
    logits = req1.get_output_tensor(0).data
    next_token = int(np.argmax(logits[0, -1, :]))
    generated.append(next_token)

    # Decode
    for i in range(1, max_tokens):
        pos = seq_len + i - 1
        ids = np.array([[next_token]], dtype=np.int64)
        c, s = precompute_cos_sin_at(pos)
        req0.infer({0: ids, 1: c, 2: s})
        hidden = req0.get_output_tensor(0).data
        req1.infer({0: hidden.astype(np.float32), 1: c, 2: s})
        logits = req1.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
    return generated


# ─────────────────────── Workload A' ───────────────────────────────────────

def bench_a_prime(input_ids, n_runs, max_tokens):
    """Monolithic model but through our Python loop. Isolates C++ vs Python."""
    import openvino as ov
    print(f"[A'] Loading monolithic openvino_model.xml on {DEVICE} via ov.Core()...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(MONOLITHIC_PATH, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    # Verify shape of inputs
    n_inputs = len(compiled.inputs)
    print(f"[A'] compiled with {n_inputs} inputs", flush=True)

    # openvino_genai's monolithic IR has different input layout.
    # Let's introspect what inputs are expected.
    for i, inp in enumerate(compiled.inputs):
        print(f"  input[{i}]: names={inp.get_names()}, partial_shape={inp.get_partial_shape()}", flush=True)

    # This model may expect (input_ids, attention_mask, position_ids, beam_idx) — the
    # openvino_genai-style stateful LLM. Try a safer invocation path.
    has_beam_idx = any("beam_idx" in inp.get_names() for inp in compiled.inputs)
    print(f"[A'] has_beam_idx: {has_beam_idx}", flush=True)

    # For the genai-style stateful IR, the standard invocation is:
    #   inputs: input_ids, attention_mask, position_ids, beam_idx
    # We run a manual loop:

    def gen_one():
        request.reset_state()  # clears KV cache for genai-style stateful IR
        generated = []
        seq_len = input_ids.shape[1]
        # Prefill: feed full input_ids
        att_mask = np.ones((1, seq_len), dtype=np.int64)
        pos_ids = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        beam_idx = np.zeros(1, dtype=np.int32) if has_beam_idx else None
        feed = {"input_ids": input_ids, "attention_mask": att_mask, "position_ids": pos_ids}
        if has_beam_idx:
            feed["beam_idx"] = beam_idx
        request.infer(feed)
        logits = request.get_output_tensor(0).data
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)

        # Decode
        for i in range(1, max_tokens):
            ids = np.array([[next_token]], dtype=np.int64)
            att_mask = np.ones((1, seq_len + i), dtype=np.int64)
            pos_ids = np.array([[seq_len + i - 1]], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att_mask, "position_ids": pos_ids}
            if has_beam_idx:
                feed["beam_idx"] = np.zeros(1, dtype=np.int32)
            request.infer(feed)
            logits = request.get_output_tensor(0).data
            next_token = int(np.argmax(logits[0, -1, :]))
            generated.append(next_token)
        return generated

    # Warmup
    print("[A'] warmup x2...", flush=True)
    for _ in range(2):
        gen_one()

    # Measure
    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = gen_one()
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[A'] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


# ─────────────────────── Workload B ────────────────────────────────────────

def bench_b(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[B] Loading 1-stage shard {STAGE1_SHARD_DIR}...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(STAGE1_SHARD_DIR, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    print("[B] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate(compiled, request, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate(compiled, request, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[B] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


# ─────────────────────── Workload C ────────────────────────────────────────

def bench_c(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[C] Loading 2-stage shards...", flush=True)
    core = ov.Core()
    m0 = core.read_model(os.path.join(STAGE2_DIRS[0], "openvino_model.xml"))
    c0 = core.compile_model(m0, DEVICE)
    r0 = c0.create_infer_request()
    m1 = core.read_model(os.path.join(STAGE2_DIRS[1], "openvino_model.xml"))
    c1 = core.compile_model(m1, DEVICE)
    r1 = c1.create_infer_request()

    print("[C] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[C] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del c0, r0, m0, c1, r1, m1
    gc.collect()
    return runs, token_ids


# ─────────────────────── Workload B_v2 (SDPA + internal rotary) ─────────────

def bench_b_v2(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[B_v2] Loading 1-stage_v2 shard {STAGE1_V2_SHARD_DIR}...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(STAGE1_V2_SHARD_DIR, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    print("[B_v2] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate_v2(compiled, request, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate_v2(compiled, request, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[B_v2] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


def bench_c_v2(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[C_v2] Loading 2-stage_v2 shards...", flush=True)
    core = ov.Core()
    m0 = core.read_model(os.path.join(STAGE2_V2_DIRS[0], "openvino_model.xml"))
    c0 = core.compile_model(m0, DEVICE)
    r0 = c0.create_infer_request()
    m1 = core.read_model(os.path.join(STAGE2_V2_DIRS[1], "openvino_model.xml"))
    c1 = core.compile_model(m1, DEVICE)
    r1 = c1.create_infer_request()

    print("[C_v2] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate_2stage_v2(c0, r0, c1, r1, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate_2stage_v2(c0, r0, c1, r1, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[C_v2] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del c0, r0, m0, c1, r1, m1
    gc.collect()
    return runs, token_ids


# ─────────────────────── Workload B_v3 / C_v3 (shared rotary + SDPA) ───────

def bench_b_v3(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[B_v3] Loading 1-stage_v3 shard {STAGE1_V3_SHARD_DIR}...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(STAGE1_V3_SHARD_DIR, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    print("[B_v3] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate(compiled, request, input_ids, max_tokens)  # v1-style (cos,sin inputs)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate(compiled, request, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[B_v3] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


def bench_c_v3(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[C_v3] Loading 2-stage_v3 shards...", flush=True)
    core = ov.Core()
    m0 = core.read_model(os.path.join(STAGE2_V3_DIRS[0], "openvino_model.xml"))
    c0 = core.compile_model(m0, DEVICE)
    r0 = c0.create_infer_request()
    m1 = core.read_model(os.path.join(STAGE2_V3_DIRS[1], "openvino_model.xml"))
    c1 = core.compile_model(m1, DEVICE)
    r1 = c1.create_infer_request()

    print("[C_v3] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)  # v1-style (cos,sin shared)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[C_v3] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del c0, r0, m0, c1, r1, m1
    gc.collect()
    return runs, token_ids


def _bench_v5_shard(shard_dir, label, input_ids, n_runs, max_tokens):
    """Bench a v5-style shard with canonical inputs (input_ids, attention_mask,
    position_ids[, beam_idx])."""
    import openvino as ov
    print(f"[{label}] Loading shard {shard_dir}...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(shard_dir, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    has_beam_idx = any("beam_idx" in inp.get_names() for inp in compiled.inputs)
    req = compiled.create_infer_request()
    seq_len = input_ids.shape[1]

    def gen_one():
        req.reset_state()
        gens = []
        att = np.ones((1, seq_len), dtype=np.int64)
        pos = np.arange(seq_len, dtype=np.int64).reshape(1, -1)
        feed = {"input_ids": input_ids, "attention_mask": att, "position_ids": pos}
        if has_beam_idx:
            feed["beam_idx"] = np.zeros(1, dtype=np.int32)
        req.infer(feed)
        logits = req.get_output_tensor(0).data
        nt = int(np.argmax(logits[0, -1, :]))
        gens.append(nt)
        for i in range(1, max_tokens):
            ids = np.array([[nt]], dtype=np.int64)
            att = np.ones((1, seq_len + i), dtype=np.int64)
            pos = np.array([[seq_len + i - 1]], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": att, "position_ids": pos}
            if has_beam_idx:
                feed["beam_idx"] = np.zeros(1, dtype=np.int32)
            req.infer(feed)
            logits = req.get_output_tensor(0).data
            nt = int(np.argmax(logits[0, -1, :]))
            gens.append(nt)
        return gens

    print(f"[{label}] warmup x2...", flush=True)
    for _ in range(2):
        gen_one()
    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = gen_one()
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[{label}] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, req, model
    gc.collect()
    return runs, token_ids


def bench_b_v5_beam(input_ids, n_runs, max_tokens):
    return _bench_v5_shard(STAGE1_V5_BEAM_SHARD_DIR, "B_v5_beam", input_ids, n_runs, max_tokens)


def bench_b_v5_paged(input_ids, n_runs, max_tokens):
    return _bench_v5_shard(STAGE1_V5_PAGED_SHARD_DIR, "B_v5_paged", input_ids, n_runs, max_tokens)


def bench_b_v4(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[B_v4] Loading 1-stage_v4 (AWQ) shard...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(STAGE1_V4_SHARD_DIR, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    print("[B_v4] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate(compiled, request, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate(compiled, request, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[B_v4] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


def bench_b_v3_fp32(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[B_v3_fp32] Loading 1-stage_v3_fp32 shard...", flush=True)
    core = ov.Core()
    model = core.read_model(os.path.join(STAGE1_V3_FP32_SHARD_DIR, "openvino_model.xml"))
    compiled = core.compile_model(model, DEVICE)
    request = compiled.create_infer_request()

    print("[B_v3_fp32] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate(compiled, request, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate(compiled, request, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[B_v3_fp32] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del compiled, request, model
    gc.collect()
    return runs, token_ids


def bench_c_v3_fp32(input_ids, n_runs, max_tokens):
    import openvino as ov
    print(f"[C_v3_fp32] Loading 2-stage_v3_fp32 shards...", flush=True)
    core = ov.Core()
    m0 = core.read_model(os.path.join(STAGE2_V3_FP32_DIRS[0], "openvino_model.xml"))
    c0 = core.compile_model(m0, DEVICE)
    r0 = c0.create_infer_request()
    m1 = core.read_model(os.path.join(STAGE2_V3_FP32_DIRS[1], "openvino_model.xml"))
    c1 = core.compile_model(m1, DEVICE)
    r1 = c1.create_infer_request()

    print("[C_v3_fp32] warmup x2...", flush=True)
    for _ in range(2):
        handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)

    runs = []
    token_ids = None
    for i in range(n_runs):
        t0 = time.perf_counter()
        toks = handrolled_generate_2stage(c0, r0, c1, r1, input_ids, max_tokens)
        dt = time.perf_counter() - t0
        if token_ids is None:
            token_ids = toks
        tok_s = len(toks) / dt
        runs.append({"elapsed_s": dt, "tokens": len(toks), "tok_s": tok_s})
        print(f"[C_v3_fp32] run {i+1}: {len(toks)} tokens in {dt:.3f}s -> {tok_s:.2f} tok/s", flush=True)

    del c0, r0, m0, c1, r1, m1
    gc.collect()
    return runs, token_ids


# ─────────────────────── orchestration ─────────────────────────────────────

def summarize(name, runs, token_ids, tokenizer, all_results):
    tok_ss = [r["tok_s"] for r in runs]
    mean = statistics.mean(tok_ss)
    sd = statistics.stdev(tok_ss) if len(tok_ss) > 1 else 0.0
    decoded = tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else ""
    all_results[name] = {
        "runs": runs,
        "mean_tok_s": mean,
        "stddev_tok_s": sd,
        "token_ids": token_ids,
        "decoded": decoded,
    }
    print(f"\n[{name}] mean = {mean:.3f} tok/s  stddev = {sd:.3f}  (n={len(runs)})", flush=True)
    print(f"[{name}] first 15 tokens: {token_ids[:15]}", flush=True)
    print(f"[{name}] decoded[:120]: {decoded[:120]!r}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["a", "ap", "b", "c", "b_v2", "c_v2", "b_v3", "c_v3", "b_v3_fp32", "c_v3_fp32", "b_v4", "b_v5_beam", "b_v5_paged", "all", "v5_only", "final", "fp32_only", "v2_only", "v3_only"], required=True)
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Use raw prompt without chat template (for legacy smoke-test match)")
    parser.add_argument("--output", default=None, help="Write JSON to this file")
    args = parser.parse_args()

    apply_chat_template = not args.no_chat_template
    input_ids, tokenizer = tokenize_prompt_shared(apply_chat_template=apply_chat_template)
    print(f"Tokenized prompt (chat_template={apply_chat_template}): {input_ids.shape}", flush=True)
    print(f"Token IDs: {input_ids[0].tolist()}", flush=True)

    all_results = {
        "prompt": PROMPT,
        "apply_chat_template": apply_chat_template,
        "max_tokens": args.max_tokens,
        "n_runs": args.n,
        "input_token_ids": input_ids[0].tolist(),
        "device": DEVICE,
    }

    which = args.workload
    if which in ("a", "all"):
        try:
            runs, toks = bench_a(input_ids, args.n, args.max_tokens)
            summarize("A", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["A"] = {"error": str(e)}
    if which in ("ap", "all"):
        try:
            runs, toks = bench_a_prime(input_ids, args.n, args.max_tokens)
            summarize("A_prime", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["A_prime"] = {"error": str(e)}
    if which in ("b", "all"):
        try:
            runs, toks = bench_b(input_ids, args.n, args.max_tokens)
            summarize("B", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B"] = {"error": str(e)}
    if which in ("c", "all"):
        try:
            runs, toks = bench_c(input_ids, args.n, args.max_tokens)
            summarize("C", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["C"] = {"error": str(e)}
    if which in ("b_v2", "all", "v2_only"):
        try:
            runs, toks = bench_b_v2(input_ids, args.n, args.max_tokens)
            summarize("B_v2", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v2"] = {"error": str(e)}
    if which in ("c_v2", "all", "v2_only"):
        try:
            runs, toks = bench_c_v2(input_ids, args.n, args.max_tokens)
            summarize("C_v2", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["C_v2"] = {"error": str(e)}
    if which in ("b_v3", "all", "v3_only"):
        try:
            runs, toks = bench_b_v3(input_ids, args.n, args.max_tokens)
            summarize("B_v3", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v3"] = {"error": str(e)}
    if which in ("c_v3", "all", "v3_only"):
        try:
            runs, toks = bench_c_v3(input_ids, args.n, args.max_tokens)
            summarize("C_v3", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["C_v3"] = {"error": str(e)}
    if which in ("b_v3_fp32", "all", "fp32_only"):
        try:
            runs, toks = bench_b_v3_fp32(input_ids, args.n, args.max_tokens)
            summarize("B_v3_fp32", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v3_fp32"] = {"error": str(e)}
    if which in ("c_v3_fp32", "all", "fp32_only"):
        try:
            runs, toks = bench_c_v3_fp32(input_ids, args.n, args.max_tokens)
            summarize("C_v3_fp32", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["C_v3_fp32"] = {"error": str(e)}
    if which in ("b_v4", "all"):
        try:
            runs, toks = bench_b_v4(input_ids, args.n, args.max_tokens)
            summarize("B_v4", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v4"] = {"error": str(e)}
    if which in ("b_v5_beam", "all", "v5_only", "final"):
        try:
            runs, toks = bench_b_v5_beam(input_ids, args.n, args.max_tokens)
            summarize("B_v5_beam", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v5_beam"] = {"error": str(e)}
    if which in ("b_v5_paged", "all", "v5_only", "final"):
        try:
            runs, toks = bench_b_v5_paged(input_ids, args.n, args.max_tokens)
            summarize("B_v5_paged", runs, toks, tokenizer, all_results)
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results["B_v5_paged"] = {"error": str(e)}

    # Correctness diff
    print("\n=== CORRECTNESS DIFF ===", flush=True)
    tok_sets = {k: v.get("token_ids") for k, v in all_results.items()
                if isinstance(v, dict) and "token_ids" in v}
    names = list(tok_sets.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            t1 = tok_sets[n1]
            t2 = tok_sets[n2]
            if t1 is None or t2 is None:
                continue
            if t1 == t2:
                print(f"  {n1} == {n2}  (all {len(t1)} tokens match)", flush=True)
            else:
                diverge = next((j for j in range(min(len(t1), len(t2))) if t1[j] != t2[j]), -1)
                print(f"  {n1} != {n2}  (diverge at token {diverge}: {t1[diverge] if diverge>=0 else '?'} vs {t2[diverge] if diverge>=0 else '?'})", flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
