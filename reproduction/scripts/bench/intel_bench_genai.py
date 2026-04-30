"""Vanilla openvino_genai baseline for the Intel tok/s benchmark.

Loads an OV INT4 Llama-3.1-8B model through `openvino_genai.LLMPipeline`
— the stock, off-the-shelf path most developers would use to run a
quantized model on Arc iGPU. NO custom stateful re-export, NO manual
attention tracing, NO sharding. This is the apples-to-apples
"out of the box" baseline that DISCOVERIES #2 measured +22% / +45%
speedup against.

Emits the same stdout contract as distributed_node.py / intel_bench_local.py
so the orchestrator parses it uniformly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def log(msg):
    print(f"[genai] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Path to an OV INT4 model directory (openvino_model.xml "
                         "+ tokenizer.xml + tokenizer_config.json etc).")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()

    import openvino_genai as ov_genai

    log(f"loading LLMPipeline from {args.model_dir} on {args.device}...")
    t0 = time.time()
    pipe = ov_genai.LLMPipeline(args.model_dir, args.device)
    log(f"  pipeline ready in {time.time()-t0:.1f}s")

    # Warmup: short throwaway generation to trigger graph compile.
    log("warmup...")
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 4
    cfg.do_sample = False
    try:
        pipe.generate("ping", cfg)
    except Exception as e:
        log(f"[warn] warmup failed (continuing): {e}")

    # Real generation — greedy decode, max_new_tokens = args.max_tokens.
    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = args.max_tokens
    gen_cfg.do_sample = False

    # Per-token streaming so we can emit the per_token_ms samples the
    # orchestrator parses for steady-state decode tok/s.
    per_token_times_ms = []
    last_t = time.perf_counter()
    token_index = 0
    tokenizer = pipe.get_tokenizer()

    def streamer(subword: str) -> bool:
        nonlocal last_t, token_index
        now = time.perf_counter()
        dt_ms = int((now - last_t) * 1000)
        last_t = now
        # NB: openvino_genai streams text subwords, not raw token ids. We
        # don't have the integer id at this layer; report id=-1.
        safe = subword.replace("'", "\\'")
        print(f"token {token_index}: {safe!r} (id=-1, {dt_ms}ms, compute={dt_ms}ms)",
              flush=True)
        per_token_times_ms.append(dt_ms)
        token_index += 1
        return False   # don't stop

    log(f"prompt: {args.prompt!r}  max_new_tokens={args.max_tokens}")
    gen_start = time.perf_counter()
    last_t = gen_start
    result = pipe.generate(args.prompt, gen_cfg, streamer)
    elapsed = time.perf_counter() - gen_start

    n = len(per_token_times_ms) if per_token_times_ms else token_index
    tps = n / elapsed if elapsed > 0 else 0.0
    print(f"Tokens: {n}", flush=True)
    print(f"Elapsed: {elapsed:.2f}s", flush=True)
    print(f"Tok/s: {tps:.2f}", flush=True)


if __name__ == "__main__":
    main()
