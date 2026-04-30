# CLAIMS.md — paper-row → script map

Every numerical claim in `main.tex` traced to the script that produces it, with the expected output and the tolerance.

## §4 Reaching Monolithic Parity

### Table `tab:shard_parity` (line 131-134)

| Cell | Value | Producing script | Output line |
|---|---|---|---|
| A: mono `openvino_genai` | 22.96 tok/s | `bench/bench_methodology.py` | `MEAN A: 22.96 ± 0.5` |
| A': mono `ov.compile_model` (Python loop) | 20.61 tok/s | same | `MEAN A': 20.61 ± 0.4` |
| B: v5_beam 1-stage | 22.08 tok/s | same | `MEAN B: 22.08 ± 0.3` |
| B_v3_fp32 (pre-beam_idx) | 20.05 tok/s | same | `MEAN B_v3: 20.05 ± 0.2` |

Driver: `scripts/run/reproduce_section_4.sh`. Tolerance: ±5%. The methodology bench requires `Llama-3.1-8B-Instruct` available at both `--target-mono` (HF transformers via optimum-intel) and `--target-shard` (1-stage v5_beam shard).

### Table `tab:shard_stages` (line 151-154)

| Cell | Value | Producing script |
|---|---|---|
| Mono A_spec reference | 21.79 tok/s | `bench/bench_v5.py --mono` |
| v5_beam 1-stage | 22.08 tok/s | `bench/bench_v5.py --shard 1` |
| v5_beam 2-stage (16+16) | 19.41 tok/s | `bench/bench_v5.py --shard 2` |
| v5_beam 3-stage (11+11+10) | 19.79 tok/s | `bench/bench_v5.py --shard 3` |

Driver: same as Tab 1.

## §5 Speculative Decoding

### Table `tab:trim_cost` (line 186-188)

| Variant | ms/call | Producing script |
|---|---|---|
| `np.asarray + slice + new Tensor` | 49.5 | `bench/trim_experiment.py` |
| In-place `Tensor.set_shape()` | 46.6 | same |
| `ov.Tensor` allocate + `np.asarray(new)[:]` | 49.9 | same |

### Table `tab:spec_lan` (line 213-222) — 8 prompts

| Prompt category | Baseline | Spec | Speedup | Producing script |
|---|---|---|---|---|
| short-factual | 22.08 | 28.75 | 1.30× | `bench/bench_prompt_sweep.py` |
| reasoning | 20.60 | 27.27 | 1.32× | same |
| code-completion | 19.98 | 32.71 | 1.64× | same |
| list-enumeration | 21.10 | 28.54 | 1.35× | same |
| creative | 22.10 | 24.53 | 1.11× | same |
| technical-expl | 21.98 | 27.12 | 1.23× | same |
| chat-assistant | 21.96 | 30.73 | 1.40× | same |
| translation | 22.15 | 31.93 | 1.44× | same |
| **Mean** | **21.49** | **28.95** | **1.35× ± 0.15** | same |

Driver: `scripts/run/reproduce_section_5.sh`. Each row is a single 128-token decode.

### Table `tab:spec_shard` (line 243-246)

| Config | Tok/s | Speedup | Producing script |
|---|---|---|---|
| Mono A | 21.79 | — | `bench/bench_spec_matrix.py` |
| 3-stage v5_beam (B) | 19.79 | 0.91× | same |
| Mono + spec K=3 | 28.28 | 1.30× / own=1.30× | same |
| 3-stage shard + spec K=3 | 26.28 | 1.21× / own=1.33× | same |

### Long generation (line 410-413, in §5)

| Length | Speedup | Accept | Producing script |
|---|---|---|---|
| 128 tok | 1.35× | 70.7% | `bench/bench_spec_v7_masked.py` |
| 512 tok | 1.55× | 90.6% | `bench/bench_spec_long_gen.py` |
| 1024 tok | 1.59× | 95.1% | `bench/bench_spec_long_gen.py` |
| 2048 tok | 1.65× | 97.5% | `bench/bench_spec_very_long.py` |

## §6 Distributed Pipeline

### Table `tab:progression` (line 278-281)

| Config | Tok/s | Producing script |
|---|---|---|
| Mono single-node ($A_{\text{spec}}$) | 21.79 | `bench/bench_v5.py --mono` |
| 2-stage v5_beam, single stream | 14.51 | `coord/mini_coord_stage0.py` + `worker/mini_worker_stage1.py` |
| + 2-stream micro-batching | 29.46 | `coord/mini_coord_mbatch.py` |
| + mask-based spec decode K=3 | **41.25** | `coord/mini_coord_spec_mbatch.py` |

Driver: `scripts/run/reproduce_section_6.sh`.

### Table `tab:breakdown` (line 304-309)

| Component | Time | Producing script |
|---|---|---|
| Stage 0 compute | 23 ms | `bench/bench_per_token_breakdown.py --stage 0` |
| Stage 1 compute | 25 ms | `bench/bench_per_token_breakdown.py --stage 1` |
| TCP RTT (1 hop) | 6 ms | `proxy/tcp_latency_bench.py` |
| Python + OV dispatch | 15 ms | derived (total − stage0 − stage1 − TCP) |

### Table `tab:compression` (line 326-330)

| Compression | Tok/s | Producing script |
|---|---|---|
| None (FP32) | 13.57 | `bench/bench_compression.py --mode fp32` |
| FP16 | 12.71 | `bench/bench_compression.py --mode fp16` |
| INT8 | 15.10 | `bench/bench_compression.py --mode int8` |

### Table `tab:wan` (line 348-351)

Sleep-sim WAN sweep, 2-stage v5_beam, 2-stream + spec K=3.

| L (ms/hop) | Naïve baseline | Full stack | Producing script |
|---|---|---|---|
| 0 | 14.51 | 41.01 | `coord/mini_coord_spec_mbatch.py LATENCY_MS=0` |
| 10 | 9.89 | 37.35 | LATENCY_MS=10 |
| 50 | 4.72 | 18.25 | LATENCY_MS=50 |
| 100 | 2.77 | 11.20 | LATENCY_MS=100 |

Run via `scripts/run/run_wan_sweep.sh`.

### Table `tab:real_wan` (line 367-373)

3-stage Llama target-only, on-alpha release-time-queue proxy.

| L (ms/hop) | Tok/s | Producing script |
|---|---|---|
| 0 | 12.53 | `proxy/wan_sweep_v2.py LATENCY_MS=0` |
| 10 | 8.18 | LATENCY_MS=10 |
| 50 | 3.28 | LATENCY_MS=50 |
| 100 | 2.01 | LATENCY_MS=100 |

### Table `tab:k_sweep` (line 388-394)

3-stage v5_beam target-only single stream. Driver: `bench/bench_spec_wan_K.py` (single paired session, runs LAN/50/100 in sequence).

| K | LAN | 50 ms | 100 ms |
|---|---|---|---|
| 2 | 28.63 | 10.27 | 6.47 |
| 3 | 26.77 | 11.54 | 7.31 |
| 5 | 30.42 | 14.05 | 9.28 |
| 7 | 30.96 | 15.27 | 10.51 |
| 10 | 28.04 | 16.04 | 11.30 |

### §6.6 Table `tab:progression_3stage`

| Config | Tok/s | Producing script |
|---|---|---|
| 3-stage 2-stream K=5 LAN | 42.91 | `coord/mini_coord_3stage_spec_mbatch.py NUM_STREAMS=2 K=5 LATENCY_MS=0` |
| 3-stage 3-stream K=5 LAN | 54.92 | NUM_STREAMS=3 K=5 LATENCY_MS=0 |
| 3-stage 2-stream K=10 L=100 | 14.71 | NUM_STREAMS=2 K=10 LATENCY_MS=100 |

Run via `scripts/run/bench_3stage_sweep.sh`.

### §6.7 Table `tab:topk_compression`

| Config | Full FP32 | Top-1 | Speedup |
|---|---|---|---|
| 3-stage 2-stream K=3 LAN | 42.42 | 47.38 | 1.12× |
| Tiber 2-node 8B over DERP, K=3 | 3.97 | 22.12 | 5.57× |

Producing script: `coord/mini_coord_3stage_spec_mbatch.py SEND_TOPK=1 ...` for the LAN row; `coord/mini_coord_spec_mbatch.py SEND_TOPK=1 STAGE1_HOST=<tiber-tailscale-ip>` for the Tiber row.

### §6.8 Table `tab:tiber`

Tiber Cloud DERP-relayed 2-node 8B.

| Config | Tok/s agg | Per-stream |
|---|---|---|
| Full FP32 logits | 3.97 | 1.99 |
| + Top-1 logits compression | 22.12 | 11.07 |

### §6.7 Gemma — Table `tab:gemma_dist`

| Config | 2026-04 paper | v2 | v2_beam | Producing script |
|---|---|---|---|---|
| 1-stage single-node GPU | 13.3 | 13.69 | 14.13 | `bench/gemma_bench_v2_1s.py` and `bench/gemma_bench_v2beam_1s.py` |
| 2-stage localhost GPU→GPU | 12.1 | 12.32 | 13.13 | `bench/gemma_bench_v2_2s.py` and `bench/gemma_bench_v2beam_2s.py` |
| 2-stage multi-node GPU→GPU | 8.12 | 10.49 | 10.66 | `coord/gemma_2s_coord.py` (GPU stage_0) |
| 2-stage multi-node CPU→GPU | 7.16 | 9.77 | — | `coord/gemma_2s_coord.py DEVICE=CPU` |
| 2-stream micro-batch (multi-node, agg) | 13.51 | 16.39 | 16.15 | `coord/gemma_2s_mbatch_coord.py` |

### §6.10 Table `tab:llama70b`

| Config | Tok/s | Per-stream | Producing script |
|---|---|---|---|
| Target-only 4-stage 1-stream | 1.74 | 1.74 | `coord/mini_coord_nstage_target_only.py NUM_STAGES=4` |
| Spec K=3 1-stream | 3.86 | 3.86 | `coord/mini_coord_nstage_spec_mbatch.py K=3 NUM_STREAMS=1 NUM_STAGES=4` |
| Spec K=5 1-stream | 4.78 | 4.78 | K=5 |
| Spec K=10 1-stream | 5.42 | 5.42 | K=10 |
| Spec K=15 1-stream | 4.76 | 4.76 | K=15 |
| Spec K=10 2-stream LL coord | 5.95 | 2.95 | NUM_STREAMS=2 (matias-01 coord) |
| Spec K=10 2-stream PL coord | 6.43 | 3.21 | NUM_STREAMS=2 (tate-04 coord) |
| K=10 1024-tok | 5.72 | 5.72 | MAX_TOKENS=1024 |
| K=10 4096-tok | 5.00 | 5.00 | MAX_TOKENS=4096 |

Run via `scripts/run/run_70b_tiber_bench.sh`.

## §7 Multi-User Throughput via Micro-Batching

### Table `tab:microbatch` (line 600-602)

| Model | Export | Single-stream | 2-stream | Producing script |
|---|---|---|---|---|
| Llama 3.1 8B | external-rotary v1 | 14.1 | 19.4 | (historical; pre-paper) |
| Llama 3.1 8B | v5_beam (this work) | 14.51 | 29.46 | `coord/mini_coord_mbatch.py` |
| Gemma 4 E2B | external-rotary v1 | 8.12 | 13.51 | (historical; pre-paper) |

The `external-rotary v1` row is from a 2026-04-01 internal export iteration that predates the v5_beam pipeline. It is included as comparison context only; the v2 / v2_beam Gemma micro-batch numbers (16.39 / 16.15) are in `tab:gemma_dist`.

## Negative results (§8.5)

| Claim | Value | Producing script |
|---|---|---|
| INT8 KV cache | 4% slower (20.86 vs 21.74) | `bench/bench_kv_prec.py` |
| Async overlap K=10 / 100 ms | 11.08 vs 11.05 | `bench/bench_spec_threaded.py` |
| User Python share of wall time | 0.6% | `bench/bench_feed_overhead.py` |
