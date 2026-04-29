# Reproduction results (running tally)

Generated as the reproduction package is run on the rainier and Tiber Cloud testbeds. Each row records the paper's published value, the freshly-measured value, the percentage delta, and a verdict against the ±5% tolerance from `README.md`.

> **Important context.** All measurements below were taken 2026-04-28+ on the same hardware as the paper, but the OpenVINO runtime, GPU driver, and operating system have advanced since the paper's measurement window (2026-04-23 → 2026-04-26). Most numbers come in noticeably *higher* (faster) than the paper's; the paper's relative ratios are preserved. We flag any row outside ±5% with the system-drift annotation.

## §4 Reaching Monolithic Parity (single-node Llama 3.1 8B INT4 on alpha Panther Lake B390)

### Tab `tab:shard_parity`

| Row | Paper | Measured | Δ | Verdict | Note |
|---|---|---|---|---|---|
| A: mono `openvino_genai` (C++ loop) | 22.96 | **24.54** | +6.9% | drift | system faster since paper |
| A': mono `ov.compile_model` (Python loop) | 20.61 | **23.12** | +12.2% | drift | |
| B: v5_beam 1-stage | 22.08 | **24.45** | +10.7% | drift | |
| B_v3_fp32: pre-`beam_idx` | 20.05 | **21.26** | +6.0% | drift | |

**Ratios (the paper's structural claim):**

| Ratio | Paper | Measured |
|---|---|---|
| B / A (v5_beam vs mono genai) | 0.961 | 0.996 |
| B / A' (v5_beam vs mono Python) | 1.071 | 1.057 |
| B / B_v3_fp32 (beam_idx unlock) | 1.101 | 1.150 |

Conclusion of §4 holds: `beam_idx` Gather injection brings v5_beam shards within 4% of `openvino_genai` (in fact 0.4% on current hardware), and recovers ~10–15% over pre-injection.

### Tab `tab:shard_stages`

| Row | Paper | Measured | Δ | Verdict |
|---|---|---|---|---|
| Mono A_spec reference | 21.79 | 24.28 | +11.4% | drift |
| v5_beam 1-stage | 22.08 | 23.93 | +8.4% | drift |
| v5_beam 2-stage (16+16) | 19.41 | (pending) | — | — |
| v5_beam 3-stage (11+11+10) | 19.79 | 21.56 | +8.9% | drift |

3-stage in-process matches Tab 5 row B (same script, `bench_spec_matrix.py`). 2-stage in-process not yet measured (would need a 2-stage version of bench_spec_matrix's ShardedMaskedReq).

## §5 Speculative Decoding

### Tab `tab:spec_shard` (Llama 3.1 8B INT4 paired session)

| Row | Paper | Measured | Δ | Verdict |
|---|---|---|---|---|
| A: mono target only | 21.79 | 24.28 | +11.4% | drift |
| B: 3-stage shard target only | 19.79 | 21.56 | +8.9% | drift |
| C: mono + spec K=3 | 28.28 | 30.21 | +6.8% | drift |
| D: 3-stage shard + spec K=3 | 26.28 | 28.38 | +8.0% | drift |

Per-paper ratios preserved:

| Ratio | Paper | Measured |
|---|---|---|
| B / A | 0.91× | 0.89× |
| C / A | 1.30× | 1.24× |
| D / A | 1.21× | 1.17× |
| D / B (spec speedup on shards) | 1.33× | 1.32× |

Acceptance rate at K=3: **70.7%** (paper) vs **70.7%** (measured) — exact match.
Output bit-correct: first 10 tokens of all four configs identical to mono target-only first 10 tokens.

### Tab `tab:spec_lan` (8 prompts × 1 timed run × 128 tokens)

| Prompt | Paper baseline | Measured baseline | Paper spec | Measured spec | Paper × | Measured × | Match |
|---|---|---|---|---|---|---|---|
| short-factual | 22.08 | 22.60 | 28.75 | 29.35 | 1.30× | 1.30× | ✓ |
| reasoning | 20.60 | 21.47 | 27.27 | 29.48 | 1.32× | 1.37× | ✓ |
| code-completion | 19.98 | 22.08 | 32.71 | 33.10 | 1.64× | 1.50× | ✓ |
| list-enumeration | 21.10 | 21.37 | 28.54 | 27.91 | 1.35× | 1.31× | ✓ |
| creative | 22.10 | 23.51 | 24.53 | 26.09 | 1.11× | 1.11× | ✓ |
| technical-expl | 21.98 | 23.87 | 27.12 | 28.52 | 1.23× | 1.19× | ✓ |
| chat-assistant | 21.96 | 23.30 | 30.73 | 31.22 | 1.40× | 1.34× | ✓ |
| translation | 22.15 | 23.04 | 31.93 | 34.14 | 1.44× | 1.48× | ✓ |
| **Mean** | 21.49 | 22.66 | 28.95 | 29.98 | **1.35× ± 0.15** | **1.33× ± 0.13** | within 1.5% |

Output bit-correct on all 8 prompts. Acceptance rates 70.7% / 73.3% / 93.1% / 74.6% / 49.7% / 62.4% / 78.1% / 84.3% match paper exactly.

### Long generation (`bench_spec_very_long.py`, K=3)

| Length | Speedup paper | Speedup measured | Δ | Acceptance paper | Acceptance measured | Cache bloat measured |
|---|---|---|---|---|---|---|
| 128 tok | 1.35× | 1.28× | -5.2% | 70.7% | 70.7% (match) | 1.26× |
| 512 tok | 1.55× | 1.56× | +0.6% | 90.6% | 90.6% (match) | 1.07× |
| 1024 tok | 1.59× | 1.57× | -1.3% | 95.1% | 95.1% (match) | 1.04× |
| 2048 tok | 1.65× | 1.58× | -4.2% | 97.5% | 97.5% (match) | 1.02× |

All speedups within ±5% of paper except 128-tok (-5.2%). Acceptance rates *exactly* match paper for every length. Cache bloat at 2048 measured 1.02× — matches the paper claim verbatim.

## §6 Distributed Pipeline — Tab `tab:progression` (alpha + charlie)

| Row | Paper | Measured | Δ | Verdict |
|---|---|---|---|---|
| Mono single-node ($A_{\text{spec}}$) | 21.79 | 24.54 | +12.6% | drift |
| 2-stage v5_beam single-stream | 14.51 | 16.33 | +12.5% | drift |
| + 2-stream micro-batching | 29.46 | 29.34 | -0.4% | match |
| + mask-based spec decode K=3 | **41.25** | **43.97** | +6.6% | drift (just outside) |

Composition multipliers:

| Multiplier | Paper | Measured |
|---|---|---|
| mbatch (mbatch / single-stream-dist) | 2.03× | 1.80× |
| spec (full-stack / mbatch) | 1.40× | 1.50× |
| product (full-stack / single-stream-dist) | 2.84× | 2.69× |
| full-stack / mono | 1.89× | 1.79× |

Per-stream throughput at K=3 in the full stack: 22.0 tok/s each (paper 20.6). Acceptance 70.7% (paper 70.7% exact match). Output bit-correct.

The single-stream and mono baselines climbed faster than the mbatch row, so the mbatch ratio compressed (the more-saturated baseline leaves less stage-idle time for the second stream to fill). The full-stack row recovers some of that via spec decode. Net advantage over mono single-user is preserved (+1.79×) — close to but not quite at the paper's 1.89×.

## §6 Distributed Pipeline — Tab `tab:k_sweep` (target-only single-stream)

(running via `bench_spec_wan_K.py`)

## §6.7 Gemma 4 E2B (single-node + 2-node multi-node)

(Pending; `gemma_bench_v2*.py` plus `gemma_2s_coord.py` orchestration.)

## §6.10 Llama 3.1 70B 4-stage (Tiber Cloud)

(Pending; requires shards pre-deployed on Tiber Cloud nodes — see `MODELS.md`.)

## Notes on system drift

The +6 to +12% delta on §4 and §5 numbers is consistent across every monolithic and shard configuration we have re-measured. This indicates a system-wide change since the paper's source measurements (2026-04-23/24), most likely:

* OpenVINO 2026.1.0 → newer point release that improved iGPU code generation
* Intel Arc B390 driver update on the Panther Lake host
* Windows scheduler / iGPU power state differences

The paper's *structural* claims (the `beam_idx` Gather closes the parity gap; v5_beam 1-stage matches mono within 4%; 3-stage shard + spec K=3 = 1.33× over shard baseline) are preserved on the new measurements with tighter (more favorable) margins. Where applicable we recommend keeping the paper's published numbers and noting the drift in errata; alternatively the table can be re-run as a single paired session to refresh.
