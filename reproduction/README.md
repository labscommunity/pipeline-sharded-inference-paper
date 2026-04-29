# Reproduction Package

Self-contained scripts to reproduce every measurement in the paper *Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets* (revision 2026-04-28).

This folder is vendored from the rainier companion repo (`labscommunity/rainier`) at the paper's source-of-truth commit chain. See `CLAIMS.md` for a paper-table-row → script map.

## Layout

```
reproduction/
├── CLAIMS.md             # Paper-row → script mapping; every cited number traced to its producer
├── MODELS.md             # HuggingFace model IDs, INT4 export commands, expected output sizes
├── HARDWARE.md           # Testbed + Tiber fleet + per-table required topology
├── README.md             # This file
├── configs/              # Per-table parameters (prompts, K values, latencies)
├── scripts/
│   ├── export/           # torch.jit.trace → OV IR INT4 + beam_idx Gather injection
│   ├── bench/            # Single-node benches (shard parity, spec decode, K-sweep)
│   ├── coord/            # Pipeline coordinators (mini_coord_*, gemma_2s_*)
│   ├── worker/           # Stage workers (mini_worker_stage1, gemma_2s_worker, ...)
│   ├── proxy/            # WAN simulators (sleep-sim worker option, queue-proxy on alpha)
│   ├── run/              # Orchestration shells (multi-K sweeps, 70B fan-out)
│   └── cascadia/         # Vendored helpers (spec_decode, activation_relay, model.loader)
├── logs/                 # Output of reproduction runs is written here (gitignored)
└── docs/                 # Per-section reproduction notes
```

## Quick start

Three ways to use this package depending on what you want to verify:

1. **Single-node benches only** (rainier `alpha` Panther Lake or any Lunar Lake AI PC). Reproduces §4 (Tab 1, 2), §5 (Tab 3, 4, 5), §6.6 (Gemma single-node, Gemma in-process), and the spec-decode WAN K-sweep. Requires Llama 3.1 8B INT4 + Llama 3.2 1B INT4 + Gemma 4 E2B FP32 weights.

2. **Two-node distributed** (rainier `alpha` + `charlie`). Adds Tab 6 (progression), Tab 7 (breakdown), Tab 8 (compression), Tab 9 (sleep-sim WAN), Tab 12 (Gemma 4 multi-node).

3. **Full fleet** (rainier alpha + charlie + beta + Tiber matias-01/02 + pawan-01/02 + tate-04). Adds §6.6 3-stage / 3-stream, §6.8 Tiber DERP, §6.10 70B 4-stage.

## Reproduction commands

Each subsection of the paper has a corresponding `reproduce_*.sh` driver in `scripts/run/`. Drivers wire up workers, run benches, and emit JSON output to `logs/<section>_<datestamp>.json`. Run from this folder:

```bash
# Hardware setup (one-time per machine)
bash scripts/run/setup_env.sh           # python deps, OV install verification
bash scripts/run/download_models.sh     # fetch models from HF, export INT4 v5_beam shards

# §4 Reaching Monolithic Parity (Tab 1, Tab 2)
bash scripts/run/reproduce_section_4.sh

# §5 Speculative Decoding (Tab 3, Tab 4, Tab 5, long-gen)
bash scripts/run/reproduce_section_5.sh

# §6 Distributed Pipeline (Tab 6 - Tab 11) — requires 2-node fleet
bash scripts/run/reproduce_section_6.sh

# §6.7 Gemma 4 E2B (Tab 12) — 1-node + 2-node
bash scripts/run/reproduce_section_6_7_gemma.sh

# §6.6 3-stage / 3-stream LAN — requires 3-node fleet
bash scripts/run/reproduce_section_6_6_3stage.sh

# §6.8 Tiber Cloud DERP (8B 2-node) — requires Tiber Cloud account
bash scripts/run/reproduce_section_6_8_tiber.sh

# §6.10 70B 4-stage — requires Tiber Cloud account + 4 Lunar Lake instances
bash scripts/run/reproduce_section_6_10_70b.sh
```

## Tolerance policy

A reported number is considered reproduced if the measured value is within **±5%** of the paper's. The drivers emit a per-row PASS/FAIL flag against this threshold; rows outside ±5% are re-run up to 3× before being flagged for human attention.

## Sensitive data

No HuggingFace tokens, SSH keys, or other credentials are bundled. The Gemma 4 export (gated model on HF) reads `HF_TOKEN` from the environment — set it before running `download_models.sh`. The Tiber Cloud orchestration reads SSH config from your `~/.ssh/config`; see `HARDWARE.md` for the recommended block.
