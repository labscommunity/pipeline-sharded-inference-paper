#!/usr/bin/env bash
# reproduce_section_6_10_70b.sh — §6.10 Llama 3.1 70B 4-stage on Tiber Cloud.
#
# Topology: matias-01 (LL coord, stage_0+draft) + matias-02 (stage_1) +
# pawan-01 (stage_2) + pawan-02 (stage_3). Optional: tate-04 PL coord variant.
#
# Prerequisite: 70B 4-stage v5_beam shards are pre-deployed under
#   C:\cascadia\shards_70b_v5_beam\stage_{0,1,2,3} on each of the four nodes
# (use scripts/run/distribute_70b_shards.sh from the miner export host).
#
# This driver runs the 1-stream K=10 peak, 2-stream K=10 LL coord, 2-stream
# K=10 PL coord, and the long-context 1024-token run.

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${LOG_DIR:=$REPRO/logs/section_6_10}"
mkdir -p "$LOG_DIR"

# Defer to the existing rainier orchestrator (uses Tailscale IPs from configs/tiber_ips.env)
SCRIPT="$REPRO/scripts/run/run_70b_tiber_bench.sh"

KS="3 5 10 15" STREAMS=1 MAX_TOKENS=128 bash "$SCRIPT" 2>&1 | tee "$LOG_DIR/70b_1stream_ksweep.log"
KS="10"        STREAMS=2 MAX_TOKENS=128 bash "$SCRIPT" 2>&1 | tee "$LOG_DIR/70b_2stream_K10.log"

# Long context (1024 / 4096) — use coord on tate-04 PL for headroom
COORD_HOST=cascadia-tate-04 KS="10" STREAMS=1 MAX_TOKENS=1024 bash "$SCRIPT" 2>&1 | tee "$LOG_DIR/70b_long_1024.log"
COORD_HOST=cascadia-tate-04 KS="10" STREAMS=1 MAX_TOKENS=4096 bash "$SCRIPT" 2>&1 | tee "$LOG_DIR/70b_long_4096.log"

# Target-only baseline (correctness)
COORD_HOST=cascadia-matias-01 SCRIPT_NAME=mini_coord_nstage_target_only.py \
  bash "$SCRIPT" 2>&1 | tee "$LOG_DIR/70b_target_only.log"

echo "Section 6.10 logs in $LOG_DIR"
