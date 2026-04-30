#!/usr/bin/env bash
# reproduce_section_6_6_3stage.sh — §6.6 3-stage full stack and 3-stream concurrency.
#
# Drives the rainier 3-node fleet (alpha + charlie + beta) for the new §6.6
# table_progression_3stage rows: 42.91 K=5 LAN, 54.92 K=5 3-stream, 14.71 K=10 L=100.

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${ALPHA:=cascadia@192.168.86.250}"
: "${CHARLIE:=cascadia@192.168.86.28}"
: "${BETA:=cascadia@192.168.86.36}"
: "${LOG_DIR:=$REPRO/logs/section_6_6}"
mkdir -p "$LOG_DIR"
SSH="ssh -i ~/.ssh/cascadia_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# Use the existing rainier orchestrator if it's deployed
$SSH "$ALPHA" "if exist C:\\cascadia\\scripts\\bench_3stage_sweep.sh (echo deployed) else (echo missing)" 2>&1 | tail -1
$SSH "$ALPHA" "cd /d C:\\cascadia & bash scripts/bench_3stage_sweep.sh" 2>&1 | tee "$LOG_DIR/3stage_sweep.log"

# 3-stream LAN
$SSH "$ALPHA" "cd /d C:\\cascadia & bash scripts/run_3stream_lan.sh" 2>&1 | tee "$LOG_DIR/3stream_lan.log"

echo "Section 6.6 logs in $LOG_DIR"
