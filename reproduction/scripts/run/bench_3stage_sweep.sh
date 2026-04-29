#!/bin/bash
# Orchestrate K-sweep × WAN-sweep on the 3-stage v5_beam pipeline.
# Run from the Mac Mini (orchestrator).
#
# For each LATENCY_MS in {0, 10, 50, 100}:
#   1. Kill any running workers on charlie + beta
#   2. Restart workers with that LATENCY_MS
#   3. Wait for both to print "Listening on"
#   4. For each K in {2, 3, 5, 7, 10}: run mini_coord_3stage_spec_mbatch.py once
#
# Output: per-config tok/s, accept rate, bit-exact stream match.
set -euo pipefail

ALPHA=192.168.86.35
CHARLIE=192.168.86.28
BETA=192.168.86.36
KEY=~/.ssh/cascadia_ed25519
LOGDIR=/Users/tatef/Workspaces/rainier/docs/distributed_wan_sweep_3stage
mkdir -p "$LOGDIR"

LATENCIES="${LATENCIES:-0 10 50 100}"
KS="${KS:-2 3 5 7 10}"
STREAMS="${STREAMS:-2}"
MAX_TOKENS="${MAX_TOKENS:-128}"

# We build a PowerShell script per launch (so the C:\Program Files\ path stays
# safely double-quoted on the remote side regardless of how bash expands).

start_worker() {
  local host=$1 shard=$2 latency=$3 logfile=$4
  ssh -i "$KEY" "cascadia@$host" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_worker.ps1 -Shard $shard -ListenPort 19100 -NumStreams $STREAMS -LatencyMs $latency" \
    > "$logfile" 2>&1 &
  echo $!
}

run_coord() {
  local k=$1 logfile=$2
  ssh -i "$KEY" "cascadia@$ALPHA" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_coord.ps1 -NumStreams $STREAMS -K $k -MaxTokens $MAX_TOKENS -Stage1Host $CHARLIE -Stage2Host $BETA" \
    > "$logfile" 2>&1 || true
}

kill_workers() {
  for h in "$CHARLIE" "$BETA"; do
    ssh -i "$KEY" "cascadia@$h" 'powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force; Start-Sleep -Milliseconds 500"' >/dev/null 2>&1 || true
  done
}

wait_listening() {
  local logfile=$1 name=$2 timeout=${3:-180}
  local elapsed=0
  while [ $elapsed -lt $timeout ]; do
    if grep -q "Listening on" "$logfile" 2>/dev/null; then
      echo "  $name listening (${elapsed}s)"
      return 0
    fi
    if grep -q -i "error\|traceback\|exception\|failed" "$logfile" 2>/dev/null; then
      echo "  $name FAILED (${elapsed}s):"
      grep -E -i "error|exception|traceback" "$logfile" | head -3 | sed 's/^/    /'
      return 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "  $name TIMEOUT after ${timeout}s"
  return 1
}

echo "=== bench_3stage_sweep.sh starting ==="
echo "LATENCIES=$LATENCIES  KS=$KS  STREAMS=$STREAMS  MAX_TOKENS=$MAX_TOKENS"
echo "logs: $LOGDIR"
echo

for LAT in $LATENCIES; do
  echo "=== LATENCY_MS=$LAT (one-way per hop) ==="
  kill_workers
  sleep 1

  C_LOG="$LOGDIR/charlie_L${LAT}.log"
  B_LOG="$LOGDIR/beta_L${LAT}.log"
  : > "$C_LOG"; : > "$B_LOG"

  C_PID=$(start_worker "$CHARLIE" 'C:\cascadia\shards_3stage_v5_beam_stage_1' "$LAT" "$C_LOG")
  B_PID=$(start_worker "$BETA"    'C:\cascadia\shards_3stage_v5_beam_stage_2' "$LAT" "$B_LOG")
  echo "  workers spawned (charlie pid $C_PID, beta pid $B_PID), waiting for ready..."

  if ! wait_listening "$C_LOG" charlie 240; then kill_workers; continue; fi
  if ! wait_listening "$B_LOG" beta 240; then kill_workers; continue; fi

  for K in $KS; do
    R_LOG="$LOGDIR/coord_L${LAT}_K${K}.log"
    : > "$R_LOG"
    echo "  -- K=$K --"
    run_coord "$K" "$R_LOG"
    grep -E "Mean aggregate|run [0-9]:|streams produce|first10" "$R_LOG" | sed 's/^/    /'
  done

  echo "  shutting workers down..."
  kill_workers
  # Local SSH PIDs may already be dead; reap if alive
  kill "$C_PID" "$B_PID" 2>/dev/null || true
  wait "$C_PID" "$B_PID" 2>/dev/null || true
  echo
done

echo "=== sweep complete; logs in $LOGDIR ==="
