#!/bin/bash
# Run the 3-stage v5_beam pipeline at NUM_STREAMS=3 on the LAN, K=3 and K=5.
# Workers + coord both use per-stream compile_model (workaround for the
# OV reset_state multi-InferRequest bug, paper §6.2). Each compile_model
# adds ~2 GB iGPU memory per stream; this is the limit on stream count
# given Lunar Lake's 16 GB shared budget.

set -euo pipefail
ALPHA=192.168.86.35
CHARLIE=192.168.86.28
BETA=192.168.86.36
KEY=~/.ssh/cascadia_ed25519
LOGDIR=/Users/tatef/Workspaces/rainier/docs/distributed_wan_sweep_3stage
mkdir -p "$LOGDIR"

KS="${KS:-3 5}"
STREAMS="${STREAMS:-3}"
MAX_TOKENS="${MAX_TOKENS:-128}"

# Kill any existing workers
for h in "$CHARLIE" "$BETA"; do
  ssh -i "$KEY" "cascadia@$h" 'powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"' 2>&1 >/dev/null || true
done
sleep 2

# Start workers with NUM_STREAMS=3
C_LOG="$LOGDIR/charlie_3stream.log"
B_LOG="$LOGDIR/beta_3stream.log"
: > "$C_LOG"; : > "$B_LOG"

ssh -i "$KEY" "cascadia@$CHARLIE" \
  "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_worker.ps1 -Shard C:\\cascadia\\shards_3stage_v5_beam_stage_1 -ListenPort 19100 -NumStreams $STREAMS -LatencyMs 0" \
  > "$C_LOG" 2>&1 &
C_PID=$!
ssh -i "$KEY" "cascadia@$BETA" \
  "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_worker.ps1 -Shard C:\\cascadia\\shards_3stage_v5_beam_stage_2 -ListenPort 19100 -NumStreams $STREAMS -LatencyMs 0" \
  > "$B_LOG" 2>&1 &
B_PID=$!

echo "Started workers (charlie pid $C_PID, beta pid $B_PID); waiting for them to compile $STREAMS streams..."

# Wait for both to print "Listening on" (3 streams take longer to compile)
ready_c=0; ready_b=0
for _ in $(seq 1 90); do
  sleep 4
  if [ $ready_c -eq 0 ] && grep -q "Listening on" "$C_LOG"; then ready_c=1; echo "  charlie listening"; fi
  if [ $ready_b -eq 0 ] && grep -q "Listening on" "$B_LOG"; then ready_b=1; echo "  beta listening"; fi
  if grep -q -i "error\|exception\|traceback\|failed" "$C_LOG" "$B_LOG"; then
    echo "  worker error!"; grep -i "error\|exception\|traceback" "$C_LOG" "$B_LOG" | head -5; break
  fi
  [ $ready_c -eq 1 ] && [ $ready_b -eq 1 ] && break
done

if [ $ready_c -ne 1 ] || [ $ready_b -ne 1 ]; then
  echo "  workers did not become ready; aborting"
  kill $C_PID $B_PID 2>/dev/null || true
  exit 1
fi

echo
for K in $KS; do
  R_LOG="$LOGDIR/coord_3stream_K${K}.log"
  : > "$R_LOG"
  echo "=== NUM_STREAMS=$STREAMS  K=$K ==="
  ssh -i "$KEY" "cascadia@$ALPHA" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_coord.ps1 -NumStreams $STREAMS -K $K -MaxTokens $MAX_TOKENS -Stage1Host $CHARLIE -Stage2Host $BETA" \
    > "$R_LOG" 2>&1 || true
  grep -E "Mean aggregate|run [0-9]:|streams produce|first10" "$R_LOG" | sed 's/^/  /'
  echo
done

# Tear down workers
for h in "$CHARLIE" "$BETA"; do
  ssh -i "$KEY" "cascadia@$h" 'powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"' 2>&1 >/dev/null || true
done
kill $C_PID $B_PID 2>/dev/null || true
wait $C_PID $B_PID 2>/dev/null || true
echo "done"
