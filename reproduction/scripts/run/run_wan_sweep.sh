#!/bin/bash
# Sweep WAN latency on the distributed stack.
# For each latency value L ms/hop:
#   1. Restart worker on charlie with LATENCY_MS=L
#   2. Wait for worker to be listening
#   3. Run 4 coordinator configs:
#      a) single-stream baseline (no spec, no mbatch)
#      b) 2-stream mbatch (no spec)
#      c) spec K=3 (no mbatch) — single stream with spec
#      d) full stack (spec K=3 + 2-stream mbatch)

SSH="ssh -i /Users/tatef/.ssh/cascadia_ed25519"
CHARLIE="cascadia@192.168.86.28"
ALPHA="cascadia@192.168.86.35"
OUTDIR="/tmp/wan_sweep"
mkdir -p $OUTDIR

start_worker() {
  local lat=$1
  $SSH $CHARLIE "taskkill /f /im python.exe 2>NUL & set STAGE1_SHARD=C:\\cascadia\\shards_2stage_v5_beam_stage_1&&set LISTEN_PORT=19100&&set NUM_STREAMS=2&&set LATENCY_MS=$lat&& python -u C:\\cascadia\\scripts\\mini_worker_traced.py > C:\\cascadia\\worker_trace.log 2>&1" &
  SSH_PID=$!
  echo "Worker started (PID $SSH_PID), waiting for port..."
  # Wait up to 90s
  for i in $(seq 1 18); do
    if $SSH $CHARLIE "netstat -an | findstr LISTENING | findstr 19100" 2>/dev/null | grep -q 19100; then
      echo "Worker ready at L=$lat ms/hop."
      return 0
    fi
    sleep 5
  done
  echo "WORKER FAILED TO START at L=$lat"
  return 1
}

for LAT in 0 10 50 100; do
  echo ""
  echo "==================== LATENCY $LAT ms/hop ===================="
  start_worker $LAT || { echo "abort"; exit 1; }

  # Config D only: full stack (since it's the headline number)
  echo ""
  echo "--- Full stack (spec K=3 + 2-stream mbatch) ---"
  $SSH $ALPHA "set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0&&set DRAFT_MODEL=C:\\cascadia\\models\\llama-3.2-1b-int4&&set STAGE1_HOST=192.168.86.28&&set STAGE1_PORT=19100&&set NUM_STREAMS=2&&set K=3&& python -u C:\\cascadia\\scripts\\mini_coord_spec_mbatch.py" 2>&1 | tee $OUTDIR/fullstack_L${LAT}.log | tail -8

  # Give worker a chance to exit cleanly before next restart
  sleep 2
done

echo ""
echo "===== SWEEP DONE ====="
for f in $OUTDIR/fullstack_L*.log; do
  echo ""
  echo "=== $f ==="
  grep -E "Mean aggregate|first10|match" $f | head -4
done
