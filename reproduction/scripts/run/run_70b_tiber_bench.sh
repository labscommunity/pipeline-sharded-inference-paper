#!/bin/bash
# Run the 7-stage 70B fan-out benchmark on the Tiber Cloud fleet.
# Topology:
#   matias-01 (Lunar Lake) — coord + stage_0 + draft (Llama 3.2 1B INT4)
#   matias-02 (Lunar Lake) — stage_1 worker
#   pawan-01  (Lunar Lake) — stage_2 worker
#   pawan-02  (Lunar Lake) — stage_3 worker
#   tate-01   (Arrow Lake) — stage_4 worker
#   tate-02   (Arrow Lake) — stage_5 worker
#   tate-03   (Arrow Lake) — stage_6 worker (final, SEND_TOPK=1 for top-1 logits compression)
#
# Prerequisite: shards distributed via distribute_70b_shards.sh.
# Tailnet IPs are pinned (don't rotate) and should match those recorded in
# cascadia-fleet/docs/HARDWARE_INVENTORY.md.

set -euo pipefail

# stage → (alias, tailnet IP, is_arrow_lake) mapping
declare -a NODE_ALIAS NODE_TAILIP IS_AL
NODE_ALIAS=(cascadia-matias-01 cascadia-matias-02 cascadia-pawan-01 cascadia-pawan-02 cascadia-tate-01 cascadia-tate-02 cascadia-tate-03)
NODE_TAILIP=(100.88.94.47       100.77.178.45     100.127.88.82     100.75.226.6      100.113.0.105    100.88.34.75     100.73.141.16)
IS_AL=(0 0 0 0 1 1 1)  # 1 = Arrow Lake-S

KS="${KS:-3}"
STREAMS="${STREAMS:-2}"
MAX_TOKENS="${MAX_TOKENS:-128}"
LOGDIR=/Users/tatef/Workspaces/rainier/docs/distributed_wan_sweep_3stage
mkdir -p "$LOGDIR"

# Build STAGE_HOSTS env var for the coord (stages 1..6, in order)
STAGE_HOSTS="${NODE_TAILIP[1]}:19100,${NODE_TAILIP[2]}:19100,${NODE_TAILIP[3]}:19100,${NODE_TAILIP[4]}:19100,${NODE_TAILIP[5]}:19100,${NODE_TAILIP[6]}:19100"
echo "STAGE_HOSTS=$STAGE_HOSTS"

# Step 1: kill any stale workers on all 7 nodes
echo "=== killing stale workers ==="
for alias in "${NODE_ALIAS[@]}"; do
  ssh -o ConnectTimeout=10 "$alias" 'powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"' 2>&1 | grep -v "Permanently added" | tail -1 || true
done
sleep 2

# Step 2: start workers on stages 1-6 (final stage gets SEND_TOPK=1)
echo "=== starting 6 workers ==="
for stage in 1 2 3 4 5 6; do
  alias=${NODE_ALIAS[$stage]}
  shard="C:\\cascadia\\shards_70b_v5_beam\\stage_$stage"
  is_final=$([ $stage -eq 6 ] && echo 1 || echo 0)
  log="$LOGDIR/70b_worker_stage${stage}.log"
  echo "  stage $stage on $alias (SEND_TOPK=$is_final)"
  ssh -o ConnectTimeout=10 "$alias" "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_worker.ps1 -Shard $shard -ListenPort 19100 -NumStreams $STREAMS -LatencyMs 0 -SendTopK $is_final" \
    > "$log" 2>&1 &
done

# Step 3: wait for all 6 workers to print "Listening on"
echo "=== waiting for workers to compile + listen (3 streams of 70B is slow) ==="
ready_count=0
for _ in $(seq 1 90); do
  sleep 10
  ready_count=0
  for stage in 1 2 3 4 5 6; do
    log="$LOGDIR/70b_worker_stage${stage}.log"
    if grep -q "Listening on" "$log" 2>/dev/null; then
      ready_count=$((ready_count + 1))
    fi
    if grep -qiE "error|exception|traceback" "$log" 2>/dev/null; then
      echo "  stage $stage error:"
      grep -iE "error|exception|traceback" "$log" | head -3
    fi
  done
  echo "  ready: $ready_count/6"
  [ $ready_count -eq 6 ] && break
done
[ $ready_count -ne 6 ] && { echo "not all workers ready, aborting"; exit 1; }

# Step 4: run the coord on matias-01
for K in $KS; do
  R_LOG="$LOGDIR/70b_coord_K${K}.log"
  : > "$R_LOG"
  echo "=== running coord K=$K ==="
  # Pass STAGE_HOSTS as env via a small wrapper PS file we'll generate inline.
  ps_cmd="\$env:NUM_STREAMS='$STREAMS'; \$env:K='$K'; \$env:MAX_TOKENS='$MAX_TOKENS'; \$env:STAGE_HOSTS='$STAGE_HOSTS'; \$env:STAGE0_SHARD='C:\\cascadia\\shards_70b_v5_beam\\stage_0'; \$env:TARGET_MODEL='C:\\cascadia\\models\\llama-3.1-70b'; \$env:DRAFT_MODEL='C:\\cascadia\\models\\llama-3.2-1b-int4'; & 'C:\\cascadia\\inference\\venv\\Scripts\\python.exe' C:\\cascadia\\scripts\\mini_coord_nstage_spec_mbatch.py"
  ssh -o ConnectTimeout=10 "${NODE_ALIAS[0]}" "powershell -NoProfile -Command \"$ps_cmd\"" \
    > "$R_LOG" 2>&1 || true
  grep -E "Mean aggregate|run [0-9]:|streams produce|first10" "$R_LOG" | sed 's/^/  /'
done

# Step 5: tear down workers
echo "=== teardown ==="
for alias in "${NODE_ALIAS[@]}"; do
  ssh -o ConnectTimeout=10 "$alias" 'powershell -NoProfile -Command "Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force"' 2>&1 | grep -v "Permanently added" | tail -1 || true
done

echo "=== run_70b_tiber_bench.sh done ==="
