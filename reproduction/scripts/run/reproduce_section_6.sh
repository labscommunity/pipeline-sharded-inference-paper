#!/usr/bin/env bash
# reproduce_section_6.sh — §6 Distributed Pipeline Evaluation
#
# Tab 6 progression (14.51 → 29.46 → 41.25); Tab 7 breakdown; Tab 8 compression;
# Tab 9 sleep-sim WAN; Tab 10 real-WAN queue-proxy; Tab 11 K-sweep.
# Requires the rainier 2-node fleet (alpha + charlie) plus beta for Tab 9
# baseline rows. Workers must be started on charlie / beta before the coord
# bench fires; this script does that via SSH.
#
# ENV:
#   ALPHA      — coord host SSH alias (default: cascadia@192.168.86.250)
#   CHARLIE    — stage_1 worker SSH alias (default: cascadia@192.168.86.28)
#   BETA       — stage_2 worker SSH alias (default: cascadia@192.168.86.36)
#   SHARDS_DIR — shards root on each node (must be pre-deployed by download_models.sh)

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${ALPHA:=cascadia@192.168.86.250}"
: "${CHARLIE:=cascadia@192.168.86.28}"
: "${BETA:=cascadia@192.168.86.36}"
: "${LOG_DIR:=$REPRO/logs/section_6}"
mkdir -p "$LOG_DIR"
SSH="ssh -i ~/.ssh/cascadia_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

start_worker () {
  local host="$1"; shift
  local shard="$1"; shift
  local nstreams="${1:-1}"
  $SSH "$host" "taskkill /f /im python.exe 2>NUL; cd /d C:\\cascadia & set NUM_STREAMS=$nstreams& set STAGE1_SHARD=$shard& set LISTEN_PORT=19100& set DEVICE=GPU& start /B python -u scripts\\mini_worker_stage1.py > C:\\cascadia\\worker.log 2>&1" &
  WORKER_PIDS+=( $! )
}

stop_workers () {
  for h in "$CHARLIE" "$BETA"; do
    $SSH "$h" "taskkill /f /im python.exe 2>NUL" 2>/dev/null || true
  done
  WORKER_PIDS=()
}

WORKER_PIDS=()
trap stop_workers EXIT

# === Tab 6 row: 2-stage v5_beam single-stream baseline (14.51) ===
echo "=== §6.2 2-stage single-stream ==="
start_worker "$CHARLIE" "C:\\cascadia\\shards_2stage_v5_beam\\stage_1" 1
sleep 60   # worker compile
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0& set STAGE1_HOST=192.168.86.28& set STAGE1_PORT=19100& set NUM_STREAMS=1& set MAX_TOKENS=128& python -u scripts\\mini_coord_stage0.py" 2>&1 | tee "$LOG_DIR/2stage_singlestream.log"
stop_workers

# === Tab 6 row: 2-stage 2-stream micro-batch (29.46) ===
echo "=== §6.2 2-stage 2-stream mbatch ==="
start_worker "$CHARLIE" "C:\\cascadia\\shards_2stage_v5_beam\\stage_1" 2
sleep 90
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0& set STAGE1_HOST=192.168.86.28& set NUM_STREAMS=2& set MAX_TOKENS=128& python -u scripts\\mini_coord_mbatch.py" 2>&1 | tee "$LOG_DIR/2stage_mbatch.log"
stop_workers

# === Tab 6 row: full stack 2-stream + spec K=3 (41.25) ===
echo "=== §6.2 full stack mbatch + spec K=3 ==="
start_worker "$CHARLIE" "C:\\cascadia\\shards_2stage_v5_beam\\stage_1" 2
sleep 90
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0& set DRAFT_MODEL=C:\\cascadia\\models\\llama-3.2-1b-int4& set STAGE1_HOST=192.168.86.28& set NUM_STREAMS=2& set K=3& set MAX_TOKENS=128& python -u scripts\\mini_coord_spec_mbatch.py" 2>&1 | tee "$LOG_DIR/2stage_full_stack.log"
stop_workers

# === Tab 9 sleep-sim WAN sweep (LATENCY_MS in {0,10,50,100}) ===
echo "=== §6 Tab 9 sleep-sim WAN sweep ==="
for L in 0 10 50 100; do
  start_worker "$CHARLIE" "C:\\cascadia\\shards_2stage_v5_beam\\stage_1" 2
  sleep 90
  $SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0& set DRAFT_MODEL=C:\\cascadia\\models\\llama-3.2-1b-int4& set STAGE1_HOST=192.168.86.28& set NUM_STREAMS=2& set K=3& set LATENCY_MS=$L& set MAX_TOKENS=128& python -u scripts\\mini_coord_spec_mbatch.py" 2>&1 | tee "$LOG_DIR/wan_L${L}.log"
  stop_workers
done

# === Tab 11 K-sweep on 3-stage target-only ===
echo "=== §6 Tab 11 K-sweep ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u scripts\\bench_spec_wan_K.py" 2>&1 | tee "$LOG_DIR/k_sweep.log"

# === Tab 10 real-WAN queue-proxy (on-alpha proxies) ===
echo "=== §6.5 real-WAN queue-proxy ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u scripts\\wan_sweep_v2.py" 2>&1 | tee "$LOG_DIR/real_wan.log"

echo ""
echo "Section 6 logs in $LOG_DIR/"
