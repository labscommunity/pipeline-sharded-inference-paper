#!/bin/bash
# Real-WAN sweep on the 3-stage v5_beam pipeline. Two TCP latency proxies
# run on the Mac mini (orchestrator); alpha's coordinator connects to the
# Mac for both stage_1 and stage_2 hops, the Mac proxy forwards to
# charlie/beta with LATENCY_MS one-way delay per chunk. Real TCP
# semantics (retransmit, congestion window, nagle) apply on every hop --
# unlike the worker's time.sleep() sleep-sim which only adds dead time
# inside the recv/send path.
#
# Workers must already be running on charlie:19100 and beta:19100. The
# script is idempotent against existing proxies (kills stale ones first)
# but does not restart workers between latencies.
set -euo pipefail

ALPHA=192.168.86.35
CHARLIE=192.168.86.28
BETA=192.168.86.36
MAC_LAN_IP=192.168.86.243   # coord connects here for both stages
PROXY_PY=/Users/tatef/Workspaces/rainier/scripts/tcp_latency_proxy.py
OUTDIR=/Users/tatef/Workspaces/rainier/docs/distributed_wan_sweep_3stage
mkdir -p "$OUTDIR"
KEY=~/.ssh/cascadia_ed25519

LATENCIES="${LATENCIES:-0 10 50 100}"
KS="${KS:-3 5}"
STREAMS="${STREAMS:-2}"
MAX_TOKENS="${MAX_TOKENS:-128}"

run_coord() {
  local lat=$1 k=$2 outlog=$3
  ssh -i "$KEY" "cascadia@$ALPHA" \
    "powershell -NoProfile -ExecutionPolicy Bypass -File C:\\cascadia\\scripts\\run_coord.ps1 -NumStreams $STREAMS -K $k -MaxTokens $MAX_TOKENS -Stage1Host $MAC_LAN_IP -Stage1Port 29101 -Stage2Host $MAC_LAN_IP -Stage2Port 29102" \
    > "$outlog" 2>&1 || true
}

# Make sure run_coord.ps1 supports the extra Stage1Port / Stage2Port flags
# (it does in the current version of the script).
echo "=== real-WAN 3-stage sweep ==="
echo "LATENCIES=$LATENCIES  KS=$KS  STREAMS=$STREAMS  MAX_TOKENS=$MAX_TOKENS"

# Pre-cleanup: kill any stale proxies and free the ports
pkill -f tcp_latency_proxy 2>/dev/null || true
sleep 2

for LAT in $LATENCIES; do
  echo
  echo "=== LATENCY=$LAT ms (one-way per hop, real TCP) ==="

  # Stage_1 proxy: 29101 -> charlie:19100
  LOCAL_PORT=29101 REMOTE_HOST=$CHARLIE REMOTE_PORT=19100 LATENCY_MS=$LAT \
    python3 $PROXY_PY > "$OUTDIR/realwan_proxy_charlie_L${LAT}.log" 2>&1 &
  P1=$!

  # Stage_2 proxy: 29102 -> beta:19100
  LOCAL_PORT=29102 REMOTE_HOST=$BETA REMOTE_PORT=19100 LATENCY_MS=$LAT \
    python3 $PROXY_PY > "$OUTDIR/realwan_proxy_beta_L${LAT}.log" 2>&1 &
  P2=$!

  sleep 3

  # Sanity check
  if ! lsof -iTCP:29101 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  charlie proxy failed to bind 29101"; kill $P1 $P2 2>/dev/null || true; continue
  fi
  if ! lsof -iTCP:29102 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  beta proxy failed to bind 29102";    kill $P1 $P2 2>/dev/null || true; continue
  fi

  for K in $KS; do
    LOG="$OUTDIR/realwan_coord_L${LAT}_K${K}.log"
    : > "$LOG"
    echo "  -- K=$K --"
    run_coord "$LAT" "$K" "$LOG"
    grep -E "Mean aggregate|run [0-9]:|streams produce|first10" "$LOG" | sed 's/^/    /' || true
  done

  # Tear down proxies before next latency. Long sleep helps TCP TIME_WAIT clear.
  kill $P1 $P2 2>/dev/null || true
  sleep 5
done

echo
echo "=== real-WAN 3-stage sweep complete ==="
echo "logs in $OUTDIR/realwan_*.log"
