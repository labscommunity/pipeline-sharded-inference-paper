#!/bin/bash
# Real-WAN sweep using a Mac-side TCP latency proxy between alpha and charlie.
# The worker on charlie stays up across latency points; we restart the proxy
# per latency value, so real TCP semantics (retransmit, nagle, etc.) are in play.

SSH="ssh -i /Users/tatef/.ssh/cascadia_ed25519"
ALPHA="cascadia@192.168.86.250"
OUTDIR="/tmp/real_wan_sweep"
mkdir -p $OUTDIR

# Kill any old proxies
pkill -f tcp_latency_proxy 2>/dev/null
sleep 2

for LAT in 0 10 50 100; do
  echo "==================== LATENCY $LAT ms (one-way) ===================="

  # Launch proxy on Mac (listens on 29100, forwards to charlie 19100)
  LOCAL_PORT=29100 REMOTE_HOST=192.168.86.28 REMOTE_PORT=19100 LATENCY_MS=$LAT \
    python3 /tmp/tcp_latency_proxy.py > $OUTDIR/proxy_L${LAT}.log 2>&1 &
  PROXY_PID=$!
  sleep 5

  # Preflight: verify proxy listens AND can reach charlie
  if ! lsof -iTCP:29100 -sTCP:LISTEN > /dev/null 2>&1; then
    echo "proxy failed to start"
    kill $PROXY_PID 2>/dev/null
    continue
  fi
  if ! nc -z -w 3 192.168.86.28 19100 > /dev/null 2>&1; then
    echo "WARN: charlie unreachable, waiting..."
    sleep 15
  fi

  # Alpha coord — connects to mac (192.168.86.243:29100) for stage_1 instead of direct to charlie
  echo "--- full stack (2-stream mbatch + spec K=3) ---"
  $SSH $ALPHA "set STAGE0_SHARD=C:\\cascadia\\shards_2stage_v5_beam\\stage_0&&set DRAFT_MODEL=C:\\cascadia\\models\\llama-3.2-1b-int4&&set STAGE1_HOST=192.168.86.243&&set STAGE1_PORT=29100&&set NUM_STREAMS=2&&set K=3&& python -u C:\\cascadia\\scripts\\mini_coord_spec_mbatch.py" 2>&1 | tee $OUTDIR/fullstack_L${LAT}.log | tail -6

  # Kill proxy before next iteration — wait for TCP state to clear
  kill $PROXY_PID 2>/dev/null
  sleep 10
done

echo ""
echo "===== REAL-WAN SWEEP DONE ====="
for f in $OUTDIR/fullstack_L*.log; do
  echo "$(basename $f):"
  grep "Mean aggregate" $f | head -1
done
