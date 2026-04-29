#!/usr/bin/env bash
# reproduce_section_6_8_tiber.sh — §6.8 Tiber Cloud 2-node 8B over DERP.
#
# Topology: matias-01 (LL coord + stage_0) + matias-02 (LL stage_1 worker).
# Communicate over Tailscale DERP relay (~16 ms RTT, SEA region).
# Validates §6.8 tab:tiber: 3.97 baseline; 22.12 with top-1 logits compression.
#
# Prerequisite: shards_2stage_v5_beam pre-deployed on matias-01 and matias-02.

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${LOG_DIR:=$REPRO/logs/section_6_8}"
mkdir -p "$LOG_DIR"

# Tailscale IP of matias-02 (read from configs/tiber_ips.env)
source "$REPRO/configs/tiber_ips.env"
COORD_HOST="cascadia-matias-01"
WORKER_HOST="cascadia-matias-02"
WORKER_TS_IP="$MATIAS_02_TS_IP"

# Step 1: start worker on matias-02 listening on Tailscale port 19100
ssh "$WORKER_HOST" "powershell -Command 'taskkill /f /im python.exe 2>$null; \$env:STAGE1_SHARD=\"C:\\cascadia\\shards_2stage_v5_beam\\stage_1\"; \$env:LISTEN_PORT=\"19100\"; \$env:NUM_STREAMS=\"2\"; \$env:DEVICE=\"GPU\"; Start-Process -WindowStyle Hidden python -ArgumentList \"-u\",\"C:\\cascadia\\scripts\\mini_worker_stage1.py\" -RedirectStandardOutput C:\\cascadia\\worker_tiber.log -RedirectStandardError C:\\cascadia\\worker_tiber.err'" &
sleep 90  # worker compile

# Step 2: run coord on matias-01 — full FP32 logits baseline
echo "=== Tiber 8B 2-node DERP, full FP32 logits ==="
ssh "$COORD_HOST" "powershell -Command '\$env:STAGE0_SHARD=\"C:\\cascadia\\shards_2stage_v5_beam\\stage_0\"; \$env:DRAFT_MODEL=\"C:\\cascadia\\models\\llama-3.2-1b-int4\"; \$env:STAGE1_HOST=\"$WORKER_TS_IP\"; \$env:STAGE1_PORT=\"19100\"; \$env:NUM_STREAMS=\"2\"; \$env:K=\"3\"; \$env:MAX_TOKENS=\"128\"; \$env:SEND_TOPK=\"0\"; & \"C:\\Program Files\\Python311\\python.exe\" C:\\cascadia\\scripts\\mini_coord_spec_mbatch.py'" 2>&1 | tee "$LOG_DIR/tiber_8b_full_fp32.log"

# Step 3: run coord with top-1 logits compression
echo "=== Tiber 8B 2-node DERP, top-1 logits compression ==="
ssh "$COORD_HOST" "powershell -Command '\$env:STAGE0_SHARD=\"C:\\cascadia\\shards_2stage_v5_beam\\stage_0\"; \$env:DRAFT_MODEL=\"C:\\cascadia\\models\\llama-3.2-1b-int4\"; \$env:STAGE1_HOST=\"$WORKER_TS_IP\"; \$env:STAGE1_PORT=\"19100\"; \$env:NUM_STREAMS=\"2\"; \$env:K=\"3\"; \$env:MAX_TOKENS=\"128\"; \$env:SEND_TOPK=\"1\"; & \"C:\\Program Files\\Python311\\python.exe\" C:\\cascadia\\scripts\\mini_coord_spec_mbatch.py'" 2>&1 | tee "$LOG_DIR/tiber_8b_topk1.log"

# Cleanup
ssh "$WORKER_HOST" "powershell -Command 'taskkill /f /im python.exe 2>$null'" || true

echo "Section 6.8 logs in $LOG_DIR"
