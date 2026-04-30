#!/bin/bash
# Distribute 70B v5_beam shards from beta (rainier home LAN) to the Tiber
# Cloud fleet. Beta isn't on Tailscale, so each transfer hops via this Mac
# (scp -3 / explicit pull-then-push).
#
# Plan: 7-stage 70B export on beta → 7 Tiber nodes total
#   stage_0 → matias-01 (coord)
#   stage_1 → matias-02
#   stage_2 → pawan-01
#   stage_3 → pawan-02
#   stage_4 → tate-01
#   stage_5 → tate-02
#   stage_6 → tate-03  (final stage; will run with SEND_TOPK=1)
#
# Each shard ~5 GB INT4 (35 GB total) = ~30 min over Tailscale at 35 Mbps
# per flow, parallelised across 6 destinations.

set -euo pipefail

BETA=cascadia@192.168.86.36
KEY=~/.ssh/cascadia_ed25519
SRC_DIR=C:/cascadia/shards_70b_v5_beam
DST_DIR=C:/cascadia/shards_70b_v5_beam
LOCAL_STAGING=/tmp/shards_70b_v5_beam
mkdir -p "$LOCAL_STAGING"

# stage → tiber alias mapping
declare -a STAGES
STAGES[0]="cascadia-matias-01"
STAGES[1]="cascadia-matias-02"
STAGES[2]="cascadia-pawan-01"
STAGES[3]="cascadia-pawan-02"
STAGES[4]="cascadia-tate-01"
STAGES[5]="cascadia-tate-02"
STAGES[6]="cascadia-tate-03"

echo "=== Pulling shards from beta to Mac staging ==="
for stage in 0 1 2 3 4 5 6; do
  if [ ! -d "$LOCAL_STAGING/stage_$stage" ]; then
    echo "  pull stage_$stage..."
    scp -i "$KEY" -r "$BETA:$SRC_DIR/stage_$stage" "$LOCAL_STAGING/" 2>&1 | tail -2
  else
    echo "  stage_$stage already in staging, skipping"
  fi
done
echo "  pulled $(du -sh $LOCAL_STAGING | cut -f1) total"

echo
echo "=== Pushing each shard to its target Tiber node (parallel) ==="
for stage in 0 1 2 3 4 5 6; do
  alias=${STAGES[$stage]}
  log=/tmp/push_70b_stage${stage}.log
  echo "  stage_$stage → $alias"
  ssh -o ConnectTimeout=10 "$alias" "powershell -NoProfile -Command 'New-Item -ItemType Directory -Path $DST_DIR -Force | Out-Null; New-Item -ItemType Directory -Path $DST_DIR/stage_$stage -Force | Out-Null'" 2>&1 | grep -v "Permanently added" | tail -1
  scp -o ConnectTimeout=15 -r "$LOCAL_STAGING/stage_$stage/." "$alias:$DST_DIR/stage_$stage/" \
    > "$log" 2>&1 &
done
wait
echo
echo "=== Verify shard sizes on each Tiber node ==="
for stage in 0 1 2 3 4 5 6; do
  alias=${STAGES[$stage]}
  echo -n "  stage_$stage on $alias: "
  ssh -o ConnectTimeout=10 "$alias" "powershell -NoProfile -Command '\$s = (Get-ChildItem $DST_DIR/stage_$stage -File 2>\$null | Measure-Object -Property Length -Sum).Sum; Write-Host \"\$([Math]::Round(\$s/1MB,0)) MB\"'" 2>&1 | grep -v "Permanently added" | tail -1
done

echo "=== distribute_70b_shards.sh done ==="
