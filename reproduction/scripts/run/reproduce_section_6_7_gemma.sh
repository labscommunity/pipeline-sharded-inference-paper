#!/usr/bin/env bash
# reproduce_section_6_7_gemma.sh — §6.7 Gemma 4 E2B (Tab tab:gemma_dist).
#
# Single-node v2 / v2_beam for 1-stage and 2-stage; 2-node multi-node;
# CPU→GPU stage_0; 2-stream micro-batch (multi-node aggregate).

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${ALPHA:=cascadia@192.168.86.250}"
: "${CHARLIE:=cascadia@192.168.86.28}"
: "${LOG_DIR:=$REPRO/logs/section_6_7}"
mkdir -p "$LOG_DIR"
SSH="ssh -i ~/.ssh/cascadia_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"

# In-process v2 (rotary fix)
echo "=== Gemma 1-stage v2 (in-process) ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u gemma_bench_v2_1s.py" 2>&1 | tee "$LOG_DIR/gemma_v2_1s.log"

echo "=== Gemma 2-stage v2 in-process ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u gemma_bench_v2_2s.py" 2>&1 | tee "$LOG_DIR/gemma_v2_2s.log"

# In-process v2_beam
echo "=== Gemma 1-stage v2_beam (in-process) ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u gemma_bench_v2beam_1s.py" 2>&1 | tee "$LOG_DIR/gemma_v2beam_1s.log"

echo "=== Gemma 2-stage v2_beam in-process ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & python -u gemma_bench_v2beam_2s.py" 2>&1 | tee "$LOG_DIR/gemma_v2beam_2s.log"

# Multi-node 2-stage v2 (charlie worker)
echo "=== Gemma 2-stage multi-node GPU->GPU v2 ==="
$SSH "$CHARLIE" "taskkill /f /im python.exe 2>NUL; cd /d C:\\cascadia & set SHARD_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2_stage_1& set LISTEN_PORT=19101& set DEVICE=GPU& start /B python -u scripts\\gemma_2s_worker.py > C:\\cascadia\\worker.log 2>&1" &
sleep 30
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\stage_0& set TOK_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\tokenizer& set STAGE1_HOST=192.168.86.28& set STAGE1_PORT=19101& set DEVICE=GPU& set MAX_TOKENS=50& set N_RUNS=5& python -u scripts\\gemma_2s_coord.py" 2>&1 | tee "$LOG_DIR/gemma_2stage_multinode_v2.log"

# Multi-node 2-stage v2 CPU->GPU (stage_0 on CPU)
echo "=== Gemma 2-stage multi-node CPU->GPU v2 ==="
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\stage_0& set TOK_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\tokenizer& set STAGE1_HOST=192.168.86.28& set STAGE1_PORT=19101& set DEVICE=CPU& set MAX_TOKENS=50& set N_RUNS=3& python -u scripts\\gemma_2s_coord.py" 2>&1 | tee "$LOG_DIR/gemma_2stage_multinode_cpugpu_v2.log"

# Multi-node 2-stage v2_beam GPU->GPU
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2_beam\\stage_0& set TOK_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2_beam\\tokenizer& set STAGE1_HOST=192.168.86.28& set STAGE1_PORT=19101& set DEVICE=GPU& set MAX_TOKENS=50& set N_RUNS=5& python -u scripts\\gemma_2s_coord.py" 2>&1 | tee "$LOG_DIR/gemma_2stage_multinode_v2beam.log"

# Multi-node 2-stream mbatch (NUM_STREAMS=2 on both sides)
echo "=== Gemma 2-stream micro-batch (multi-node aggregate) ==="
$SSH "$CHARLIE" "taskkill /f /im python.exe 2>NUL; cd /d C:\\cascadia & set SHARD_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2_stage_1& set LISTEN_PORT=19101& set DEVICE=GPU& set NUM_STREAMS=2& start /B python -u scripts\\gemma_2s_mbatch_worker.py > C:\\cascadia\\worker.log 2>&1" &
sleep 30
$SSH "$ALPHA" "cd /d C:\\cascadia & set STAGE0_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\stage_0& set TOK_DIR=C:\\cascadia\\shards_e2b_cached_2s_v2\\tokenizer& set STAGE1_HOST=192.168.86.28& set STAGE1_PORT=19101& set DEVICE=GPU& set NUM_STREAMS=2& set MAX_TOKENS=50& set N_RUNS=3& python -u scripts\\gemma_2s_mbatch_coord.py" 2>&1 | tee "$LOG_DIR/gemma_2stream_mbatch_v2.log"

echo "Section 6.7 logs in $LOG_DIR"
