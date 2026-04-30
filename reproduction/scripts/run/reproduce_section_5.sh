#!/usr/bin/env bash
# reproduce_section_5.sh — §5 Speculative Decoding via Mask-Based KV Rewind
#
# Tab 3 trim cost; Tab 4 spec_lan 8-prompt; Tab 5 spec_shard 4-row;
# long-generation 128/512/1024/2048-token speedup.

set -uo pipefail
: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${SHARDS_DIR:=$HOME/cascadia/shards}"
: "${LOG_DIR:=$REPRO/logs/section_5}"
mkdir -p "$LOG_DIR"
PY=python
SCRIPTS=$REPRO/scripts

# Tab 3: trim_kv variants (CPU + GPU latency)
echo "=== §5 Tab 3 trim cost ==="
$PY "$SCRIPTS/bench/trim_experiment.py" 2>&1 | tee "$LOG_DIR/trim_experiment.log"

# Tab 4: 8-prompt spec_lan sweep
echo "=== §5 Tab 4 spec_lan ==="
$PY "$SCRIPTS/bench/bench_prompt_sweep.py" 2>&1 | tee "$LOG_DIR/prompt_sweep.log"

# Tab 5: spec_shard 4-row matrix
echo "=== §5 Tab 5 spec_shard matrix ==="
$PY "$SCRIPTS/bench/bench_spec_matrix.py" 2>&1 | tee "$LOG_DIR/spec_matrix.log"

# Long-generation
echo "=== §5 long-generation 128/256/1024 ==="
$PY "$SCRIPTS/bench/bench_spec_long_gen.py" 2>&1 | tee "$LOG_DIR/long_gen.log"

echo "=== §5 long-generation 2048 ==="
$PY "$SCRIPTS/bench/bench_spec_very_long.py" 2>&1 | tee "$LOG_DIR/very_long.log"

# Mask vs physical trim: bit-exact verification (correctness, not perf)
echo "=== §5 mask-vs-physical bit-exact verify ==="
$PY "$SCRIPTS/bench/mask_trim_test.py" 2>&1 | tee "$LOG_DIR/mask_trim_verify.log"

echo ""
echo "Section 5 logs in $LOG_DIR/"
