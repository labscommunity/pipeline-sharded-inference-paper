#!/usr/bin/env bash
# reproduce_section_4.sh — §4 Reaching Monolithic Parity (Tab 1 + Tab 2)
#
# Runs the four configurations in tab:shard_parity (A, A', B_v5_beam, B_v3_fp32)
# plus the splitting-cost rows in tab:shard_stages (1-stage, 2-stage, 3-stage).
# Designed to run on a single Panther Lake or Lunar Lake AI PC ("alpha"
# in the rainier testbed). Requires shards from download_models.sh.

set -uo pipefail

: "${REPRO:=$(cd "$(dirname "$0")/../.." && pwd)}"
: "${SHARDS_DIR:=$HOME/cascadia/shards}"
: "${LOG_DIR:=$REPRO/logs/section_4}"
: "${MAX_TOKENS:=128}"
: "${N:=5}"
mkdir -p "$LOG_DIR"

PY=python
SCRIPTS=$REPRO/scripts

cd "$SHARDS_DIR/.." 2>/dev/null || true   # so models/ resolves on relative paths

run_workload () {
  local label="$1"; shift
  local workload="$1"; shift
  echo "=== Section 4 :: workload $label ($workload) ==="
  $PY "$SCRIPTS/bench/bench_methodology.py" \
    --workload "$workload" \
    --n "$N" \
    --max-tokens "$MAX_TOKENS" \
    --output "$LOG_DIR/methodology_${label}.json" \
    2>&1 | tee "$LOG_DIR/methodology_${label}.log"
  echo "=== done $label ==="
}

# Tab 1 rows
run_workload A         a
run_workload A_prime   ap
run_workload B_v5_beam b_v5_beam
run_workload B_v3_fp32 b_v3_fp32

# Tab 2 — splitting cost via simple 1-stage / 2-stage / 3-stage benches
echo "=== Section 4 :: Tab 2 1-stage v5_beam ==="
$PY "$SCRIPTS/bench/bench_v5.py" "$SHARDS_DIR/llama-8b-1stage_v5_beam/stage_0" 2>&1 | tee "$LOG_DIR/tab2_1stage.log"

# (2-stage and 3-stage in-process target-only is captured by bench_spec_matrix.py
#  in section 5 — it uses the 3-stage shards as the B baseline. To get the
#  2-stage in-process row, run bench_spec_v7_sharded.py with TARGET_STAGES set
#  to two paths; see §5 driver.)

# Build a measured-vs-paper JSON for compare_to_paper.py
cat > "$LOG_DIR/measured.json" <<EOF
{
  "section": "4.shard_parity",
  "rows": [
    {"label": "A_genai_mono",       "paper": 22.96, "measured": $(grep "mean = " "$LOG_DIR/methodology_A.log" | tail -1 | sed 's/.*mean = //; s/ .*//')},
    {"label": "A_prime_mono_python","paper": 20.61, "measured": $(grep "mean = " "$LOG_DIR/methodology_A_prime.log" | tail -1 | sed 's/.*mean = //; s/ .*//')},
    {"label": "B_v5_beam_1stage",   "paper": 22.08, "measured": $(grep "mean = " "$LOG_DIR/methodology_B_v5_beam.log" | tail -1 | sed 's/.*mean = //; s/ .*//')},
    {"label": "B_v3_fp32_no_beamidx","paper": 20.05, "measured": $(grep "mean = " "$LOG_DIR/methodology_B_v3_fp32.log" | tail -1 | sed 's/.*mean = //; s/ .*//')},
    {"label": "tab2_1stage_v5_beam","paper": 22.08, "measured": $(grep "mean = " "$LOG_DIR/tab2_1stage.log" | tail -1 | sed 's/mean = //; s/ .*//')}
  ]
}
EOF

$PY "$SCRIPTS/run/compare_to_paper.py" "$LOG_DIR/measured.json" --output "$LOG_DIR/report.json" \
  | tee "$LOG_DIR/report.txt"
