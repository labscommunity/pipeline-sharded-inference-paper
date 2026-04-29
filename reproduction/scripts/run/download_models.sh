#!/usr/bin/env bash
# download_models.sh — fetch HuggingFace models and export to OV INT4 v5_beam shards.
#
# Reads:
#   HF_TOKEN     — required for gated models (Llama, Gemma)
#   SHARDS_DIR   — destination root (default ~/cascadia/shards)
#   MODELS       — space-separated subset to export (default "all"). See list below.
#
# Models exported (by tag):
#   llama-8b-1stage     — Llama 3.1 8B INT4 v5_beam, 1 stage. Used for §4 Tab 1, §5 spec.
#   llama-8b-2stage     — Llama 3.1 8B INT4 v5_beam, 2 stages (16+16). §6.2-§6.5.
#   llama-8b-3stage     — Llama 3.1 8B INT4 v5_beam, 3 stages (11+11+10). §6.6, §6.K_sweep.
#   llama-1b-draft      — Llama 3.2 1B INT4 (single stage, draft model). §5, §6, §6.10.
#   gemma-e2b-2stage    — Gemma 4 E2B FP32, 2 stages (rotary fix, "v2"). §6.7.
#   gemma-e2b-2stage-beam — same + post-export beam_idx Gather injection on stage_0.
#   gemma-e2b-1stage    — Gemma 4 E2B FP32, 1 stage. §6.7 single-node.
#   llama-70b-4stage    — Llama 3.1 70B INT4 v5_beam, 4 stages (20 layers each). §6.10.
#                          Requires ≥256 GB RAM during NNCF compression.

set -uo pipefail

: "${HF_TOKEN:?Set HF_TOKEN env var (HuggingFace token with access to Llama / Gemma)}"
: "${SHARDS_DIR:=$HOME/cascadia/shards}"
: "${MODELS:=all}"

mkdir -p "$SHARDS_DIR"
SCRIPTS=$(cd "$(dirname "$0")/.." && pwd)
EXPORT="$SCRIPTS/export"
echo "Shards root:  $SHARDS_DIR"
echo "Scripts root: $SCRIPTS"

want() {
  case "$MODELS" in
    all) return 0 ;;
    *"$1"*) return 0 ;;
    *) return 1 ;;
  esac
}

if want llama-8b-1stage; then
  echo "=== Exporting Llama 3.1 8B INT4 v5_beam, 1 stage ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_cached_shards_v5.py" \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --output-dir "$SHARDS_DIR/llama-8b-1stage_v5_beam" --num-stages 1
fi

if want llama-8b-2stage; then
  echo "=== Exporting Llama 3.1 8B INT4 v5_beam, 2 stages ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_cached_shards_v5.py" \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --output-dir "$SHARDS_DIR/llama-8b-2stage_v5_beam" --num-stages 2
fi

if want llama-8b-3stage; then
  echo "=== Exporting Llama 3.1 8B INT4 v5_beam, 3 stages ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_cached_shards_v5.py" \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --output-dir "$SHARDS_DIR/llama-8b-3stage_v5_beam" --num-stages 3
fi

if want llama-1b-draft; then
  echo "=== Exporting Llama 3.2 1B INT4 (draft) ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_cached_shards_v5.py" \
    --model-id meta-llama/Llama-3.2-1B-Instruct \
    --output-dir "$SHARDS_DIR/llama-3.2-1b-int4" --num-stages 1
fi

if want gemma-e2b-1stage; then
  echo "=== Exporting Gemma 4 E2B FP32, 1 stage (v2 rotary fix) ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_gemma4_e2b_cached_shards.py" \
    --output-dir "$SHARDS_DIR/shards_e2b_1stage_v2" --num-stages 1
fi

if want gemma-e2b-2stage; then
  echo "=== Exporting Gemma 4 E2B FP32, 2 stages (v2 rotary fix) ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_gemma4_e2b_cached_shards.py" \
    --output-dir "$SHARDS_DIR/shards_e2b_cached_2s_v2" --num-stages 2
fi

if want gemma-e2b-2stage-beam; then
  echo "=== Injecting beam_idx Gather into Gemma stage_0 (v2_beam) ==="
  python "$EXPORT/inject_beam_idx_gemma.py" \
    "$SHARDS_DIR/shards_e2b_cached_2s_v2/stage_0" \
    "$SHARDS_DIR/shards_e2b_cached_2s_v2_beam/stage_0"
  cp -r "$SHARDS_DIR/shards_e2b_cached_2s_v2/stage_1" \
        "$SHARDS_DIR/shards_e2b_cached_2s_v2_beam/stage_1"
  cp -r "$SHARDS_DIR/shards_e2b_cached_2s_v2/tokenizer" \
        "$SHARDS_DIR/shards_e2b_cached_2s_v2_beam/tokenizer"
  # 1-stage variant
  python "$EXPORT/inject_beam_idx_gemma.py" \
    "$SHARDS_DIR/shards_e2b_1stage_v2/stage_0" \
    "$SHARDS_DIR/shards_e2b_1stage_v2_beam/stage_0"
  cp -r "$SHARDS_DIR/shards_e2b_1stage_v2/tokenizer" \
        "$SHARDS_DIR/shards_e2b_1stage_v2_beam/tokenizer"
fi

if want llama-70b-4stage; then
  echo "=== Exporting Llama 3.1 70B INT4 v5_beam, 4 stages (~28 min, ≥256 GB RAM!) ==="
  HF_TOKEN="$HF_TOKEN" python "$EXPORT/export_cached_shards_v5.py" \
    --model-id meta-llama/Llama-3.1-70B-Instruct \
    --output-dir "$SHARDS_DIR/llama-70b-4stage_v5_beam" --num-stages 4
fi

echo ""
echo "Done. Shards under $SHARDS_DIR:"
ls -la "$SHARDS_DIR" 2>/dev/null
