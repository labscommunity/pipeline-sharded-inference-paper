# Models

All models used in the paper, their HuggingFace IDs, and the export procedure.

## Model registry

| Symbol in paper | HuggingFace ID | Format used | Approx INT4 size | Required by |
|---|---|---|---|---|
| Llama 3.1 8B (target) | `meta-llama/Llama-3.1-8B-Instruct` | OV IR INT4 (sym, group=128) | 4.3 GB monolithic; 1.5 GB / stage | §4–§6 |
| Llama 3.2 1B (draft) | `meta-llama/Llama-3.2-1B-Instruct` | OV IR INT4 | 0.7 GB | §5 spec decode, §6 K-sweeps |
| Gemma 4 E2B | `google/gemma-4-E2B-it` | OV IR FP32 (PLI quant-sensitive) | 7 GB / stage | §6.7 Gemma |
| Llama 3.1 70B (target) | `meta-llama/Llama-3.1-70B-Instruct` | OV IR INT4 | 9 GB / shard × 4 stages = 36 GB | §6.10 70B |

## Export procedure

The paper's results use `v5_beam` shards — `torch.jit.trace` exports with precomputed rotary embeddings *and* the post-export `beam_idx` Gather injection that unlocks OV's `IndirectKVCache` GPU fusion (§4 of paper). Export scripts:

* `scripts/export/export_cached_shards_v5.py` — Llama 8B / 70B; emits `--num-stages` shards
* `scripts/export/export_gemma4_e2b_cached_shards.py` — Gemma 4 E2B (custom rotary; see Discovery #22)
* `scripts/export/inject_beam_idx_gemma.py` — post-hoc beam_idx injection for Gemma stage_0

Top-level wrapper `scripts/run/download_models.sh` chains HF download → export → INT4 compression and writes shards to a configurable `SHARDS_DIR` (default `~/cascadia/shards/`).

### Llama 3.1 8B INT4 v5_beam (1-stage / 2-stage / 3-stage)

```bash
HF_TOKEN=hf_yourtoken python scripts/export/export_cached_shards_v5.py \
  --model-id meta-llama/Llama-3.1-8B-Instruct \
  --output-dir $SHARDS_DIR/shards_2stage_v5_beam --num-stages 2
```

Substitute `--num-stages 1` for the 1-stage parity bench, `--num-stages 3` for the 3-stage in-process and distributed measurements. Each shard emits an `openvino_model.xml` + `openvino_model.bin` plus a `tokenizer/` directory.

### Llama 3.1 70B INT4 v5_beam, 4-stage

```bash
HF_TOKEN=hf_yourtoken python scripts/export/export_cached_shards_v5.py \
  --model-id meta-llama/Llama-3.1-70B-Instruct \
  --output-dir $SHARDS_DIR/shards_70b_4stage_v5_beam --num-stages 4
```

This requires ≥256 GB RAM on the export host (the FP16 calibration weights are ~141 GB during NNCF compression). On hardware with <256 GB the alternative is to export the shards on a beefier machine and rsync to the fleet — see §6.10 of the paper for the actual deploy path used.

### Llama 3.2 1B INT4 (draft)

```bash
python scripts/export/export_cached_shards_v5.py \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --output-dir $SHARDS_DIR/llama-3.2-1b-int4 --num-stages 1
```

### Gemma 4 E2B v2 / v2_beam

```bash
HF_TOKEN=hf_yourtoken python scripts/export/export_gemma4_e2b_cached_shards.py \
  --output-dir $SHARDS_DIR/shards_e2b_cached_2s_v2 --num-stages 2

# v2_beam variant: post-export beam_idx Gather injection on stage_0
python scripts/export/inject_beam_idx_gemma.py \
  $SHARDS_DIR/shards_e2b_cached_2s_v2/stage_0 \
  $SHARDS_DIR/shards_e2b_cached_2s_v2_beam/stage_0
# stage_1 is KV-shared (no ReadValue ops to inject); copy as-is
cp -r $SHARDS_DIR/shards_e2b_cached_2s_v2/stage_1 \
     $SHARDS_DIR/shards_e2b_cached_2s_v2_beam/stage_1
cp -r $SHARDS_DIR/shards_e2b_cached_2s_v2/tokenizer \
     $SHARDS_DIR/shards_e2b_cached_2s_v2_beam/tokenizer
```

## Disk requirements per node

| Node | Models needed | Approx disk |
|---|---|---|
| Single-node bench (e.g., alpha) | Llama 8B 1-stage, 2-stage, 3-stage; Llama 1B; Gemma 4 E2B 2-stage | ~25 GB |
| Charlie (stage_1 worker) | Llama 8B 2-stage stage_1; Llama 8B 3-stage stage_1; Gemma 4 E2B v2 stage_1 | ~6 GB |
| Beta (stage_2 worker) | Llama 8B 3-stage stage_2 | ~3 GB |
| Tiber matias-01 (coord) | Llama 70B 4-stage stage_0; Llama 8B for §6.8 | ~14 GB |
| Tiber matias-02 / pawan-01 / pawan-02 (workers) | Llama 70B 4-stage stage_{1,2,3} respectively | ~9 GB each |
| Tiber tate-04 (PL coord variant) | Same as matias-01 | ~14 GB |
