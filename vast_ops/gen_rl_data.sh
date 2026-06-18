#!/bin/bash
# Generate a small RL parquet for continue-RL from the released Qwen2.5-7B NLA.
# Activations: Qwen2.5-7B-Instruct, layer 20, raw (norm=none). Prompt: stage3 default
# AV template (byte-identical to the released AV sidecar's `av` template). NO API.
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
export PYTHONPATH=/workspace/nla:${PYTHONPATH:-}
cd /workspace/nla
OUT=/workspace/rldata
mkdir -p "$OUT"
MODEL=/workspace/models/base

echo "=== [stage0] extract Qwen2.5-7B L20 activations (NeelNanda/pile-10k) $(date) ==="
python -m nla.datagen.stage0_extract \
  --base-model "$MODEL" \
  --corpus NeelNanda/pile-10k --corpus-split train --text-column text \
  --corpus-start 0 --corpus-length 1200 --positions-per-doc 8 \
  --layer-index 20 \
  --output "$OUT/base.parquet"

echo "=== [stage1] split (rl=1.0) $(date) ==="
python -m nla.datagen.stage1_split \
  --base "$OUT/base.parquet" \
  --av-sft-frac 0.0 --ar-sft-frac 0.0 --rl-frac 1.0 \
  --output-dir "$OUT/splits"

echo "=== [stage3] build rl parquet $(date) ==="
python -m nla.datagen.stage3_build \
  --input "$OUT/splits/rl_raw.parquet" --stage rl --output "$OUT/rl.parquet"

echo "=== DONE $(date) ==="
python - <<'PY'
import pyarrow.parquet as pq
pf = pq.ParquetFile("/workspace/rldata/rl.parquet")
print("rl rows:", pf.metadata.num_rows)
print("cols:", pf.schema_arrow.names)
PY
