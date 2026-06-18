#!/bin/bash
# Held-out activation set for per-penalty qualitative analysis. Uses pile-10k docs
# 5000+ (DISJOINT from the RL training docs 0-1200) so explanations are on unseen
# activations. ~1000+ rows. Same Qwen2.5-7B L20 / injection config as the RL data.
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
export PYTHONPATH=/workspace/nla:${PYTHONPATH:-}
cd /workspace/nla
OUT=/workspace/heldout
mkdir -p "$OUT"
MODEL=/workspace/models/base

echo "=== [stage0] held-out activations (pile-10k docs 5000-5180) $(date) ==="
python -m nla.datagen.stage0_extract \
  --base-model "$MODEL" \
  --corpus NeelNanda/pile-10k --corpus-split train --text-column text \
  --corpus-start 5000 --corpus-length 180 --positions-per-doc 8 \
  --layer-index 20 \
  --output "$OUT/base.parquet"
python -m nla.datagen.stage1_split --base "$OUT/base.parquet" \
  --av-sft-frac 0.0 --ar-sft-frac 0.0 --rl-frac 1.0 --output-dir "$OUT/splits"
python -m nla.datagen.stage3_build --input "$OUT/splits/rl_raw.parquet" --stage rl \
  --output "$OUT/heldout.parquet" --keep-debug-metadata 2>/dev/null || \
  python -m nla.datagen.stage3_build --input "$OUT/splits/rl_raw.parquet" --stage rl --output "$OUT/heldout.parquet"
python - <<PY
import pyarrow.parquet as pq
pf=pq.ParquetFile("/workspace/heldout/heldout.parquet")
print("held-out rows:", pf.metadata.num_rows, "cols:", pf.schema_arrow.names)
PY
