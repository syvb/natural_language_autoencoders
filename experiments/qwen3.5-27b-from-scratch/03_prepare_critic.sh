#!/bin/bash
# Step 3 — build the from-scratch critic init: base model truncated to layers
# 0..LAYER_INDEX (num_hidden_layers = LAYER_INDEX+1) + identity-initialized
# value head + nla_meta.yaml copied from the ar_sft dataset sidecar.
# Output: $WORK/models/critic_init  (this is AR_HF_CKPT for run_ar_sft.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_config.sh"

OUT_DIR="${CRITIC_INIT_DIR:-$WORK/models/critic_init}"
# cd out of any dir whose child is named `nla` (e.g. /workspace with the repo
# at /workspace/nla): `python -m` puts cwd on sys.path and the repo dir then
# shadows the installed package as a namespace package — nla.scripts resolves
# to the repo's top-level scripts/ and the module is "not found".
cd "$HERE"
"${PYTHON:-python}" -m nla.scripts.prepare_critic_checkpoint \
    --base-model "${BASE_MODEL_DIR:-$WORK/models/base}" \
    --num-layers "$LAYER_INDEX" \
    --dataset-sidecar "$WORK/out/ar_sft.parquet" \
    --output "$OUT_DIR"
echo "CRITIC_INIT_DONE $OUT_DIR"
