#!/bin/bash
# Sequential warm-start on a single GPU: AV (actor) then AR (critic), both with wandb.
set -e
cd /workspace
echo "=== AV_SFT_START $(date -u +%FT%TZ) ==="
bash run_av_sft.sh
echo "=== AV_SFT_COMPLETE $(date -u +%FT%TZ) ==="
bash run_ar_sft.sh
echo "=== AR_SFT_COMPLETE $(date -u +%FT%TZ) ==="
echo "=== BOTH_SFT_COMPLETE ==="
