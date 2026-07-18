#!/bin/bash
# Driver for the RunPod box: env setup + smoke + both scoring arms.
# Layout expected under /workspace/hm (rsynced from the dev box):
#   score_subsets.py  nla/  data/mu.npy
#   se/data/manifest.json  se/results/explanations_{mat,std}.json
#   se/sidecar_mat/  se/sidecar_std/
# Sentinels: OK_SETUP OK_SMOKE OK_MAT OK_STD DONE_ALL / FAIL_<stage>
set -u
cd /workspace/hm

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { touch "FAIL_$1"; log "FAILED at $1"; exit 1; }

export HF_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1

if [ ! -f OK_SETUP ]; then
  log "pip install…"
  pip install -q "transformers==5.5.4" numpy pyyaml safetensors accelerate \
    hf_transfer pyarrow orjson "huggingface_hub>=0.34" || fail setup
  pip install -q flash-linear-attention || log "fla install failed (continuing)"
  python -c "import fla; print('fla OK')" || {
    log "fla import failed — removing torchvision/torchaudio and retrying"
    pip uninstall -q -y torchvision torchaudio
    python -c "import fla; print('fla OK after removal')" || log "fla unusable, torch fallback"
  }
  python - <<'PY' || fail setup
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| gpu", torch.cuda.get_device_name(0))
PY
  touch OK_SETUP
fi

if [ ! -f OK_SMOKE ]; then
  log "smoke: mat --limit 5"
  python score_subsets.py --arm mat --suffix-eval se --limit 5 \
    --workdir smoke_work --out results/smoke_mat.json > smoke.log 2>&1
  grep -q SCORE_DONE_MAT smoke.log && [ -s results/smoke_mat.json ] || fail smoke
  rm -rf smoke_work results/smoke_mat.json
  touch OK_SMOKE
fi

if [ ! -f OK_MAT ]; then
  log "full: mat"
  python score_subsets.py --arm mat --suffix-eval se \
    --out results/subset_scores_mat.json > mat.log 2>&1
  grep -q SCORE_DONE_MAT mat.log && [ -s results/subset_scores_mat.json ] || fail mat
  touch OK_MAT
fi

if [ ! -f OK_STD ]; then
  log "full: std"
  python score_subsets.py --arm std --suffix-eval se \
    --out results/subset_scores_std.json > std.log 2>&1
  grep -q SCORE_DONE_STD std.log && [ -s results/subset_scores_std.json ] || fail std
  touch OK_STD
fi

touch DONE_ALL
log "all done"
