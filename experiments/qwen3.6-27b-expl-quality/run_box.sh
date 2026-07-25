#!/bin/bash
# Driver for the RunPod box: env setup + smoke + both generation arms.
# Layout expected under /workspace/eq (scp'd from the dev box):
#   gen_expls.py  nla/  data/manifest_quality.json  sidecar_mat/  sidecar_std/
# Sentinels: OK_SETUP OK_SMOKE OK_MAT OK_STD DONE_ALL / FAIL_<stage>
set -u
cd /workspace/eq

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { touch "FAIL_$1"; log "FAILED at $1"; exit 1; }

export HF_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/hf   # base model won't fit the container disk
mkdir -p results

if [ ! -f OK_SETUP ]; then
  log "pip install…"
  pip install -q "transformers==5.5.4" "peft==0.19.1" numpy pyyaml safetensors \
    accelerate hf_transfer pyarrow orjson "huggingface_hub>=0.34" || fail setup
  pip install -q flash-linear-attention || log "fla install failed (continuing)"
  python -c "import fla; print('fla OK')" || {
    log "fla import failed — removing torchvision/torchaudio and retrying"
    pip uninstall -q -y torchvision torchaudio
    python -c "import fla; print('fla OK after removal')" || log "fla unusable, torch fallback"
  }
  python - <<'PY' || fail setup
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| gpu", torch.cuda.get_device_name(0))
PY
  touch OK_SETUP
fi

if [ ! -f OK_SMOKE ]; then
  log "smoke: mat + std, --limit 4"
  python gen_expls.py --model mat --limit 4 --out results/smoke_mat.json > smoke_mat.log 2>&1
  grep -q GEN_DONE_MAT smoke_mat.log && [ -s results/smoke_mat.json ] || fail smoke
  python gen_expls.py --model std --limit 4 --out results/smoke_std.json > smoke_std.log 2>&1
  grep -q GEN_DONE_STD smoke_std.log && [ -s results/smoke_std.json ] || fail smoke
  python - <<'PY' || fail smoke
import json
for arm in ("mat", "std"):
    p = json.load(open(f"results/smoke_{arm}.json"))
    assert p["meta"]["cjk_explanations"] == 0, f"{arm}: CJK taint — injection failed"
    print(arm, "smoke sample:", " / ".join(p["entries"][0]["explanations"][0])[:200])
PY
  touch OK_SMOKE
fi

if [ ! -f OK_MAT ]; then
  log "full: mat (1000 ctx)"
  python gen_expls.py --model mat --out results/explanations_mat.json > mat.log 2>&1
  grep -q GEN_DONE_MAT mat.log && [ -s results/explanations_mat.json ] || fail mat
  touch OK_MAT
fi

if [ ! -f OK_STD ]; then
  log "full: std (1000 ctx)"
  python gen_expls.py --model std --out results/explanations_std.json > std.log 2>&1
  grep -q GEN_DONE_STD std.log && [ -s results/explanations_std.json ] || fail std
  touch OK_STD
fi

touch DONE_ALL
log "all done"
