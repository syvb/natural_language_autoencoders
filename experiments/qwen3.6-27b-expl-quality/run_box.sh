#!/bin/bash
# Driver for the RunPod box: env setup + smoke + both generation arms.
# Layout expected under /workspace/eq (scp'd from the dev box):
#   gen_expls.py  nla/  data/manifest_quality.json  sidecar_mat/  sidecar_std/
# Sentinels: OK_SETUP OK_SMOKE OK_MAT OK_STD DONE_ALL / FAIL_<stage>
set -u
cd /workspace/eq || exit 1

# torch 2.7.1+cu126 is installed for python3.10 ONLY on this image (system
# python3 = 3.12, torch-less) — pin everything to it
PY=python3.10

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { touch "FAIL_$1"; log "FAILED at $1"; exit 1; }

export HF_TOKEN=$(cat /root/.hf_token)
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/hf   # base model won't fit the container disk
mkdir -p results

if [ ! -f OK_SETUP ]; then
  log "pip install…"
  $PY -m pip install -q "transformers==5.5.4" "peft==0.19.1" numpy pyyaml \
    safetensors accelerate hf_transfer pyarrow orjson "huggingface_hub>=0.34" \
    || fail setup
  $PY -m pip install -q flash-linear-attention || log "fla install failed (continuing)"
  $PY -c "import fla; print('fla OK')" || {
    log "fla import failed — removing torchvision/torchaudio and retrying"
    $PY -m pip uninstall -q -y torchvision torchaudio
    $PY -c "import fla; print('fla OK after removal')" || log "fla unusable, torch fallback"
  }
  $PY - <<'PYEOF' || fail setup
import torch, transformers, peft
print("torch", torch.__version__, "| transformers", transformers.__version__,
      "| peft", peft.__version__, "| gpu", torch.cuda.get_device_name(0))
PYEOF
  touch OK_SETUP
fi

if [ ! -f OK_SMOKE ]; then
  log "smoke: mat + std, --limit 4"
  $PY gen_expls.py --model mat --limit 4 --out results/smoke_mat.json > smoke_mat.log 2>&1
  grep -q GEN_DONE_MAT smoke_mat.log && [ -s results/smoke_mat.json ] || fail smoke
  $PY gen_expls.py --model std --limit 4 --out results/smoke_std.json > smoke_std.log 2>&1
  grep -q GEN_DONE_STD smoke_std.log && [ -s results/smoke_std.json ] || fail smoke
  $PY - <<'PYEOF' || fail smoke
import json, re
# stray CJK chars happen at T=1; injection FAILURE = wholesale Chinese
# free-association, so gate on the CJK character FRACTION per explanation
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")
for arm in ("mat", "std"):
    p = json.load(open(f"results/smoke_{arm}.json"))
    for e in p["entries"]:
        txt = "\n".join(e["explanations"][0])
        frac = len(CJK.findall(txt)) / max(1, len(txt))
        assert frac < 0.10, f"{arm} ci={e['ci']}: {frac:.0%} CJK — injection failed"
        assert txt.strip(), f"{arm} ci={e['ci']}: empty explanation"
    print(arm, "smoke sample:", " / ".join(p["entries"][0]["explanations"][0])[:200])
PYEOF
  touch OK_SMOKE
fi

# same CJK-fraction gate as the smoke, applied to a full-run output
cjk_gate() {
  $PY - "$1" <<'PYEOF'
import json, re, sys
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")
p = json.load(open(sys.argv[1]))
bad = 0
for e in p["entries"]:
    txt = "\n".join(e["explanations"][0])
    if len(CJK.findall(txt)) / max(1, len(txt)) >= 0.10 or not txt.strip():
        bad += 1
assert bad == 0, f"{bad} heavy-CJK/empty explanations — injection suspect"
print("cjk gate OK:", p["meta"]["cjk_explanations"], "stray-CJK expls (light)")
PYEOF
}

if [ ! -f OK_MAT ]; then
  log "full: mat (500 ctx)"
  $PY gen_expls.py --model mat --out results/explanations_mat.json > mat.log 2>&1
  grep -q GEN_DONE_MAT mat.log && [ -s results/explanations_mat.json ] || fail mat
  cjk_gate results/explanations_mat.json || fail mat
  touch OK_MAT
fi

if [ ! -f OK_STD ]; then
  log "full: std (500 ctx)"
  $PY gen_expls.py --model std --out results/explanations_std.json > std.log 2>&1
  grep -q GEN_DONE_STD std.log && [ -s results/explanations_std.json ] || fail std
  cjk_gate results/explanations_std.json || fail std
  touch OK_STD
fi

touch DONE_ALL
log "all done"
