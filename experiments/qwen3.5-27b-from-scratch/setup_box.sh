#!/bin/bash
# RL/SFT environment bring-up. Run INSIDE the training machine/container.
#
# Expected starting point: an `lmsysorg/sglang` image (torch + sglang + CUDA
# prebuilt, python >= 3.12) with this repo present at $NLA (rsync/git clone).
# Validated baseline: lmsysorg/sglang:v0.5.7-cu129-amd64 (7B smoke).
# For Qwen3.5 you need an image with sglang >= 0.5.9 (Qwen3.5 GDN support) —
# set SGLANG_IMAGE accordingly when creating the container and expect the
# regex-anchored patches below to need re-anchoring if sglang moved a lot
# (each helper asserts its anchor and names the file if it misses).
#
# Env knobs: NLA (repo path), TRANSFORMERS_PIN, RAY_PIN, FLASH_ATTN_WHEEL
# (prebuilt wheel path; falls back to a source build, ~15 min).
set -e
NLA="${NLA:-/workspace/nla}"
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-4.57.1}"
RAY_PIN="${RAY_PIN:-2.47.1}"

echo "=== [0] baseline ==="
python -c "import sys,torch;print('py',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)"
python -c "import sglang;print('sglang',sglang.__version__,sglang.__file__)"

echo "=== [1] miles @ pin + nla patches ==="
cd "$(dirname "$NLA")" && [ -d miles ] || git clone -q https://github.com/radixark/miles.git
cd miles && git checkout -q "$(cut -d@ -f2 "$NLA/nla/miles_patches/UPSTREAM_PIN")"
git apply "$NLA/nla/miles_patches/0001_miles_nla_integration.patch" 2>/dev/null && echo "0001 applied" || echo "0001 already?"
git apply "$NLA/nla/miles_patches/0002_train_py_nla_hooks.patch" 2>/dev/null && echo "0002 applied" || echo "0002 already?"
MILES_DIR="$(pwd)"

echo "=== [2] NLA sglang patches in place (image sglang is editable) ==="
SGLANG_SRC="${SGLANG_SRC:-/sgl-workspace/sglang}"
bash "$NLA/patches/apply_sglang_patches.sh" "$SGLANG_SRC"

echo "=== [3] rollout.py fix (drop NOSET + LD_LIBRARY_PATH; needed on lmsysorg images) ==="
python - "$MILES_DIR/miles/ray/rollout.py" << 'PYEOF'
import re, sys
f = sys.argv[1]; s = open(f).read()
anchor = '        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {'
if anchor in s:
    s = s.replace(anchor, '        env_vars = {  # NLA: NOSET dropped so engine takes its own Ray GPU'); print("NOSET dropped")
elif "NLA: NOSET dropped" in s:
    print("NOSET already dropped")
else:
    print("WARN NOSET anchor not found — check miles/ray/rollout.py by hand")
s2 = re.sub(r'\n\s*"LD_LIBRARY_PATH": f"/usr/local/cuda/compat:[^\n]*\n', "\n", s)
print("LD_LIBRARY_PATH removed" if s2 != s else "LD_LIBRARY_PATH (none/already)")
open(f, "w").write(s2)
PYEOF

echo "=== [4] deps (no torch; hf_hub<1.0 for transformers 4.x) ==="
pip install -q pyarrow safetensors pyyaml numpy datasets accelerate wandb orjson \
    "httpx[http2]" anthropic tqdm hf_transfer omegaconf tensorboard blobfile \
    pybase64 pylatexenc ring_flash_attn sglang-router
pip install -q "transformers==$TRANSFORMERS_PIN" "huggingface_hub>=0.34,<1.0"

echo "=== [5] ray $RAY_PIN ==="
pip install -q "ray[default]==$RAY_PIN"

echo "=== [6] miles + nla editable ==="
pip install -q -e "$MILES_DIR"
pip install -q -e "$NLA"

echo "=== [7] flash-attn ==="
if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "$FLASH_ATTN_WHEEL" ]; then
    pip install -q "$FLASH_ATTN_WHEEL"
else
    python -c "import flash_attn" 2>/dev/null || {
        echo "no prebuilt wheel (\$FLASH_ATTN_WHEEL) — building from source (~15 min)"
        pip install -q flash-attn --no-build-isolation
    }
fi

echo "=== verify ==="
python -c "import torch,sglang,miles,nla,flash_attn,transformers,ray; from sglang.srt.configs.model_config import ModelConfig; print('IMPORT_OK torch',torch.__version__,'sglang',sglang.__version__,'tf',transformers.__version__,'ray',ray.__version__,'fa',flash_attn.__version__)"
echo "SETUP_FULL_DONE"
