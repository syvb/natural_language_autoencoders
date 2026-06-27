#!/bin/bash
# Known-good RL env bring-up for lmsysorg/sglang:v0.5.7-cu129-amd64 (validated
# end-to-end on the 20-step smoke test). torch/sglang/py312/cuda all prebuilt.
set -e
NLA=/workspace/nla
echo "=== [0] baseline ==="
python -c "import sys,torch;print('py',sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda)"
python -c "import sglang;print('sglang',sglang.__version__,sglang.__file__)"

echo "=== [1] miles @ pin + nla patches ==="
cd /workspace && [ -d miles ] || git clone -q https://github.com/radixark/miles.git
cd /workspace/miles && git checkout -q "$(cut -d@ -f2 $NLA/nla/miles_patches/UPSTREAM_PIN)"
git apply $NLA/nla/miles_patches/0001_miles_nla_integration.patch 2>/dev/null && echo "0001 applied" || echo "0001 already?"
git apply $NLA/nla/miles_patches/0002_train_py_nla_hooks.patch 2>/dev/null && echo "0002 applied" || echo "0002 already?"

echo "=== [2] NLA sglang patches in place (image sglang is editable at /sgl-workspace/sglang) ==="
bash $NLA/patches/apply_sglang_patches.sh /sgl-workspace/sglang

echo "=== [3] rollout.py step-8 fix (drop NOSET + LD_LIBRARY_PATH; needed on lmsysorg too) ==="
python - << 'PYEOF'
import re
f="/workspace/miles/miles/ray/rollout.py"; s=open(f).read()
anchor='        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {'
if anchor in s:
    s=s.replace(anchor,'        env_vars = {  # NLA: NOSET dropped so engine takes its own Ray GPU'); print("NOSET dropped")
elif "NLA: NOSET dropped" in s:
    print("NOSET already dropped")
else:
    print("WARN NOSET anchor not found")
s2=re.sub(r'\n\s*"LD_LIBRARY_PATH": f"/usr/local/cuda/compat:[^\n]*\n', "\n", s)
print("LD_LIBRARY_PATH removed" if s2!=s else "LD_LIBRARY_PATH (none/already)")
open(f,"w").write(s2)
PYEOF

echo "=== [4] deps (no torch; hf_hub<1.0 for transformers) ==="
pip install -q pyarrow safetensors pyyaml numpy datasets accelerate wandb orjson "httpx[http2]" anthropic tqdm hf_transfer omegaconf tensorboard blobfile pybase64 pylatexenc ring_flash_attn sglang-router
pip install -q "transformers==4.57.1" "huggingface_hub>=0.34,<1.0"

echo "=== [5] ray 2.47.1 ==="
pip install -q "ray[default]==2.47.1"

echo "=== [6] miles + nla editable ==="
pip install -q -e /workspace/miles
pip install -q -e $NLA

echo "=== [7] prebuilt flash-attn wheel ==="
pip install -q /workspace/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl

echo "=== verify ==="
python -c "import torch,sglang,miles,nla,flash_attn,transformers,ray; from sglang.srt.configs.model_config import ModelConfig; print('IMPORT_OK torch',torch.__version__,'sglang',sglang.__version__,'tf',transformers.__version__,'ray',ray.__version__,'fa',flash_attn.__version__)"
echo "SETUP_FULL_DONE"
