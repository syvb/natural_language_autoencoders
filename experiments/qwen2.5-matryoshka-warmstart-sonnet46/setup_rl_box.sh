#!/bin/bash
# Turnkey setup for the TRUNCATED RL run on a fresh 8xH100 (stock
# pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image). Unlike the SFT warm-start
# (which stubbed SGLang via --debug-train-only), RL runs the REAL SGLang rollout
# stack, so this installs the full miles+SGLang env per miles' build_conda pins
# AND applies the fixes needed to run it on a stock image with current package
# versions. Run as root after rsyncing the repo to /workspace/nla.
#
# Validated 2026-06-26: with these fixes the RL loop reaches steady-state
# training (~32s/step on actor4/critic2/rollout2, step-0 reward/train MSE
# ratio 1.0000, truncation active). Each fix below maps to a concrete failure
# hit while bringing the run up — see the inline notes.
set -e
PIP=/opt/conda/bin/pip; PY=/opt/conda/bin/python; NLA=/workspace/nla

echo "=== [1] torch 2.9.1 + cu129 (miles build_conda pin; SGLang 0.5.7 needs it) ==="
$PIP install -q torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

echo "=== [2] SGLang @ miles' pinned commit + miles & NLA input_embeds patches ==="
cd /workspace && [ -d sglang ] || git clone -q https://github.com/sgl-project/sglang.git
cd /workspace/sglang && git checkout -q 24c91001cf99ba642be791e099d358f4dfe955f5
git apply /workspace/miles/docker/patch/v0.5.7/sglang.patch 2>/dev/null || echo "miles sglang.patch already applied"
bash $NLA/patches/apply_sglang_patches.sh /workspace/sglang
$PIP install -q -e "python[all]"   # NB: '[all]' extra warns "not provided" — harmless; sgl-kernel 0.3.20 still installed

echo "=== [3] miles deps + transformers pin ==="
$PIP install -q pyarrow safetensors pyyaml numpy datasets accelerate wandb orjson "httpx[http2]" anthropic tqdm huggingface_hub hf_transfer omegaconf tensorboard blobfile pybase64 pylatexenc ring_flash_attn sglang-router
$PIP install -q "transformers==4.57.1"

echo "=== [4] FIX: pin Ray to 2.47.1 (miles' Dockerfile pin) ==="
# Latest ray (2.55) changed accelerator/CUDA_VISIBLE_DEVICES semantics → colocated
# rollout engines fail to get GPUs. miles is built against 2.47.1.
$PIP install -q "ray[default]==2.47.1"

echo "=== [5] miles + nla editable ==="
cd /workspace/miles && git apply $NLA/nla/miles_patches/*.patch 2>/dev/null || echo "miles nla patches already applied"
$PIP install -q -e /workspace/miles
$PIP install -q -e $NLA

echo "=== [6] flash-attn (actor FA2) ==="
$PIP install -q flash-attn --no-build-isolation

echo "=== [7] FIX: libnuma (sgl-kernel fp8 ops need libnuma.so.1; stock image lacks it) ==="
apt-get install -y -q libnuma-dev && ldconfig

echo "=== [8] FIX: miles rollout-engine env on a stock (non-lmsysorg) image ==="
# Two edits to miles/ray/rollout.py's SGLang-engine runtime_env, both because
# miles assumes its own Docker image's CUDA layout:
#  (a) drop the NOSET_VISIBLE_DEVICES flags so each engine takes its OWN Ray-
#      allocated GPU bundle instead of inheriting RolloutManager(num_gpus=0)'s
#      empty CUDA_VISIBLE_DEVICES;
#  (b) drop the LD_LIBRARY_PATH override (it forces the image's CUDA-12.4 compat
#      libs, which break torch-2.9.1+cu129's CUDA init → "No accelerator").
$PY - <<'PYEOF'
import re
f="/workspace/miles/miles/ray/rollout.py"; s=open(f).read()
a='        env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {'
if a in s:
    s=s.replace(a,'        env_vars = {  # NLA: NOSET dropped so engine takes its own Ray GPU')
s2=re.sub(r'\n\s*"LD_LIBRARY_PATH": f"/usr/local/cuda/compat:[^\n]*\n', "\n", s)
open(f,"w").write(s2)
print("patched rollout.py (NOSET + LD_LIBRARY_PATH)")
PYEOF

echo "=== verify ==="
$PY -c "import torch,sglang,miles,nla,flash_attn; from sglang.srt.configs.model_config import ModelConfig; print('IMPORT_OK torch',torch.__version__,'sglang',sglang.__version__,'ray', __import__('ray').__version__)"
echo "SETUP_RL_BOX_DONE"
