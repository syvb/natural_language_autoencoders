#!/bin/bash
# Lean FSDP-only NLA stack install (skips Megatron/TE/apex — not needed for 7B FSDP).
# Follows miles/build_conda.sh pins: sglang 24c9100 (v0.5.7), torch 2.9.1+cu129, flash-attn 2.7.4.
set -eo pipefail
echo "===== INSTALL START $(date) ====="
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q '^miles '; then
  echo "=== [1/9] create env miles (py3.12) ==="
  conda create -n miles python=3.12 pip -y -c conda-forge
fi
conda activate miles
export CUDA_HOME="$CONDA_PREFIX"
echo "env: $CONDA_PREFIX  python: $(python --version)"

echo "=== [2/9] cuda 12.9 toolkit + nccl + cudnn ($(date +%H:%M)) ==="
conda install -n miles cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
conda install -n miles -c conda-forge cudnn -y

echo "=== [3/9] torch 2.9.1 cu129 ($(date +%H:%M)) ==="
pip install -q cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

echo "=== [4/9] sglang @24c9100 editable ($(date +%H:%M)) ==="
cd /workspace
[ -d sglang ] || git clone --quiet https://github.com/sgl-project/sglang.git
cd sglang && git checkout --quiet 24c91001cf99ba642be791e099d358f4dfe955f5
pip install -e "python[all]"

echo "=== [5/9] flash-attn 2.7.4 ($(date +%H:%M)) ==="
pip install -q cmake ninja
MAX_JOBS=32 pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "=== [6/9] torch_memory_saver ($(date +%H:%M)) ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall || echo "WARN tms"

echo "=== [7/9] miles patches + editable install ($(date +%H:%M)) ==="
cd /workspace/miles
git apply /workspace/nla/nla/miles_patches/0001_miles_nla_integration.patch && echo "OK miles_integration_patch" || echo "FAIL miles_integration_patch"
git apply /workspace/nla/nla/miles_patches/0002_train_py_nla_hooks.patch && echo "OK miles_hooks_patch" || echo "FAIL miles_hooks_patch"
pip install -e .
pip install -q nvidia-cudnn-cu12==9.16.0.29

echo "=== [8/9] sglang patches (miles v0.5.7 + NLA) ($(date +%H:%M)) ==="
cd /workspace/sglang
git apply /workspace/miles/docker/patch/v0.5.7/sglang.patch && echo "OK miles_sglang_patch" || echo "WARN miles_sglang_patch (may already be in source)"
bash /workspace/nla/patches/apply_sglang_patches.sh /workspace/sglang

echo "=== [9/9] nla editable ($(date +%H:%M)) ==="
cd /workspace/nla && pip install -e .

echo "===== INSTALL DONE $(date) ====="
python -c "import torch, sglang; print('IMPORT torch', torch.__version__, 'cuda_avail', torch.cuda.is_available())"
python -c "import nla; print('IMPORT nla OK')"
echo "===== VERIFY DONE ====="
