#!/bin/bash
# Full stack install on box2 (4x H100), using the prebuilt flash-attn wheel (no compile).
# Models already present at /workspace/models; nla repo at /workspace/nla.
set -eo pipefail
echo "===== BOX2 INSTALL START $(date) ====="
source /opt/conda/etc/profile.d/conda.sh

if ! conda env list | grep -q '^miles '; then
  conda create -n miles python=3.12 pip -y -c conda-forge
fi
conda activate miles
export CUDA_HOME="$CONDA_PREFIX"

echo "=== [1] cuda 12.9 toolkit + nccl + cudnn ($(date +%H:%M)) ==="
conda install -n miles cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
conda install -n miles -c conda-forge cudnn -y

echo "=== [2] torch 2.9.1 cu129 ($(date +%H:%M)) ==="
pip install -q cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

echo "=== [3] sglang @24c9100 editable ($(date +%H:%M)) ==="
cd /workspace
[ -d sglang ] || git clone --quiet https://github.com/sgl-project/sglang.git
cd sglang && git checkout --quiet 24c91001cf99ba642be791e099d358f4dfe955f5
pip install -e "python[all]"

echo "=== [4] flash-attn 2.8.3 prebuilt wheel ($(date +%H:%M)) ==="
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "=== [5] torch_memory_saver ($(date +%H:%M)) ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall || echo "WARN tms"

echo "=== [6] miles @051cd15 patches + editable ($(date +%H:%M)) ==="
cd /workspace
[ -d miles ] || git clone --quiet https://github.com/radixark/miles.git
cd miles && git checkout --quiet 051cd15
git apply /workspace/nla/nla/miles_patches/0001_miles_nla_integration.patch && echo OK int || echo "SKIP int"
git apply /workspace/nla/nla/miles_patches/0002_train_py_nla_hooks.patch && echo OK hooks || echo "SKIP hooks"
pip install -e .

echo "=== [7] sglang patches ($(date +%H:%M)) ==="
cd /workspace/sglang
git apply /workspace/miles/docker/patch/v0.5.7/sglang.patch && echo OK miles_sglang || echo "SKIP miles_sglang"
bash /workspace/nla/patches/apply_sglang_patches.sh /workspace/sglang

echo "=== [8] nla editable + cudnn pin ($(date +%H:%M)) ==="
cd /workspace/nla && pip install -e .
pip install -q nvidia-cudnn-cu12==9.16.0.29

echo "===== BOX2 INSTALL DONE $(date) ====="
python -c "import torch,sglang,flash_attn,nla; print('IMPORT OK torch',torch.__version__,'cuda_avail',torch.cuda.is_available(),'fa',flash_attn.__version__)"
echo "===== VERIFY DONE ====="
