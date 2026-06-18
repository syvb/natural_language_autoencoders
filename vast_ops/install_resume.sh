#!/bin/bash
# Resume install after the flash-attn OOM: install prebuilt FA2 wheel (torch2.9/cu12/cp312),
# then run the remaining stack steps. Idempotent-ish; safe to re-run.
set -eo pipefail
echo "===== RESUME START $(date) ====="
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
export CUDA_HOME="$CONDA_PREFIX"

echo "=== [5b] flash-attn 2.8.3 prebuilt wheel (no compile) ==="
WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
pip install "$WHL"
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"

echo "=== [6] torch_memory_saver ==="
pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall || echo "WARN tms"

echo "=== [7] miles patches + editable ==="
cd /workspace/miles
git apply --check /workspace/nla/nla/miles_patches/0001_miles_nla_integration.patch 2>/dev/null && \
  git apply /workspace/nla/nla/miles_patches/0001_miles_nla_integration.patch && echo "OK miles_integration_patch" || echo "SKIP/already miles_integration_patch"
git apply --check /workspace/nla/nla/miles_patches/0002_train_py_nla_hooks.patch 2>/dev/null && \
  git apply /workspace/nla/nla/miles_patches/0002_train_py_nla_hooks.patch && echo "OK miles_hooks_patch" || echo "SKIP/already miles_hooks_patch"
pip install -e .
pip install -q nvidia-cudnn-cu12==9.16.0.29 || echo "WARN cudnn pin"

echo "=== [8] sglang patches (miles v0.5.7 + NLA) ==="
cd /workspace/sglang
git apply --check /workspace/miles/docker/patch/v0.5.7/sglang.patch 2>/dev/null && \
  git apply /workspace/miles/docker/patch/v0.5.7/sglang.patch && echo "OK miles_sglang_patch" || echo "SKIP/already miles_sglang_patch"
bash /workspace/nla/patches/apply_sglang_patches.sh /workspace/sglang

echo "=== [9] nla editable ==="
cd /workspace/nla && pip install -e .

echo "===== RESUME DONE $(date) ====="
python -c "import torch, sglang, flash_attn; print('IMPORT torch', torch.__version__, 'sglang', sglang.__version__, 'fa', flash_attn.__version__)"
python -c "import nla; print('IMPORT nla OK')"
echo "===== VERIFY DONE ====="
