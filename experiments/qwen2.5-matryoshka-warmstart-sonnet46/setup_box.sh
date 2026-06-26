#!/bin/bash
# Turnkey environment setup on a fresh pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel box.
# Applies the fixes documented in ENV_FIXES.md. Run as root on the GPU box.
#   /root/.hf_token and /root/.wandb_key must already be present.
#   Assumes this experiment dir is rsynced to /workspace/nla/experiments/... and the
#   repo root is /workspace/nla.
set -euo pipefail
PIP=/opt/conda/bin/pip
PY=/opt/conda/bin/python
NLA_REPO=/workspace/nla
export HF_TOKEN=$(cat /root/.hf_token)

echo "=== 1. deps ==="
$PIP install -q "transformers==4.57.1" pyarrow safetensors pyyaml numpy datasets accelerate \
    wandb orjson "httpx[http2]" anthropic tqdm huggingface_hub hf_transfer
$PIP install -q flash-attn --no-build-isolation || echo "flash-attn install issue (FA2 used by actor)"
$PIP install -q "ray[default]" ring_flash_attn sglang-router omegaconf tensorboard blobfile pybase64 pylatexenc

echo "=== 2. miles ==="
cd /workspace && [ -d miles ] || git clone -q https://github.com/radixark/miles.git
cd /workspace/miles && git checkout -q "$(cat $NLA_REPO/nla/miles_patches/UPSTREAM_PIN | cut -d@ -f2)"
git apply $NLA_REPO/nla/miles_patches/*.patch || echo "patches may already be applied"
# torch-2.5.1 FSDP2 compat (ENV_FIXES §3)
sed -i 's|from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard|from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard|g' \
    /workspace/miles/miles/backends/fsdp_utils/actor.py
$PIP install -q -e /workspace/miles
$PIP install -q -e "$NLA_REPO"

echo "=== 3. patch nla/_sglang_sft_stubs.py (ENV_FIXES §2) ==="
$PY - "$NLA_REPO/nla/_sglang_sft_stubs.py" <<'PYEOF'
import sys, re
f = sys.argv[1]; s = open(f).read()
# (a) remove the sglang_router stub block — use the real installed sglang-router
try:
    a = s.index("# ─── sglang_router.launch_router.RouterArgs ───")
    b = s.index("_launch.RouterArgs = _RouterArgs") + len("_launch.RouterArgs = _RouterArgs")
    s = s.replace(s[a:b], "# (sglang_router NOT stubbed — real installed sglang-router provides RouterArgs)")
except ValueError:
    pass
# (b) _add_sglang_arguments must return parser AND register the args miles reads
old = "def _add_sglang_arguments(parser):  # noqa: ARG001\n    pass"
new = '''def _add_sglang_arguments(parser):  # noqa: ARG001
    g = parser.add_argument_group("sglang (stubbed for SFT)")
    g.add_argument("--sglang-router-ip", type=str, default=None)
    g.add_argument("--sglang-router-port", type=int, default=None)
    g.add_argument("--sglang-router-request-timeout-secs", type=int, default=3600)
    g.add_argument("--sglang-tp-size", type=int, default=1)
    g.add_argument("--sglang-dp-size", type=int, default=1)
    g.add_argument("--sglang-pp-size", type=int, default=1)
    g.add_argument("--sglang-ep-size", type=int, default=1)
    g.add_argument("--sglang-data-parallel-size", type=int, default=1)
    g.add_argument("--sglang-pipeline-parallel-size", type=int, default=1)
    g.add_argument("--sglang-expert-parallel-size", type=int, default=1)
    g.add_argument("--sglang-enable-dp-attention", action="store_true", default=False)
    g.add_argument("--sglang-enable-metrics", action="store_true", default=False)
    g.add_argument("--sglang-server-concurrency", type=int, default=512)
    g.add_argument("--sglang-dest-name", type=str, default=None)
    g.add_argument("--sglang-moe-a", type=str, default=None)
    g.add_argument("--sglang-moe-runner-backend", type=str, default=None)
    g.add_argument("--sglang-speculative-algorithm", type=str, default=None)
    return parser'''
if old in s: s = s.replace(old, new)
# (c) RouterArgs.add_cli_args returns parser (if the stub block still present in some fork)
s = s.replace("    def add_cli_args(parser, use_router_prefix=True, exclude_host_port=True):  # noqa: ARG004\n        pass",
              "    def add_cli_args(parser, use_router_prefix=True, exclude_host_port=True):  # noqa: ARG004\n        return parser")
# (d) FSDP-backend weight-sync stubs (no torch import!)
if "FSDP backend weight-sync imports" not in s:
    s += '''

# ─── FSDP backend weight-sync imports (unused in SFT, imported at module load) ───
def _pkg(name):
    m = _stub_module(name); m.__path__ = []; return m
for _p in ("sglang", "sglang.srt", "sglang.srt.utils", "sglang.srt.weight_sync", "sglang.srt.model_executor"):
    if _p not in sys.modules: _pkg(_p)
for _m in ("sglang.srt.patch_torch", "sglang.srt.utils.patch_torch"):
    _mod = _stub_module(_m); _mod.monkey_patch_torch_reductions = lambda *a, **k: None  # noqa: E731
sys.modules["sglang.srt.utils"].MultiprocessingSerializer = object
for _m in ("sglang.srt.weight_sync.tensor_bucket", "sglang.srt.model_executor.model_runner"):
    _mod = _stub_module(_m); _mod.FlattenedTensorBucket = object
_biv = _stub_module("sglang.srt.batch_invariant_ops")
_biv.enable_batch_invariant_mode = lambda *a, **k: None  # noqa: E731
'''
open(f, "w").write(s); print("patched stub")
PYEOF

echo "=== 4. .pth so the stub loads in every process incl Ray workers (ENV_FIXES §4) ==="
SP=$($PY -c 'import site;print(site.getsitepackages()[0])')
echo 'import nla._sglang_sft_stubs' > "$SP/zzz_nla_sglang_stub.pth"

echo "=== 5. launcher ==="
cp "$NLA_REPO/experiments/qwen2.5-matryoshka-warmstart-sonnet46/train_sft.py" /workspace/miles/train_sft.py

echo "=== 6. models + source dataset ==="
HF_HUB_ENABLE_HF_TRANSFER=1 $PY - <<'PYEOF'
import os
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/workspace/models/qwen2.5-7b-instruct",
                  token=tok, max_workers=16,
                  allow_patterns=["*.json","*.safetensors","*.txt","tokenizer*","merges*","vocab*","*.jinja"])
snapshot_download("kitft/nla-qwen2.5-7b-L20-av", local_dir="/workspace/models/rl-av", token=tok, max_workers=16)
snapshot_download("kitft/nla-qwen2.5-7b-L20-ar", local_dir="/workspace/models/rl-ar", token=tok, max_workers=16)
hf_hub_download("ceselder/nla-matryoshka-warmstart-sonnet46", "data/train-00000-of-00001.parquet",
                repo_type="dataset", local_dir="/workspace/data/matryoshka", token=tok)
print("downloads done")
PYEOF
echo "SETUP_DONE — verify: cd /workspace/miles && python train_sft.py --help 2>/dev/null | head"
