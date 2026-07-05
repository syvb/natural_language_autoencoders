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
# "keep" = trust the image's transformers/hf_hub pair (0.5.7 ships 4.57.1;
# Qwen3.5-era images ship 5.x). Set an explicit version only to override.
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-keep}"
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
python - "$SGLANG_SRC" <<'PYEOF'
import sys, sglang, os
src, mod = os.path.realpath(sys.argv[1]), os.path.realpath(sglang.__file__)
assert mod.startswith(src), (
    f"imported sglang ({mod}) is NOT under SGLANG_SRC ({src}) — patches would "
    f"apply to a dead tree; point SGLANG_SRC at the imported checkout"
)
print("sglang import path matches SGLANG_SRC")
PYEOF
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
    raise SystemExit("NOSET anchor not found in miles/ray/rollout.py — miles moved; "
                     "port the fix by hand (engines otherwise die with no-accelerator errors)")
s2 = re.sub(r'\n\s*"LD_LIBRARY_PATH": f"/usr/local/cuda/compat:[^\n]*\n', "\n", s)
print("LD_LIBRARY_PATH removed" if s2 != s else "LD_LIBRARY_PATH (none/already)")
open(f, "w").write(s2)
PYEOF

echo "=== [3c] miles apply_fsdp2 vs transformers>=5 (_no_split_modules is a set) ==="
python - "$MILES_DIR/miles/backends/fsdp_utils/actor.py" << 'PYEOF'
import sys
f = sys.argv[1]; s = open(f).read()
old = "    layer_cls_to_wrap = model._no_split_modules\n"
new = "    layer_cls_to_wrap = list(model._no_split_modules)  # NLA: tf>=5 makes this a set\n"
if new in s:
    print("apply_fsdp2 already patched")
elif old in s:
    open(f, "w").write(s.replace(old, new, 1)); print("apply_fsdp2 patched (set->list)")
else:
    print("WARN apply_fsdp2 anchor not found — check by hand")
PYEOF

echo "=== [4] deps (no torch) ==="
pip install -q pyarrow safetensors pyyaml numpy datasets accelerate wandb orjson \
    "httpx[http2]" anthropic tqdm hf_transfer omegaconf tensorboard blobfile \
    pybase64 pylatexenc ring_flash_attn sglang-router
if [ "$TRANSFORMERS_PIN" = "keep" ]; then
    # New-stack mode: the image ships a coherent transformers/hf_hub pair
    # (needed for Qwen3.5-era models) — do NOT downgrade it.
    python -c "import transformers,huggingface_hub;print('keeping image transformers',transformers.__version__,'hf_hub',huggingface_hub.__version__)"
else
    pip install -q "transformers==$TRANSFORMERS_PIN" "huggingface_hub>=0.34,<1.0"
fi

echo "=== [5] ray $RAY_PIN ==="
pip install -q "ray[default]==$RAY_PIN"

echo "=== [6] miles + nla editable (--no-deps: deps are step [4]'s job; with
# deps pip tries to upgrade distro-managed packages, e.g. PyJWT on the
# v0.5.13 image, and dies on the missing RECORD file) ==="
pip install -q -e "$MILES_DIR" --no-deps
pip install -q -e "$NLA" --no-deps

echo "=== [7] flash-attn ==="
if [ -n "${FLASH_ATTN_WHEEL:-}" ] && [ -f "$FLASH_ATTN_WHEEL" ]; then
    pip install -q "$FLASH_ATTN_WHEEL"
fi
# A cached wheel can be ABI-incompatible with a newer image torch — verify the
# import, not the install; fall back to a source build. Cap MAX_JOBS: the
# default (~nproc ninja jobs) ate ~700 GB RAM and starved sshd on a big host.
python -c "import flash_attn" 2>/dev/null || {
    echo "flash_attn missing or ABI-broken — building from source (~20 min)"
    pip uninstall -q -y flash-attn 2>/dev/null || true
    # FORCE_BUILD skips setup.py's prebuilt-wheel probe — an unguarded
    # urlopen that wedged indefinitely on a torch version with no upstream
    # wheel (torch 2.11: none exist as of 2026-07).
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS="${FLASH_ATTN_MAX_JOBS:-16}" \
        pip install -q flash-attn --no-build-isolation
}

echo "=== [7b] ring_flash_attn vs transformers>=5 ==="
# transformers 5.x removed is_flash_attn_greater_or_equal_2_10 (renamed to
# flash_attn_supports_top_left_mask with inverted sense); ring_flash_attn
# (<=0.1.8 AND git main, 2026-07) still imports it — and its own try/except
# fallback repeats the SAME import. The predicate means "flash-attn >= 2.1.0",
# unconditionally true for any env this script builds. Shim it in place.
python - <<'PYEOF'
import importlib.util, re, sys
try:
    import ring_flash_attn  # noqa: F401
    print("ring_flash_attn imports fine — no shim needed")
except ImportError:
    spec = importlib.util.find_spec("ring_flash_attn")
    f = spec.submodule_search_locations[0] + "/adapters/hf_adapter.py"
    s = open(f).read()
    pat = re.compile(
        r"try:\n"
        r"    from transformers\.modeling_flash_attention_utils import \(\n"
        r"        is_flash_attn_greater_or_equal_2_10,\n"
        r"    \)\n"
        r"except ImportError:\n"
        r"    # transformers <= 4\.53\.x\n"
        r"    from transformers\.modeling_flash_attention_utils import \(\n"
        r"        is_flash_attn_greater_or_equal_2_10,\n"
        r"    \)\n"
    )
    sub = (
        "try:\n"
        "    from transformers.modeling_flash_attention_utils import (\n"
        "        is_flash_attn_greater_or_equal_2_10,\n"
        "    )\n"
        "except ImportError:\n"
        "    # NLA shim: transformers >=5 removed it; means flash-attn >= 2.1.0,\n"
        "    # unconditionally true in this environment.\n"
        "    def is_flash_attn_greater_or_equal_2_10():\n"
        "        return True\n"
    )
    s2, n = pat.subn(sub, s, count=1)
    assert n == 1, f"ring_flash_attn hf_adapter.py import block not found in {f} — patch by hand"
    open(f, "w").write(s2)
    print(f"shimmed {f}")
    import ring_flash_attn  # noqa: F401  (verify)
    print("ring_flash_attn imports after shim")
PYEOF


echo "=== verify ==="
python -c "import torch,sglang,miles,nla,flash_attn,transformers,ray; from sglang.srt.configs.model_config import ModelConfig; print('IMPORT_OK torch',torch.__version__,'sglang',sglang.__version__,'tf',transformers.__version__,'ray',ray.__version__,'fa',flash_attn.__version__)"
echo "SETUP_FULL_DONE"
