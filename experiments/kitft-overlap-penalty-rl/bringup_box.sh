#!/bin/bash
# One-shot box bring-up for the copied-bits RL run (lmsysorg sglang image).
# Prereqs: repo at /workspace/nla, tokens at /root/.hf_token + /root/.wandb_key,
# flash-attn wheel at /workspace/flash_attn-*.whl. RL data is NOT built here —
# it's prebuilt/validated on the dev box; the launcher downloads it from HF.
# Run:  nohup bash .../bringup_box.sh > /workspace/out/bringup.log 2>&1 &
set -euo pipefail
mkdir -p /workspace/out /workspace/models /workspace/out/dumps

echo "=== [A] env setup (miles + patches + deps + flash-attn wheel) ==="
bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh

echo "=== [B] flash-attn runtime check ==="
python - << 'PY'
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
assert torch.isfinite(flash_attn_func(q, q, q, causal=True)).all()
print("FLASH_ATTN_RUNTIME_OK on", torch.cuda.get_device_name(0))
PY

echo "=== [C] models (instruct + kitft av/ar) + prebuilt RL data ==="
python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
for repo, dst in [
    ("Qwen/Qwen2.5-7B-Instruct", "/workspace/models/qwen2.5-7b-instruct"),
    ("kitft/nla-qwen2.5-7b-L20-av", "/workspace/models/kitft_av"),
    ("kitft/nla-qwen2.5-7b-L20-ar", "/workspace/models/kitft_ar"),
]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, local_dir=dst, token=tok, max_workers=16)
for f in ("data/rl_tagged_ctx.parquet", "data/rl_tagged_ctx.parquet.nla_meta.yaml",
          "data/unigrams.json"):
    hf_hub_download("syvb/nla-qwen2.5-7b-L20-rl-overlappen", f, token=tok,
                    local_dir="/workspace/_data")
import shutil
shutil.copy("/workspace/_data/data/rl_tagged_ctx.parquet", "/workspace/out/")
shutil.copy("/workspace/_data/data/rl_tagged_ctx.parquet.nla_meta.yaml", "/workspace/out/")
shutil.copy("/workspace/_data/data/unigrams.json", "/workspace/out/")
print("DOWNLOADS_DONE", flush=True)
PY

echo "=== [D] end-to-end plumbing smoke: data source metadata + reward scorer ==="
python - << 'PY'
import os, sys
os.environ["NLA_OVERLAP_PENALTY"] = "0.01"
os.environ["NLA_OVERLAP_UNIGRAMS"] = "/workspace/out/unigrams.json"
sys.path.insert(0, "/workspace/nla")
# NLADataSource must forward the context into metadata (the reward asserts on it)
import pyarrow.parquet as pq
t = pq.read_table("/workspace/out/rl_tagged_ctx.parquet",
                  columns=["prompt", "detokenized_text_truncated"])
ctx = t.column("detokenized_text_truncated")[0].as_py()
assert ctx and len(ctx) > 50, "empty context in parquet"
from nla.reward import _copied_bits, _overlap_penalty
b = _copied_bits(ctx[-200:], ctx)       # self-copy: must be large
z = _copied_bits("completely unrelated paraphrase about gardening", ctx)
p = _overlap_penalty(ctx[-200:], ctx)
assert b > 50 and z < 5 and p < -0.3, (b, z, p)
print(f"REWARD_PLUMBING_OK self-copy={b:.0f}b unrelated={z:.1f}b penalty={p:.2f}")
PY

echo "BRINGUP_DONE"
