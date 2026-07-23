#!/bin/bash
# Samples + dense-sweep box pass (1×H100, pytorch image). Ops fixes from the
# eval-box run baked in: git install, httpx/orjson in deps, conda PATH.
# Run:  nohup bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/samples_box_run.sh \
#         > /workspace/out/samplesbox.log 2>&1 &
set -euo pipefail
export PATH=/opt/conda/bin:$PATH
mkdir -p /workspace/out /workspace/m /workspace/data
export HF_TOKEN="$(cat /root/.hf_token)"

echo "=== [A] deps ==="
pip install -q "transformers==4.57.1" "huggingface_hub>=0.34,<1.0" pyarrow pyyaml safetensors accelerate numpy httpx orjson

echo "=== [B] models (ws + sfx50 pairs) + eval parquet ==="
python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
for repo, pat, dst in [
    ("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3", None, "/workspace/m/ws_actor"),
    ("syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3", None, "/workspace/m/ws_critic"),
    ("syvb/nla-qwen2.5-7b-L20-suffix-rl", "iter_0000050/*", "/workspace/m/sfx50"),
]:
    print("downloading", repo, flush=True)
    snapshot_download(repo, local_dir=dst, token=tok, max_workers=16,
                      allow_patterns=[pat] if pat else None)
for f in ("v3/av_eval_v3.parquet", "v3/av_eval_v3.parquet.nla_meta.yaml"):
    hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46", f,
                    repo_type="dataset", token=tok, local_dir="/workspace/data")
print("DOWNLOADS_DONE", flush=True)
PY

echo "=== [C] generate + sweep ==="
cd /root
PYTHONPATH=/workspace/nla python /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/gen_samples_and_sweep.py
echo "SAMPLES_BOX_DONE"
