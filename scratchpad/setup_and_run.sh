#!/bin/bash
set -e
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(cat /root/.hf_token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "=== deps ==="
pip install -q -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors "huggingface_hub>=0.34,<1.0" hf_transfer accelerate 2>&1 | tail -3
echo "=== download model (kl0.01/iter_0000200/av) ==="
python - <<'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok=open("/root/.hf_token").read().strip()
snapshot_download("syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard",
    allow_patterns="kl0.01/iter_0000200/av/*", local_dir="/workspace/dl",
    token=tok, max_workers=16)
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
    "av_eval.parquet", repo_type="dataset", local_dir="/workspace/out",
    token=tok)
print("DOWNLOADS_DONE", flush=True)
PY
rm -rf /workspace/hf_out/av
mv /workspace/dl/kl0.01/iter_0000200/av /workspace/hf_out/av
echo "=== model contents ==="; ls /workspace/hf_out/av
echo "=== generate 100 samples ==="
cd /workspace
python /workspace/gen_md_samples.py 100 /workspace/av_kl0.01_iter200_100samples.md
echo "ALL_DONE"
