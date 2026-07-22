#!/bin/bash
# Suffix-RL PoC eval pass — single H100, transformers-only (no sglang/miles).
# Runs the four both-sides FVE evals that turn the training curve into the
# PoC verdict:
#   ws      v3 warm-start pair        — the baseline (never measured on suffixes)
#   it25    suffix iter_0000025 pair  — AV untouched (ACTOR_LR=0), critic
#                                       suffix-recalibrated: isolates how much
#                                       "improvement" is critic adaptation alone
#   it50    suffix iter_0000050 pair  — the PoC checkpoint (25 actor steps)
#   v3rl    v3 RL iter_0000200 pair   — gap-flip control: a front-loaded RLed
#                                       pair's prefix-vs-suffix asymmetry
# All arms: NLA_TRUNC_SIDE=both, T=1 sampling, NLA_GEN_MAX_NEW=120 (= the
# suffix run's training cap — see run_rl_suffix.sh header), N=100,
# lens 1,2,5,10,30,60,120.
# Prereqs: repo at /workspace/nla, /root/.hf_token. Run from /root (ENV_FIXES
# §6: keep cwd clean of a dir named "av" — torchvision's PyAV import).
# Run:  nohup bash /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/eval_box_run.sh \
#         > /workspace/out/evalbox.log 2>&1 &
set -euo pipefail
mkdir -p /workspace/out /workspace/m
export HF_TOKEN="$(cat /root/.hf_token)"

echo "=== [A] deps (transformers pin per ENV_FIXES; no sglang) ==="
pip install -q "transformers==4.57.1" "huggingface_hub>=0.34,<1.0" pyarrow pyyaml safetensors accelerate numpy

echo "=== [B] models + eval parquet ==="
python - << 'PY'
from huggingface_hub import snapshot_download, hf_hub_download
tok = open("/root/.hf_token").read().strip()
pairs = [
    ("syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3", None, "/workspace/m/ws_actor"),
    ("syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3", None, "/workspace/m/ws_critic"),
    ("syvb/nla-qwen2.5-7b-L20-suffix-rl", "iter_0000025/av/*", "/workspace/m/sfx25"),
    ("syvb/nla-qwen2.5-7b-L20-suffix-rl", "iter_0000025/ar/*", "/workspace/m/sfx25"),
    ("syvb/nla-qwen2.5-7b-L20-suffix-rl", "iter_0000050/av/*", "/workspace/m/sfx50"),
    ("syvb/nla-qwen2.5-7b-L20-suffix-rl", "iter_0000050/ar/*", "/workspace/m/sfx50"),
    ("syvb/nla-qwen2.5-7b-L20-v3-rl", "iter_0000200/av/*", "/workspace/m/v3rl"),
    ("syvb/nla-qwen2.5-7b-L20-v3-rl", "iter_0000200/ar/*", "/workspace/m/v3rl"),
]
for repo, pat, dst in pairs:
    print("downloading", repo, pat or "(all)", flush=True)
    snapshot_download(repo, local_dir=dst, token=tok, max_workers=16,
                      allow_patterns=[pat] if pat else None)
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
                "v3/av_eval_v3.parquet", repo_type="dataset", token=tok,
                local_dir="/workspace/data")
hf_hub_download("syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46",
                "v3/av_eval_v3.parquet.nla_meta.yaml", repo_type="dataset", token=tok,
                local_dir="/workspace/data")
print("DOWNLOADS_DONE", flush=True)
PY

echo "=== [C] four eval arms ==="
cd /root
EVALSCRIPT=/workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/eval_round_trip_fve.py
run_arm () {  # name av_dir ar_dir
    echo "=== ARM $1 ==="
    EVAL=/workspace/data/v3/av_eval_v3.parquet AV_DIR="$2" AR_DIR="$3" \
    NLA_TRUNC_SIDE=both NLA_GEN_TEMP=1 NLA_GEN_MAX_NEW=120 \
    PYTHONPATH=/workspace/nla python "$EVALSCRIPT" 100 1,2,5,10,30,60,120 \
        2>&1 | tee "/workspace/out/fve_$1.txt"
    echo "=== ARM $1 DONE ==="
}
run_arm ws   /workspace/m/ws_actor            /workspace/m/ws_critic
run_arm it25 /workspace/m/sfx25/iter_0000025/av /workspace/m/sfx25/iter_0000025/ar
run_arm it50 /workspace/m/sfx50/iter_0000050/av /workspace/m/sfx50/iter_0000050/ar
run_arm v3rl /workspace/m/v3rl/iter_0000200/av /workspace/m/v3rl/iter_0000200/ar
echo "EVAL_ALL_DONE"
