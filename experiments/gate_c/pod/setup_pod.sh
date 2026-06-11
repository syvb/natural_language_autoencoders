#!/bin/bash
# Gate C session-2 pod setup. Run as: bash setup_pod.sh [--kl]
# --kl additionally predownloads Qwen2.5-7B-Instruct for the KL judge.
set -x
mkdir -p /work/gate_c && cd /work/gate_c
pip install -q bitsandbytes wandb
# hub 1.x: `huggingface-cli` is a silent no-op, use `hf`
hf download syvb/nla-layer-diff-experiments --repo-type dataset \
  --include "gate_c/arms/*" "gate_c/diff_targets.npy" "gate_c/meta2.json" \
  "gate_c/doc_tokens.jsonl" "gate_c/acts18.npy" --local-dir /work/dl
ln -sf /work/dl/gate_c/arms/* /work/gate_c/
ln -sf /work/dl/gate_c/diff_targets.npy /work/dl/gate_c/acts18.npy \
       /work/dl/gate_c/meta2.json /work/dl/gate_c/doc_tokens.jsonl /work/gate_c/
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir /work/gate_c/models/ar
if [ "$1" = "--kl" ]; then
  hf download Qwen/Qwen2.5-7B-Instruct
fi
echo SETUP_DONE
