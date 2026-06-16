#!/bin/bash
# Gate G gap-8 setup: deps + models (incl. base for extraction) + data.
set -x
mkdir -p /work/gate_g /work/gate_c && cd /work/gate_g
pip install -q bitsandbytes wandb peft accelerate
# data: meta2 + doc_tokens (for extraction), acts20 (reused as condition)
hf download syvb/nla-layer-diff-experiments --repo-type dataset gate_c/meta2.json --local-dir /work/dl
hf download syvb/nla-layer-diff-experiments --repo-type dataset gate_c/doc_tokens.jsonl --local-dir /work/dl
hf download syvb/nla-layer-diff-experiments --repo-type dataset gate_c/acts20.npy --local-dir /work/dl
ln -sf /work/dl/gate_c/meta2.json /work/dl/gate_c/doc_tokens.jsonl /work/dl/gate_c/acts20.npy /work/gate_c/
# reuse Stage-0 AV-of-h20 descriptions for the s20 arm (no re-gen)
hf download syvb/nla-layer-diff-experiments --repo-type dataset --include "gate_g/av_h20_out/*" --local-dir /work/dl2
mkdir -p /work/gate_g/av_h20_out && cp /work/dl2/gate_g/av_h20_out/*.jsonl /work/gate_g/av_h20_out/
# models: AV + AR (NLA) + base Qwen (for activation extraction)
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir /work/gate_g/models/av
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir /work/gate_g/models/ar
hf download Qwen/Qwen2.5-7B-Instruct >/dev/null 2>&1
echo SETUP_DONE
