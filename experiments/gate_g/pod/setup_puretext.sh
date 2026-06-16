#!/bin/bash
# Gate G pure-text pre-test setup: AR + base Qwen (for extraction) + AV texts.
set -x
mkdir -p /work/gate_g /work/gate_c && cd /work/gate_g
pip install -q bitsandbytes wandb peft accelerate
hf download syvb/nla-layer-diff-experiments --repo-type dataset gate_c/meta2.json --local-dir /work/dl
hf download syvb/nla-layer-diff-experiments --repo-type dataset gate_c/doc_tokens.jsonl --local-dir /work/dl
ln -sf /work/dl/gate_c/meta2.json /work/dl/gate_c/doc_tokens.jsonl /work/gate_c/
hf download syvb/nla-layer-diff-experiments --repo-type dataset --include "gate_g/av_h16_out/*" --local-dir /work/dl2
hf download syvb/nla-layer-diff-experiments --repo-type dataset --include "gate_g/av_h24_out/*" --local-dir /work/dl2
mkdir -p av_h16_out av_h24_out
cp /work/dl2/gate_g/av_h16_out/*.jsonl av_h16_out/
cp /work/dl2/gate_g/av_h24_out/*.jsonl av_h24_out/
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir /work/gate_g/models/ar
hf download Qwen/Qwen2.5-7B-Instruct >/dev/null 2>&1
echo SETUP_DONE
