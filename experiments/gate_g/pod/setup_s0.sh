#!/bin/bash
# Gate G Stage 0 pod setup: deps + models + h_20/h_22 activations.
set -x
mkdir -p /work/gate_g /work/gate_c && cd /work/gate_g
pip install -q bitsandbytes wandb peft accelerate
hf download syvb/nla-layer-diff-experiments --repo-type dataset \
  --include "gate_c/acts20.npy" "gate_c/acts22.npy" "gate_c/meta2.json" \
  --local-dir /work/dl
ln -sf /work/dl/gate_c/acts20.npy /work/dl/gate_c/acts22.npy \
       /work/dl/gate_c/meta2.json /work/gate_c/
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir /work/gate_g/models/av
hf download kitft/nla-qwen2.5-7b-L20-ar --local-dir /work/gate_g/models/ar
echo SETUP_DONE
