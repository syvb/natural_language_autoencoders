#!/bin/bash
# Driver: setup (artifact-gated) then the reversed-order FVE sweep.
set -uo pipefail
bash /root/run/rev_setup.sh 2>&1 | tee /workspace/rev_setup.log
grep -q SETUP_DONE /workspace/rev_setup.log || { echo REV_FAIL_SETUP | tee /workspace/REV_FAIL; exit 1; }
export HF_TOKEN=$(cat /root/.hf_token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HOME=/workspace/hf
cd /root/run
/root/venv/bin/python -u clean_rev_fve.py 2>&1 | tee /workspace/rev_fve.log
grep -q REV_ALL_DONE /workspace/rev_fve.log || { echo REV_FAIL_SWEEP | tee /workspace/REV_FAIL; exit 1; }
echo REV_DONE | tee /workspace/REV_DONE
