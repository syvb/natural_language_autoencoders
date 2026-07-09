#!/bin/bash
# Run the FVE-vs-truncation sweep for BOTH the RLed truncation model (hybrid head)
# and the published kitft control, on a box already prepared by setup_box.sh.
# Outputs CSVs to /workspace/sweep (RLed) and /workspace/sweep_kitft (control).
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N="${1:-100}"
cd /workspace
echo "=== RLed truncation (RLed backbone + clean SFT head) ==="
AV_DIR=/workspace/hf_out/av AR_DIR=/workspace/hf_out/ar OUTDIR=/workspace/sweep \
  TAG="kl0.01/iter_0000200" PYTHONPATH=/workspace/nla python "$HERE/sweep_fve.py" "$N"
echo "=== control: published kitft ==="
AV_DIR=/workspace/kitft/av AR_DIR=/workspace/kitft/ar OUTDIR=/workspace/sweep_kitft \
  TAG="kitft-published (control)" PYTHONPATH=/workspace/nla python "$HERE/sweep_fve.py" "$N"
echo "=== warm-start (pre-RL) ==="
AV_DIR=/workspace/ws/av AR_DIR=/workspace/ws/ar OUTDIR=/workspace/sweep_ws \
  TAG="warm-start (pre-RL)" PYTHONPATH=/workspace/nla python "$HERE/sweep_fve.py" "$N"
echo "ALL_SWEEPS_DONE  (csvs in /workspace/sweep, sweep_kitft, sweep_ws)"
