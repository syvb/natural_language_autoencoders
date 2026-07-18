#!/bin/bash
# Supplemental pass: +24 AV samples at the steer token for BOTH models (equal
# selection pools), re-encode with plan-preferring selection, rerun all steer
# arms. Sentinel: POETRY2_DONE / PHASE2_FAILED.
set -uo pipefail
exec > /root/poetry2.log 2>&1
cd /root/evalsuite
PY=/root/venv/bin/python
export HF_HOME=/workspace/hf PYTHONPATH=/root/evalsuite
echo "[driver2] start $(date -u)"
$PY poetry_steer.py av mat --extra 24  || { echo PHASE2_FAILED; exit 1; }
$PY poetry_steer.py encode mat         || { echo PHASE2_FAILED; exit 1; }
$PY poetry_steer.py av std --extra 24  || { echo PHASE2_FAILED; exit 1; }
$PY poetry_steer.py encode std         || { echo PHASE2_FAILED; exit 1; }
$PY poetry_steer.py steer              || { echo PHASE2_FAILED; exit 1; }
echo POETRY2_DONE
