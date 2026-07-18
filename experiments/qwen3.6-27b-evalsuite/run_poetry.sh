#!/bin/bash
# Poetry-steering driver (two couplet configs handled inside each phase).
# Sentinels: PHASE_*_DONE, POETRY_DONE, PHASE_FAILED.
set -uo pipefail
exec > /root/poetry.log 2>&1
cd /root/evalsuite
PY=/root/venv/bin/python
export HF_HOME=/workspace/hf PYTHONPATH=/root/evalsuite
echo "[driver] start $(date -u)"
$PY poetry_steer.py baseline   || { echo PHASE_FAILED; exit 1; }
echo PHASE_BASELINE_DONE
$PY poetry_steer.py av mat     || { echo PHASE_FAILED; exit 1; }
$PY poetry_steer.py encode mat || { echo PHASE_FAILED; exit 1; }
echo PHASE_MAT_DONE
$PY poetry_steer.py av std     || { echo PHASE_FAILED; exit 1; }
$PY poetry_steer.py encode std || { echo PHASE_FAILED; exit 1; }
echo PHASE_STD_DONE
$PY poetry_steer.py steer      || { echo PHASE_FAILED; exit 1; }
echo POETRY_DONE
