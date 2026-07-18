#!/bin/bash
# Multi-token NLA steering pass. Sentinel: POETRY3_DONE / PHASE3_FAILED.
set -uo pipefail
exec > /root/poetry3.log 2>&1
cd /root/evalsuite
PY=/root/venv/bin/python
export HF_HOME=/workspace/hf PYTHONPATH=/root/evalsuite
echo "[driver3] start $(date -u)"
$PY poetry_steer.py mencode mat || { echo PHASE3_FAILED; exit 1; }
$PY poetry_steer.py mencode std || { echo PHASE3_FAILED; exit 1; }
$PY poetry_steer.py msteer      || { echo PHASE3_FAILED; exit 1; }
echo POETRY3_DONE
