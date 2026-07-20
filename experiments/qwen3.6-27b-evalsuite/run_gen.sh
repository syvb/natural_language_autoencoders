#!/bin/bash
# Generalization driver. Sentinels: GEN_DONE / GEN_FAILED.
set -uo pipefail
exec > /root/gen.log 2>&1
cd /root/evalsuite
PY=/root/venv/bin/python
export HF_HOME=/workspace/hf PYTHONPATH=/root/evalsuite
echo "[gen-driver] start $(date -u)"
$PY poetry_gen.py screen      || { echo GEN_FAILED; exit 1; }
$PY poetry_gen.py av mat      || { echo GEN_FAILED; exit 1; }
$PY poetry_gen.py mencode mat || { echo GEN_FAILED; exit 1; }
echo GEN_MAT_DONE
$PY poetry_gen.py av std      || { echo GEN_FAILED; exit 1; }
$PY poetry_gen.py mencode std || { echo GEN_FAILED; exit 1; }
echo GEN_STD_DONE
$PY poetry_gen.py msteer      || { echo GEN_FAILED; exit 1; }
echo GEN_DONE
