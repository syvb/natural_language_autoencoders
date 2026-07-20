#!/bin/bash
# Secrets-audit driver. Sentinels: SEC_GEN_DONE / SEC_MAT_DONE / SEC_DONE / SEC_FAILED.
set -uo pipefail
exec > /root/secrets.log 2>&1
cd /root/evalsuite
PY=/root/venv/bin/python
export HF_HOME=/workspace/hf PYTHONPATH=/root/evalsuite
echo "[sec-driver] start $(date -u)"
$PY secrets_av.py gen    || { echo SEC_FAILED; exit 1; }
echo SEC_GEN_DONE
$PY secrets_av.py av mat || { echo SEC_FAILED; exit 1; }
echo SEC_MAT_DONE
$PY secrets_av.py av std || { echo SEC_FAILED; exit 1; }
echo SEC_DONE
