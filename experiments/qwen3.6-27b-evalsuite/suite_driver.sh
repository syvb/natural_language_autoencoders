#!/bin/bash
# Runs the full 27B eval suite for $MODEL (std|mat) on this box.
# Artifact-gated: re-running resumes at the first missing output.
set -uo pipefail
cd /workspace/EasyNLA/scripts
export PYTHONPATH=/workspace/EasyNLA:/workspace/EasyNLA/scripts
export WORK=/workspace
: "${MODEL:?set MODEL=std|mat}"
S=/workspace/suite
mkdir -p $S

fail() { echo "SUITE_FAIL: $1" | tee /workspace/SUITE_FAIL; exit 1; }
rm -f /workspace/SUITE_FAIL

if [ ! -f $S/suite_dirs.npz ]; then
  python -u suite_dirs.py 2>&1 | tee $S/dirs.log
  grep -q DIRS_DONE $S/dirs.log || fail dirs
fi

if [ ! -f $S/lines_fve_$MODEL.csv ]; then
  python -u suite_fve.py 2>&1 | tee $S/fve_$MODEL.log
  grep -q FVE_DONE $S/fve_$MODEL.log || fail fve
fi

if [ ! -f $S/frontload_raw_$MODEL.json ]; then
  python -u suite_gen.py 2>&1 | tee $S/gen_$MODEL.log
  grep -q GEN_DONE $S/gen_$MODEL.log || fail gen
fi

echo SUITE_DONE | tee /workspace/SUITE_DONE
