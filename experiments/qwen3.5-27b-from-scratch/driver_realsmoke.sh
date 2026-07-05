#!/bin/bash
# REAL-MODEL smoke: the actual Qwen3.5-27B pipeline end-to-end at production
# topology (reduced rows/steps only). Artifact-gated; SMOKE3_FAIL/SMOKE3_DONE.
set -uo pipefail
cd /workspace
export HF_TOKEN="$(cat /root/.hf_token)" HUGGING_FACE_HUB_TOKEN="$(cat /root/.hf_token)"
export HF_HUB_ENABLE_HF_TRANSFER=1
export NLA=/workspace/nla
export NLA_RUN_CONFIG="${NLA_RUN_CONFIG:-/workspace/nla/experiments/qwen3.5-27b-from-scratch/smoke_27b.env}"
EXP=/workspace/nla/experiments/qwen3.5-27b-from-scratch

fail() { echo "SMOKE3_FAIL: $1" | tee /workspace/SMOKE3_FAIL; exit 1; }
stage_done() { echo "=== STAGE_OK: $1 ($(date -u +%T)) ==="; }
rm -f /workspace/SMOKE3_FAIL

# ── [1] env (v0.5.10.post1 image; keep-transformers default) ────────────────
if ! grep -q SETUP_FULL_DONE /workspace/setup.log 2>/dev/null; then
  FLASH_ATTN_WHEEL=/workspace/flash_attn-2.8.3.post1-cp312-cp312-linux_x86_64.whl \
    bash "$EXP/setup_box.sh" > /workspace/setup.log 2>&1
  grep -q SETUP_FULL_DONE /workspace/setup.log || fail "setup_box"
fi
stage_done setup

# ── [2] fetch the 27B + matryoshka ───────────────────────────────────────────
if ! grep -q FETCH_DONE /workspace/fetch.log 2>/dev/null; then
  python "$EXP/00_fetch_inputs.py" > /workspace/fetch.log 2>&1
  grep -q FETCH_DONE /workspace/fetch.log || fail "00_fetch"
fi
stage_done fetch

# ── [3] L42 extraction on the real model ─────────────────────────────────────
if ! grep -q EXTRACT_DONE /workspace/extract.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=0 python "$EXP/01_extract_activations.py" > /workspace/extract.log 2>&1
  grep -q EXTRACT_DONE /workspace/extract.log || fail "01_extract"
fi
stage_done extract

# ── [4] bring-up gate (runbook order: before any SFT) ────────────────────────
if ! grep -q BRINGUP_PASS /workspace/bringup.log 2>/dev/null; then
  CUDA_VISIBLE_DEVICES=0 python "$EXP/10_bringup_check.py" > /workspace/bringup.log 2>&1
  grep -q BRINGUP_PASS /workspace/bringup.log || fail "10_bringup"
fi
stage_done bringup

# ── [5] datasets + 43-layer critic init (~17B) ───────────────────────────────
if ! grep -q BUILD_DONE /workspace/build.log 2>/dev/null; then
  python "$EXP/02_build_datasets.py" > /workspace/build.log 2>&1
  grep -q BUILD_DONE /workspace/build.log || fail "02_build"
fi
if ! grep -q CRITIC_INIT_DONE /workspace/critic_init.log 2>/dev/null; then
  bash "$EXP/03_prepare_critic.sh" > /workspace/critic_init.log 2>&1
  grep -q CRITIC_INIT_DONE /workspace/critic_init.log || fail "03_critic_init"
fi
stage_done datasets

# ── [6] 27B AV SFT, 20 real FSDP steps ───────────────────────────────────────
if ! compgen -G "/workspace/ckpt/av_ws/iter_*" > /dev/null; then
  bash "$EXP/run_av_sft.sh" --num-rollout 20 > /workspace/av_sft.log 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true; sleep 5
  compgen -G "/workspace/ckpt/av_ws/iter_*" > /dev/null || fail "av_sft produced no checkpoint"
fi
stage_done av_sft

# ── [7] 17B AR SFT (from-scratch critic), 20 steps ───────────────────────────
if ! compgen -G "/workspace/ckpt/ar_ws/iter_*/hf" > /dev/null; then
  bash "$EXP/run_ar_sft.sh" --num-rollout 20 > /workspace/ar_sft.log 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true; sleep 5
  compgen -G "/workspace/ckpt/ar_ws/iter_*/hf" > /dev/null || fail "ar_sft produced no hf checkpoint"
fi
echo "=== AR-SFT grad-guard skips: $(grep -c 'SKIPPED optimizer step' /workspace/ar_sft.log || true) ==="
stage_done ar_sft

# ── [8] 27B DCP -> HF convert ────────────────────────────────────────────────
if ! grep -q CONVERT_UPLOAD_DONE /workspace/convert.log 2>/dev/null; then
  UPLOAD=0 python "$EXP/04_convert_upload.py" > /workspace/convert.log 2>&1
  grep -q CONVERT_UPLOAD_DONE /workspace/convert.log || fail "04_convert"
fi
stage_done convert

# ── [9] REAL RL: 27B actor + 17B critic + sglang-served 27B, 10 steps ────────
if ! grep -q "RL done" /workspace/rl.log 2>/dev/null; then
  export ACTOR_SFT_CKPT=/workspace/hf_out/av_ws
  CRITIC_SL_CKPT="$(ls -d /workspace/ckpt/ar_ws/iter_*/hf | tail -1)"
  export CRITIC_SL_CKPT
  bash "$EXP/run_rl.sh" > /workspace/rl.log 2>&1 || true
  ray stop --force >/dev/null 2>&1 || true
  grep -q "RL done" /workspace/rl.log || fail "run_rl did not reach 'RL done'"
fi
[ -d /workspace/rl_run/actor/iter_0000008 ] || fail "RL save@8 missing (actor DCP)"
[ -d /workspace/rl_run/critic/iter_0000008/hf ] || fail "RL save@8 missing (critic hf export)"
stage_done rl

# ── [10] real export path (upload skipped) ───────────────────────────────────
UPLOAD=0 ACTOR_ORIGIN=/workspace/hf_out/av_ws \
  bash "$EXP/push_ckpt_to_hf.sh" 0000008 /workspace/rl_run > /workspace/export.log 2>&1
grep -q "PUSH_DONE iter_0000008" /workspace/export.log || fail "export validation"
stage_done export

# ── [11] health gates ────────────────────────────────────────────────────────
python "$EXP/check_health.py" /workspace/ar_sft.log --sft > /workspace/health_ar.txt 2>&1 || fail "ar_sft health"
python "$EXP/check_health.py" /workspace/rl.log --rl --taper > /workspace/health_rl.txt 2>&1 || fail "rl health"
stage_done health

touch /workspace/SMOKE3_DONE
echo SMOKE3_DONE
