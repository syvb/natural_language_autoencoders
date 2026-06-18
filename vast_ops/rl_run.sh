#!/bin/bash
# One continue-RL run from the released Qwen2.5-7B NLA, with a given length penalty.
# Starts from the released AV (HF) as actor + KL ref, released AR as critic. No DCP.
# Captures stdout (fve_nrm per step) + per-step rollout dumps (response_length per sample).
#
# Env: PENALTY (required), SEED (default 1234), STEPS (default 50), TAG (required),
#      ACTOR_GPUS/CRITIC_GPUS/ROLLOUT_GPUS (default 2/1/1), KL_COEF (default 0.01)
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate miles
export PYTHONPATH=/workspace/nla:${PYTHONPATH:-}
export NLA_EMBED_DUMP_DIR=/dev/shm/nla; mkdir -p "$NLA_EMBED_DUMP_DIR"
# Ray 2.55 blanks CUDA_VISIBLE_DEVICES for num_gpus=0 actors; miles' SGLang
# rollout engine (num_gpus=0) reads CVD via _to_local_gpu_id to pick its
# physical GPU slice. Blanking -> engine sees "No accelerator". Fix: stop the
# override AND set the explicit full device list so miles maps the rollout to
# the right GPU (after actor/critic) instead of colliding on GPU 0.
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
# Set the Ray "noset" var in the DRIVER so every worker inherits it at fork,
# BEFORE Ray's set_visible_accelerator_ids runs. miles sets it only in
# runtime_env (applied too late under modern Ray) → first rollout-engine attempt
# sees empty CVD → "No accelerator" → deadlock. miles self-assigns GPUs per
# group, so disabling Ray's CVD management globally is consistent with its design.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
NGPU=$(nvidia-smi -L | wc -l)
export CUDA_VISIBLE_DEVICES=$(python3 -c "print(','.join(str(i) for i in range($NGPU)))")

PENALTY="${PENALTY:?set PENALTY}"; SEED="${SEED:-1234}"; STEPS="${STEPS:-50}"; TAG="${TAG:?set TAG}"
AGPU="${ACTOR_GPUS:-2}"; CGPU="${CRITIC_GPUS:-1}"; RGPU="${ROLLOUT_GPUS:-1}"; KL="${KL_COEF:-0.01}"
RESULTS="/workspace/results/$TAG"; mkdir -p "$RESULTS/rollout"
export NLA_LENGTH_PENALTY="$PENALTY"

# wandb (primary metric sink + cloud persistence). Set WANDB_KEY to enable.
WANDB_FLAGS=()
if [ -n "${WANDB_KEY:-}" ]; then
  # miles sets the wandb run NAME from --wandb-group, so use the per-run TAG
  # (pen<val>_seed<seed>) → each sweep run is individually identifiable when pulling.
  WANDB_FLAGS=(--use-wandb --wandb-key "$WANDB_KEY"
    --wandb-project "${WANDB_PROJECT:-nla-length-penalty}"
    --wandb-group "$TAG"
    --wandb-run-id "$TAG" --disable-wandb-random-suffix)
fi

# KL anchor to the released AV (ref on GPU, loads from HF dir). Conditional so
# KL_COEF=0 cleanly drops the ref entirely (isolates the core pipeline; also a
# fallback since --nla-ref-on-gpu is marked UNTESTED in train_actor.py).
KL_FLAGS=()
if python3 -c "import sys; sys.exit(0 if float('$KL')>0 else 1)"; then
  KL_FLAGS=(--ref-load /workspace/models/av --nla-ref-on-gpu --use-kl-loss --kl-loss-coef "$KL")
fi

printf 'tag=%s penalty=%s seed=%s steps=%s gpus=%s/%s/%s kl=%s\n' \
  "$TAG" "$PENALTY" "$SEED" "$STEPS" "$AGPU" "$CGPU" "$RGPU" "$KL" | tee "$RESULTS/meta.txt"

cd /workspace/miles
python train.py \
  --train-backend fsdp \
  --custom-actor-cls-path nla.train_actor.NLAFSDPActor \
  --loss-type policy_loss --advantage-estimator grpo --force-use-critic \
  --n-samples-per-prompt 8 \
  --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
  --custom-generate-function-path nla.rollout.nla_generate.generate \
  --custom-rm-path nla.reward.nla_rm \
  --data-source-path nla.data_source.NLADataSource \
  --prompt-data /workspace/rldata/rl.parquet --input-key prompt \
  --hf-checkpoint /workspace/models/av \
  --nla-sidecar-source /workspace/models/av \
  --critic-load /workspace/models/ar \
  --nla-critic-sidecar-source /workspace/models/ar \
  --save "$RESULTS/actor" --critic-save "$RESULTS/critic" --save-interval "${SAVE_INTERVAL:-100000}" \
  --save-debug-rollout-data "$RESULTS/rollout/step_{rollout_id}.pt" \
  --critic-lr 1.41e-5 --lr 1.41e-5 --lr-decay-style constant \
  --actor-num-nodes 1 --actor-num-gpus-per-node "$AGPU" \
  --critic-num-nodes 1 --critic-num-gpus-per-node "$CGPU" \
  --rollout-num-gpus "$RGPU" \
  --rollout-max-response-len 150 --rollout-max-context-len 300 \
  --sglang-disable-radix-cache --sglang-context-length 300 \
  --router-history-backend none --router-policy round_robin \
  --router-disable-circuit-breaker --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
  --rollout-batch-size 32 --global-batch-size 256 --micro-batch-size 4 \
  "${KL_FLAGS[@]}" \
  --num-rollout "$STEPS" \
  --loss-mask-type qwen \
  --rollout-seed "$SEED" \
  "${WANDB_FLAGS[@]}" \
  2>&1 | tee "$RESULTS/train.log"
echo "=== run $TAG done $(date) ==="
