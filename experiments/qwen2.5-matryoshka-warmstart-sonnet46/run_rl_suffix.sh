#!/bin/bash
# SUFFIX-truncation ("left-matryoshka") RL — the mirror of run_rl_v3.sh.
# Same v3 bullets-format warm-start pair, same RL parquet, same batch profile;
# the ONE experimental change is the reward direction: the critic scores the
# LAST ~U[1,120] content tokens of each rollout instead of the first, pushing
# the model to put the most important information at the END.
#
# Deliberate reuse of the v3 warm-start (no mirrored re-SFT):
#   - left-RL and v3's right-RL then start from the IDENTICAL checkpoint, so
#     the comparison isolates the reward direction alone. The shared init is
#     biased TOWARD front-loading (matryoshka targets), so any back-loading
#     result is conservative.
#   - the AR critic was pre-calibrated on U[1,120] PREFIX truncations — suffix
#     inputs are out-of-distribution at step 0. Expect the online critic to
#     re-calibrate over the first tens of steps; the grad-finiteness guard
#     turns the historical failure mode (NaN death) into a visible skip-rate.
#     Gate: critic fve_nrm off the floor and climbing by ~step 50; if not,
#     fall back to a mirrored warm-start before judging the actor.
#
# Suffix-mode mechanics (nla/truncation.py "suffix"):
#   - generation is NEVER capped: the AV never emits EOS, so every rollout runs
#     to ROLLOUT_MAX_RESP tokens — full trajectory, fixed length;
#   - the actor trains on the FULL trajectory (tokens/loss-mask/logprobs not
#     sliced — the prefix conditions the scored suffix);
#   - only the critic input (reward forward + co-training tokens) is cut to
#     the last k tokens, same per-group k at both consumers;
#   - the k draw uses the SAME RNG stream as tokens mode: identical budget
#     sequence to a v3 run with the same seed.
#
# ROLLOUT_MAX_RESP == NLA_TRUNC_MAX_TOKENS (120) is REQUIRED for the mirror,
# not a tuning choice. With RESP=120, position p is scored with probability
# (p+1)/120 — the exact mirror image of v3's (120-p)/120 prefix scoring. Any
# RESP > max budget creates a head region NEVER scored at any k (at v3's
# RESP=160: 40 tokens), i.e. a free scratchpad where the gradient-cheapest
# policy is to keep front-loading (KL-cheap vs the front-loaded reference)
# and DUPLICATE content into the tail — and suffix-FVE cannot distinguish
# duplication from the reordering under test. Eval generation must match:
# pass NLA_GEN_MAX_NEW=120 to eval_round_trip_fve.py for every arm.
#
# KL default 0.01 (v3 ran 0.03): the KL reference is the FRONT-loaded
# warm-start, so a strong anchor would penalize exactly the ordering change
# this run exists to produce. 0.01 = v1's setting, known stable. Known cost
# (accept for the existence run): v3's FVE gains were already substantially
# lexical at KL 0.03, and a weaker anchor + recalibrating critic can worsen
# that — read the result WITH the paraphrase probe, and don't claim "trains
# as well as v3" across a KL difference.
#
# TWO-PHASE LAUNCH (critic burn-in first): the reused AR critic — which IS
# the reward model — was pre-calibrated on PREFIX cuts; suffix cuts are
# out-of-distribution at step 0, and v3 attributes its stability + timeline
# precisely to starting task-calibrated. So calibrate it on frozen-actor
# rollouts before letting the actor move:
#
#   ACTOR_LR=0 NUM_ROLLOUT=25 bash run_rl_suffix.sh   # critic-only burn-in
#   NUM_ROLLOUT=100           bash run_rl_suffix.sh   # resumes iter_25; 75 actor steps
#
# (Burn-in length must be a multiple of SAVE_INTERVAL=25 or nothing is saved
# to resume from. Judge actor progress from the iter_25 checkpoint, not
# iter_0.) Gate at the end of phase 2: gate on critic fve_nrm having
# recovered during burn-in, short-SUFFIX (k=1-10) FVE above the warm-start
# baseline, grad-skip rate ~0. A positive signal = go. A flat signal is
# NO-GO only after extending to ~120 actor steps (v3's plateau was ~110).
#
# DEFAULT PROFILE: one 8×H100 node (actor 4 / critic 2 / rollout 2), 512-batch
# (64×8) — resume by re-running with a higher NUM_ROLLOUT (checkpoints +
# optimizer state every SAVE_INTERVAL).
set -euo pipefail

# ─────────────────── truncation: SUFFIX mode (left-matryoshka) ──────────────
# Hard-pinned, not defaulted: this launcher IS the suffix experiment; a
# lingering NLA_TRUNC_MODE=tokens from a v3 shell would silently rerun v3
# under the suffix RUN_DIR/wandb group. Use run_rl_v3.sh for tokens mode.
export NLA_TRUNC_MODE=suffix
export NLA_TRUNC_MIN_TOKENS="${NLA_TRUNC_MIN_TOKENS:-1}"
export NLA_TRUNC_MAX_TOKENS="${NLA_TRUNC_MAX_TOKENS:-120}"
# Suffix mode never uses the opening offset (no generation cap), but pin it
# for config-log hygiene and so a stale export can't confuse a later run.
export NLA_TRUNC_OPENING_OFFSET="${NLA_TRUNC_OPENING_OFFSET:-0}"
# Zero every lingering reward-shaping export (item mode from v2, quote/repeat/
# overlap penalties from the penalty-RL launchers). In suffix mode a stale
# penalty is extra-dangerous: it would act on the scored suffix only, and the
# actor could park the penalized behavior in the unscored head.
export NLA_TRUNC_MAX_ITEMS=0 NLA_ITEM_LEN_PENALTY=0
export NLA_QUOTE_PENALTY=0 NLA_REPEAT_PENALTY=0 NLA_OVERLAP_PENALTY=0

# ───────────────────── checkpoints (v3 warm-start, REUSED) ──────────────────
# Public HF: syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3 (AV)
#            syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3 (AR + value_head)
# Download to local dirs first; pass those as ACTOR_SFT_CKPT / CRITIC_SL_CKPT.
: "${INSTRUCT_MODEL:=Qwen/Qwen2.5-7B-Instruct}"
: "${ACTOR_SFT_CKPT:?v3 warm-start AV: LOCAL HF dir (download syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3)}"
: "${CRITIC_SL_CKPT:?v3 warm-start AR: LOCAL HF dir with value_head.safetensors (download syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3)}"
: "${RUN_DIR:=/workspace/rl_suffix}"
export HF_CKPT="${HF_CKPT:-$ACTOR_SFT_CKPT}"

# Same guard as v3: a tagged sidecar means the wrong (v1) prompt shipped.
if grep -q "<explanation>" "$ACTOR_SFT_CKPT/nla_meta.yaml" 2>/dev/null; then
    echo "FATAL: $ACTOR_SFT_CKPT/nla_meta.yaml has the v1 TAGGED actor template." >&2
    echo "Use the v3 warm-start (bullets sidecar) — see run_rl_v3.sh for the fix." >&2
    exit 1
fi

# ───────────────────────── KL 0.01 (see header) ─────────────────────────────
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"

# ───────────────────────── RL data (same v3 bullets parquet) ────────────────
export RL_PARQUET="${RL_PARQUET:-/workspace/out/rl_v3.parquet}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
if [ ! -f "$RL_PARQUET" ]; then
    echo "=== building RL parquet (bullets) -> $RL_PARQUET ==="
    EXPLANATION_FORMAT=bullets "${PYTHON:-python}" "$HERE/05_build_rl_parquet.py"
fi

# ───────────────────────── single-node 8×H100 profile ──────────────────────
export ACTOR_NODES=1 ACTOR_GPUS="${ACTOR_GPUS:-4}"
export CRITIC_NODES=1 CRITIC_GPUS="${CRITIC_GPUS:-2}"
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
export ACTOR_LR="${ACTOR_LR:-1e-5}" CRITIC_LR="${CRITIC_LR:-1e-5}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-25}"
NUM_ROLLOUT="${NUM_ROLLOUT:-75}"
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"; mkdir -p "$NLA_EMBED_DUMP_DIR"

# MUST equal NLA_TRUNC_MAX_TOKENS — see the mirror argument in the header.
# Suffix mode never caps generation and the AV never emits EOS, so EVERY
# rollout is exactly this many tokens. Expect slower steps than v3's ~41s —
# v3's average generation was much shorter than its cap.
ROLLOUT_MAX_RESP="${ROLLOUT_MAX_RESP:-$NLA_TRUNC_MAX_TOKENS}"
ROLLOUT_MAX_CTX="${ROLLOUT_MAX_CTX:-512}"
if [ "$ROLLOUT_MAX_RESP" -ne "$NLA_TRUNC_MAX_TOKENS" ]; then
    echo "FATAL: ROLLOUT_MAX_RESP ($ROLLOUT_MAX_RESP) != NLA_TRUNC_MAX_TOKENS ($NLA_TRUNC_MAX_TOKENS)." >&2
    echo "Any response length above the max suffix budget is a never-scored head region" >&2
    echo "(duplication scratchpad) — the mirror requires them equal. See the header." >&2
    exit 1
fi

# ───────────────────────── resume detection ────────────────────────────────
LATEST_ACTOR=""; LATEST_CRITIC=""
if compgen -G "$RUN_DIR/actor/iter_*" > /dev/null 2>&1; then
    LATEST_ACTOR="$(ls -d "$RUN_DIR"/actor/iter_* | sort | tail -1)"
    CRITIC_ITER="$(basename "$LATEST_ACTOR")"
    [ -d "$RUN_DIR/critic/$CRITIC_ITER/hf" ] && LATEST_CRITIC="$RUN_DIR/critic/$CRITIC_ITER/hf"
fi
if [ -n "$LATEST_ACTOR" ]; then
    export LOAD="$LATEST_ACTOR"
    [ -n "$LATEST_CRITIC" ] && export CRITIC_LOAD="$LATEST_CRITIC"
    echo "=== RESUMING from $LATEST_ACTOR (critic: ${LATEST_CRITIC:-<SFT>}) ==="
else
    export LOAD="${LOAD:-none}"
    echo "=== FRESH suffix RL start from v3 warm-start HF checkpoints ==="
fi

# ───────────────────────── wandb (always on) ───────────────────────────────
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat /root/.wandb_key)}"
WANDB_ARGS=(--use-wandb --wandb-project nla-rl-matryoshka-sonnet46-v3
            --wandb-team octahedral-systems
            --wandb-group "${WANDB_GROUP:-qwen2.5-7b-L20-rl-suffix}" --wandb-mode online)

# ───────────────────────── dump exact config ───────────────────────────────
mkdir -p "$RUN_DIR"
CFG="$RUN_DIR/launch_config.$(date -u +%Y%m%dT%H%M%SZ).txt"
{
    echo "# NLA suffix (left-matryoshka) RL launch config — $(date -u +%FT%TZ)"
    echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_status=$(git -C "$REPO_ROOT" status --porcelain | tr '\n' ';')"
    for v in INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR RL_PARQUET KL_LOSS_COEF \
             NLA_TRUNC_MODE NLA_TRUNC_MIN_TOKENS NLA_TRUNC_MAX_TOKENS \
             NLA_TRUNC_OPENING_OFFSET NLA_TRUNC_SEED NLA_TRUNC_MAX_ITEMS NLA_ITEM_LEN_PENALTY HF_CKPT \
             ACTOR_NODES ACTOR_GPUS CRITIC_NODES CRITIC_GPUS ROLLOUT_GPUS \
             ACTOR_LR CRITIC_LR SAVE_INTERVAL NUM_ROLLOUT ROLLOUT_MAX_RESP ROLLOUT_MAX_CTX \
             LOAD REF_LOAD CRITIC_LOAD; do
        echo "$v=${!v:-}"
    done
    echo "extra_args=$*"
} | tee "$CFG"

export INSTRUCT_MODEL ACTOR_SFT_CKPT CRITIC_SL_CKPT RUN_DIR
echo "=== suffix RL start $(date -u +%FT%TZ) — $NUM_ROLLOUT steps, save every $SAVE_INTERVAL ==="
echo "    truncation: $NLA_TRUNC_MODE mode, critic sees last ~U[$NLA_TRUNC_MIN_TOKENS, $NLA_TRUNC_MAX_TOKENS] content tokens, resp=$ROLLOUT_MAX_RESP, KL=$KL_LOSS_COEF, actor_lr=$ACTOR_LR"
# rl.sh fixes 128×8=1024; trailing flags OVERRIDE to the single-node profile +
# the v3 output caps (argparse: last value wins).
bash "$REPO_ROOT/configs/rl.sh" \
    "${WANDB_ARGS[@]}" \
    --rollout-batch-size 64 --global-batch-size 512 \
    --num-rollout "$NUM_ROLLOUT" \
    --rollout-max-response-len "$ROLLOUT_MAX_RESP" \
    --rollout-max-context-len "$ROLLOUT_MAX_CTX" \
    --sglang-context-length "$ROLLOUT_MAX_CTX" \
    "$@"
echo "=== suffix RL done $(date -u +%FT%TZ) ==="
