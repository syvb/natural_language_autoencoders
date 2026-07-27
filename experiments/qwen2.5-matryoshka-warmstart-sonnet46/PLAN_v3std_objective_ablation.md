# Plan: how much of matryoshka performance is just the better warm-start?

**Question.** The matryoshka models beat standard NLAs on short-prefix FVE /
front-loading — but the matryoshka runs also got a different (better) warm-start:
Sonnet-4.6 multi-snippet data that is *already* front-loaded by construction. Is
the truncation RL objective doing the work, or is the warm-start?

**Design: hold the warm-start fixed, vary ONLY the RL objective.** Two arms from
the identical v3 warm-start pair
(`syvb/nla-qwen2.5-7b-L20-{av,ar}-matryoshka-sonnet46-v3`):

| arm | objective | status |
|---|---|---|
| **M** (matryoshka) | reward on random-length prefix, K ~ U[1,120] shared per group | **exists** — the v3 RL run, reuse `syvb/nla-qwen2.5-7b-L20-v3-rl` iter 50/100/150/200 |
| **S** (standard) | reward on the full fixed-length output, K ≡ 120 | **new** — the only training spend |

If S reaches M's short-prefix FVE, matryoshka performance was mostly the
warm-start (+ generic RL polish). If S stays near the warm-start's short-prefix
numbers while M's climbed, the objective is the driver.

## KL: constant 0.03 for both, no decay

Neither arm gets a KL schedule (the miles pipeline has no KL-decay mechanism
anyway — `KL_LOSS_COEF` is a constant vs the frozen warm-start reference).
Pinning both to **0.03** lets Arm M be the *existing* v3 run verbatim, so we
only pay for one new run. Alternative rejected: KL=0 for both would require
retraining M too (the existing `kl0` reference in `rltrunc-gradguard` is
v1-era: tagged prompt, U[16,130] — not comparable), roughly doubling cost for
no attribution benefit.

### Is the same coefficient actually a fair leash given different lengths?

The arms have the SAME nominal penalty (coef 0.03, same frozen warm-start
reference) — but S generates 120 tokens every sample while M averages ~60, so
"same coefficient" would NOT mean "same effective pressure" if miles summed KL
over tokens. It doesn't: at our pin (`radixark/miles@051cd15`,
`miles/backends/training_utils/loss.py` → `cp_utils.get_sum_of_sample_mean`),
`kl_loss = sum_of_sample_mean(kl)` computes each sample's **masked per-token
MEAN** KL and gives every sample weight 1 regardless of length
(`(x_i*mask_i).sum() / mask_i.sum()`). So the per-token leash strength is
identical across arms; response length does not scale the penalty.

The residual asymmetry is inherent to the ablation itself: M's tokens beyond K
are never generated, so they get neither reward nor KL pressure that step —
but "which positions receive reward+KL gradient" *is* the objective being
ablated, not a nuisance variable. Holding the coefficient fixed is the correct
single-factor control. Empirical guard: compare the two arms' realized
`kl_loss` (per-token mean KL from ref) at plateau; if S ends up much more/less
drifted than M (v3 sat taut at ~2.5–2.9), report drift alongside FVE, and the
contingency control is a KL-matched rerun of S (coefficient tuned to match M's
realized drift) — not part of the base plan.

## Why Arm S is "fixed K=120", not truncation-off

The v3 AV never emits EOS (`NLA_NO_TRAIN_EOS=1`), so with truncation disabled
(`NLA_TRUNC_MAX_TOKENS=0`) every rollout hits the response cap, gets
`TRUNCATED→FAILED`, and collapses to the −2 penalty — no signal. Setting
`NLA_TRUNC_MIN_TOKENS=120 NLA_TRUNC_MAX_TOKENS=120` runs the **identical code
path** as Arm M (at-cap TRUNCATED samples scored, no token-limit penalty,
`max_new_tokens` capped) with a degenerate constant budget — i.e. the normal
full-length objective for this actor family, differing from M by literally one
env var. `resolve_truncation_config` accepts min==max (asserts `1 ≤ min ≤ max`)
and `sample_truncation_length(lo==hi)` returns the constant; add a trivial
pytest for both before spending GPU money (Phase 0).

Everything else matches the v3 run byte-for-byte: same warm-start ckpts, same
`rl_v3.parquet` (bullets prompt), 8-GPU actor4/critic2/rollout2, batch 512
(64×8 GRPO), lr 1e-5/1e-5, `ROLLOUT_MAX_RESP=160`, offset 0, grad-guard on,
T=1 rollouts, 200-step budget, save@50, `lmsysorg/sglang:v0.5.7-cu129-amd64` +
`setup_rl_box_lmsys.sh`, same nla commit as recorded in the v3 run's
`launch_config` (in `syvb/nla-qwen2.5-7b-L20-v3-rl-checkpoints`).

## The critic confound (mandatory control, not optional)

Round-trip FVE is scored through a co-trained critic. M's critic co-trained on
short prefixes; S's critic will see only 120-token inputs for 200 steps (its
warm-start was U[1,120]-pre-calibrated, but that calibration can drift). So an
M-over-S gap at short prefixes could be **critic-side, not actor
front-loading** — exactly the suffix-RL lesson (suffix-blindness was
critic-side; recal alone moved suffix@60 from −0.21 to +0.49). Controls:

1. **Cross-critic matrix**: eval all 3 actors (WS, S@200, M@200) under all 3
   critics (WS-AR, S-AR@200, M-AR@200) — 9 sweeps, `eval_round_trip_fve.py`
   already takes `AV_DIR`/`AR_DIR` envs. Actor front-loading must persist under
   the shared frozen WS critic to count.
2. **Critic-free order metric**: `eval_paraphrase_order.py` order-optimality
   (model order vs random vs greedy-oracle) for S@200 vs M@200.

## Scope decision (2026-07-27): Phases 0+1 only for now

Approved: Phase 0 (CPU prep — done: `test_fixed_budget_arm_via_env`,
`run_rl_v3std.sh`) + Phase 1 (50-step pilot + equal-step evals). Phase 2
(resume to 200) and Phase 3 (full cross-critic matrix) await the pilot
readout. Before destroying the pilot box, push the resumable iter_50 state
(actor DCP + optimizer + critic hf) to
`syvb/nla-qwen2.5-7b-L20-v3std-rl-checkpoints` so Phase 2 can resume on a
fresh box — prefer a $0/GB-bandwidth host (the state is ~85 GB up).

### Phase 0+1 timeline / cost

| # | step | wall-clock | cost @8×H100 ~$18/hr | @8×H200 ~$32/hr |
|---|---|---|---|---|
| 0 | tests + wrapper (dev box, done) | ~0.5 h | $0 | $0 |
| 1a | provision + egress speed-test | 0.25–0.5 h | ~$5 | ~$10 |
| 1b | box setup (`setup_rl_box_lmsys.sh`: miles+deps, cached flash-attn wheel, ckpt/data downloads, rl_v3.parquet build) | 0.75–1 h | ~$15 | ~$28 |
| 1c | Arm S, 50 steps @ ~55–70 s/step | ~1 h | ~$18 | ~$32 |
| 1d | evals on-box: S@50 ×{own, WS} critic + M@50 ×{own, WS} critic (150 held-out each) | ~0.75 h | ~$14 | ~$24 |
| 1e | push resumable state + iter_50 inference ckpts to HF; destroy | 0.25–0.5 h | ~$7 | ~$12 |
| | **total** | **~3.5–4 h** (unattended after 1b) | **~$60** | **~$105** |

WS baseline numbers are already on file (`v3_warmstart_results/v3_fve_baseline.txt`)
— no re-eval needed. Pilot compute is not throwaway: Phase 2 resumes from the
pushed iter_50 state.

## Phases

### Phase 0 — CPU, this box, $0
- pytest: min==max config accepted; constant-budget draw; groups all get 120.
- `run_rl_v3std.sh`: thin wrapper over `run_rl_v3.sh` pinning the two env vars,
  a separate `RUN_DIR=/workspace/rl_v3std`, wandb group `qwen2.5-7b-L20-rl-v3std`
  (same project), and asserting the launch-config dump shows min=max=120.

### Phase 1 — GPU pilot = first 50 steps of Arm S (~$25–70, ~2–3 h box time)
Smoke must match prod: same 8×H100 topology and **batch 512** (the historical
NaN was batch-size-sensitive — clean at 128, died at 512 — and critic1 smokes
false-pass; do not downscale anything but step count). 8×H200 fallback if no
H100 offers (price at provisioning; ~$20–35/hr). `NUM_ROLLOUT=50`.
Expect ~55–70 s/step (always generates 120 tokens vs v3's ~60 avg → slower
than v3's 41 s/step); 50 steps ≈ 1 h + ~1 h setup.

**Stability gates (wandb + rollout text dump):** `loss_nonfinite=0`; grad-guard
skip rate ≈0 (v3 had zero; sustained >10% → stop, investigate); raw_reward
improving, never pinned at −2; critic `fve_nrm` climbing from ~0.3;
`resp_len` ≈120 flat; KL loss rising smoothly (v3 early-curve shape); no CJK;
line structure intact.

**Pilot signal eval** (same box after training, or a 1×H100): truncation-sweep
FVE, 150 held-out `av_eval_v3` rows, prefix lens {1,2,5,10,30,60,120,full},
same decode settings as the existing v3 numbers, for:
- S@50 under its own critic AND under the WS critic,
- M@50 (public ckpt) under the WS critic — the equal-step comparison,
- WS baseline (already on file: 10-tok 0.11, 30 0.35, 60 0.44, 120 0.47, full 0.44).

**Decision rules:**
- Any stability gate fails → fix before more spend.
- S@50 full-length FVE not moving up from WS's 0.44 → the arm's reward wiring
  is broken (standard RL must at minimum improve full-length); stop and debug.
- Otherwise proceed regardless of which way the short-prefix signal points —
  the pilot IS the first 50 steps of the full run (resume, zero waste). The
  50-step readout is directional only; plateau is ~110 steps.

### Phase 2 — full run: resume to 200 (~$45–120, ~3–4 h)
Re-run the same command with `NUM_ROLLOUT=200` (launcher auto-resumes from
`$RUN_DIR/actor/iter_50`). Save@50, push per-iter {av,ar} to
`syvb/nla-qwen2.5-7b-L20-v3std-rl` (same layout as the v3-rl repo), plus the
resumable final state to `...-v3std-rl-checkpoints`. Destroy the box.

### Phase 3 — eval + analysis (1×H100/H200, ~$5–10)
- 9-cell cross-critic truncation sweeps (actors × critics), 150 held-out.
- Order-optimality (+ paraphrase leg, secondary) for S@200 vs M@200.
- Per-iter sweep S@{50,100,150,200} under WS critic → trajectory figure next to
  M's (M@{100,150} need one-off sweeps too if we want the full trajectories;
  M@50/M@200 already covered above).

## Readout

Primary table, FVE at each prefix L under the **frozen WS critic**:

- warm-start share of matryoshka's short-prefix gain ≈ (S − WS) / (M − WS) at
  L ∈ {5, 10, 30}; headline number is L=10 (known endpoints, own-critic:
  WS 0.11 → M 0.45).
- Full-length check: S should ≥ M at L=120/full (standard objective optimizes
  exactly this); if S also loses at full length, suspect the run, not the story.
- Verdict examples: S@10-tok ≈ 0.40+ → "mostly warm-start"; ≈ 0.15–0.20 →
  "objective is the driver"; in between → report the share.

## Caveats / scope

- **This isolates the objective ON TOP of the matryoshka warm-start.** The
  other 2×2 cell (standard-format warm-start + matryoshka objective) needs a
  standard-`<explanation>` SFT from the kitft base — optional ~$70 extension
  if the answer comes back "mostly warm-start" and we want the full decomposition.
- **Reward-token asymmetry**: S's reward sees ~2× the tokens/sample (120 vs
  E[K]≈60.5). If that asymmetry matters for interpretation, the follow-up
  control is fixed K=60 (token-matched), another ~$50. Not in scope now.
- **Single seed per arm**; differences <~0.05 FVE at a given L shouldn't be
  over-read. M is a run from 2026-07-02 on the same recipe/stack — pin S's box
  to the same image + nla commit; record `launch_config` as always.
- 8/150 stray-CJK leak existed at M@200 — track the same counter for S.

## Cost

| phase | box | est. |
|---|---|---|
| 0 CPU prep | — | $0 |
| 1 pilot (50 steps + eval) | 8×H100/H200 | $25–70 |
| 2 resume to 200 | same box, same session | $45–120 |
| 3 evals | 1×H100/H200 | $5–10 |
| **total** | | **~$75–200** (vs ~2× if M were retrained) |
