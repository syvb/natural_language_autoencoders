# Gate C — do diff labels carry extractable beyond-content information?

**Verdict: FAIL (pre-registered). Both diff-label arms are significantly
*worse* than the content-only control at predicting the residualized layer
diff. The causal KL judge agrees. No $300 RL smoke test.**

Per the pre-registered wording, the licensed conclusion is that the
**frozen-AV-derived label family is exhausted** — not that "activation diffs
resist verbalization" in general.

## Question

Gates B and C0 showed that position *content* predicts most of a layer diff
(v22 − v18), and that diff-injection labels are weakly grounded beyond
content (residual-alignment ceiling ≈ 0.23 for the diffav register, ensemble
probe). Gate C asks the decision question: if we *train* a critic on each
label source, does any diff-label arm extract more of the
content-unpredictable part of the diff than a critic trained on pure
content explanations? If yes → fund an RL smoke test (~$300). If no → the
warm-start story for layer-diff NLAs has no signal to amplify.

## Design (half-scale, post-adversarial-review)

- **Data:** 15,000 positions (2,500 FineWeb docs × 6, seed 1), layers
  18/22 of Qwen2.5-7B-Instruct, doc-level train/holdout split (12,048 /
  2,952 positions). Raw vectors throughout (`norm="none"`).
- **Arms** (all critics initialized from the released L20 AR
  `kitft/nla-qwen2.5-7b-L20-ar`, trained 2 epochs, lr 1e-5, eff. batch 64,
  direction-only MSE on the raw diff target, matched n = 8,000):
  - **arm0p** — content control: `Earlier: <L18 explanation>\n\nLater:
    <L22 explanation>` (frozen-AV content reads, temp 0.4).
  - **armA** — diff-injection AV's own verbalization of v22−v18 (temp 0.4).
  - **armB** — Claude Haiku 4.5 hybrid labels (user-authored v3 prompt):
    synthesizes the two content explanations + 4 diff-injection readings
    into one ≤3-snippet difference description. 8,150 labels, Batches API,
    ≈$12, 0 errors.
  - **armA_full** — armA at full n = 11,756 (descriptive data-size slope).
- **Primary metric:** cos(critic prediction, residual diff), where the
  residual removes the ridge prediction v18 → d (fit on train docs only,
  λ = 1e4 chosen on an inner doc split; holdout R² = 0.31 — at n = 180 in
  Gate C0 this was an unusable 6%, at 12k it is well-estimated).
- **Primary contrast (pre-registered):** paired per-position
  Δ = cos(pred_arm, resid) − cos(pred_arm0p, resid) on shared holdout
  positions; doc-clustered bootstrap (10k), Holm over the two contrasts.
  PASS = CI > 0 and Δ ≥ +0.05; MARGINAL = +0.02–0.05; FAIL = ±0.02.
- **Tie-breaker (veto only):** causal KL patching judge — patch
  v18 + scaled prediction into `hidden_states[22]` at the position, KL of
  next-token distribution vs natural. Sanity (patching the true diff) must
  be ≈ 0.

## Results

### Per-arm reconstruction (holdout)

| arm | n | cos vs raw d | cos vs residual [95% CI] | beyond-arm0p |
|---|---|---|---|---|
| **arm0p (content)** | 2,952 | **0.705** | **0.295** [0.291, 0.299] | 0 (by constr.) |
| armA (diffav) | 2,908 | 0.666 | 0.266 [0.262, 0.270] | 0.039 |
| armA_full (11.8k) | 2,908 | 0.676 | 0.268 [0.264, 0.272] | 0.048 |
| armB (hybrid) | 150 | 0.570 | 0.157 [0.138, 0.175] | 0.034 |

### Primary contrasts vs arm0p (paired, doc-clustered)

| contrast | n shared | Δ cos_resid | 95% CI | verdict |
|---|---|---|---|---|
| armA − arm0p | 2,895 | **−0.029** | [−0.031, −0.028] | WORSE |
| armB − arm0p | 150 | **−0.148** | [−0.160, −0.135] | WORSE |

Not only does neither diff arm clear the +0.05 PASS bar — both are
significantly *below* the content control, on the metric specifically
designed to favor any beyond-content information.

### Causal KL judge (144 verified positions, sanity = 0.0006)

| candidate | KL mean | KL median | vs arm0p (Wilcoxon) |
|---|---|---|---|
| zero diff (patch v18 alone) | 0.598 | 0.360 | worse, p = 4e-16 |
| **arm0p** | **0.267** | **0.178** | — |
| armA | 0.390 | 0.238 | worse, p = 2.5e-4 |
| armB | 1.405 | 0.700 | worse, p ≈ 0 |

The content control is also the best *functional* substitute for the true
diff. armB is catastrophically bad — worse than patching no diff at all:
its predictions, rescaled to the true diff norm, actively damage the
next-token distribution. The veto condition (significantly worse) is met
for both diff arms, consistent with the primary metric.

### Secondary observations

- **Data-size slope is flat.** armA 8k → 11.8k (+47% data) buys +0.010 raw
  cos and +0.002 residual cos. The diffav register is information-limited,
  not data-limited — matching the Gate C0 ensemble-probe ceiling (~0.23
  residual alignment for raw diffav texts; the trained critic squeezes a
  bit more, 0.27, but from below arm0p).
- **Best-of-16 (descriptive, oracle selection):** scoring the 16
  diff-injection reads per holdout position with the trained armA critic:
  single sample 0.273 → best-of-16 0.296 (permuted-target null: 0.013).
  Real selection headroom exists, but even *oracle* selection over 16
  samples only just matches the content control's mean (0.295).
- **"Beyond-arm0p" alignment is positive but tiny** (0.03–0.05 after
  projecting out arm0p's prediction direction per position): diff labels do
  carry a sliver of unique signal, but far below the +0.05/Δ>0 bar and far
  below the ~0.5 functional break-even from Gate C0's KL geometry.
- A caution on the residual: ridge is linear, so cos_resid 0.295 for the
  *content* arm shows the "residual" still contains content-correlated
  structure reachable nonlinearly. This biases the contrast *toward* the
  content arm on shared structure — which is exactly why the pre-registered
  rule scored diff arms by their *margin over* arm0p rather than absolute
  residual alignment, and why the beyond-arm0p projection is reported as
  robustness. Both views agree.

### Worked examples

See `EXAMPLES.md` — side-by-side arm0p / armA / armB labels with per-arm
residual cosines for positions where each register wins.

## Why armB (hybrid Claude labels) underperformed so badly

The v3 labels are abstract meta-descriptions ("Strengthened focus on…",
"Sharpened expectation of…") — a register the AR backbone never saw in
training. The critic had to learn this register from 8k pairs from a cold
start; armA's diffav texts are in-distribution for the AR by construction.
The KL blow-up (1.4) plus low residual cos suggests the armB critic's
predictions sit in a damaging direction when forced to the true diff norm.
n = 150 eval also makes armB the noisiest estimate, but the CI excludes
even armA's level. A labeler whose output mimics the AV register might do
better — but that is again a frozen-AV-derived signal, which this gate was
designed to bound.

## Decision

Pre-registered rules → **FAIL** for both arms (in fact significantly
negative). **Do not fund the ~$300 RL smoke test from this label family.**

What this does and does not establish:
- Established: every label source derived from the frozen L20 AV pair
  (content reads, diff-injection reads, and LLM syntheses thereof) trains a
  critic that is *no better than content alone* at capturing the
  content-unpredictable part of a layer diff — under a metric and judge
  chosen to detect exactly that.
- Not established: that layer diffs are unverbalizable. The bottleneck is
  the label source: everything we can cheaply generate routes through an AV
  that was never trained to read diffs.

Pivot candidates (in order of our enthusiasm): (1) attention-output /
MLP-output NLAs — verbalize what a *component writes* instead of a layer
diff, sidestepping the "diff ≈ re-representation" problem found in Gate A/B
(cos(diff, v22) = 0.86); (2) train an AV *with RL from scratch* on diff
reconstruction reward (no warm-start labels needed — but no cheap smoke
test either); (3) larger gap diffs (e.g. 8→24) where re-representation is a
smaller fraction of the diff.

## Incidents & costs

- HF Hub anonymously rate-limited a pod IP → all downloads silently failed
  while `setup.sh` reported done; fixed by passing `HF_TOKEN`. Also `hf
  download` with multiple `--include` patterns mis-parsed (downloaded the
  excluded `acts22` and skipped two includes) — list files explicitly.
- KL judge needed `accelerate` (transformers `device_map` path) — one crash,
  one relaunch.
- The 8,150-label batch finished in ~20 min, 8,150/8,150 succeeded.
- Costs (whole gate): session-1 generation fleet + smokes ≈ $21, label
  batch ≈ $12, session-2 training pods (2 × A100-SXM ≈ 2.8 pod-hours)
  ≈ $4.5. **Total ≈ $37 of the $60 budget.**

## Artifacts

- Code: `extract2.py`, `gen2.py`, `build_arms.py`, `hybrid_v2.py`,
  `train_critic.py`, `predict_critic.py`, `kl_judge.py`, `evaluate.py`,
  `kl_bo16.py`, `pod/*.sh` (this directory).
- Numbers: `eval_results.json`, `kl_results.json`, `kl_bo16_results.json`,
  `RESULT_*.json`.
- Data (public): hf.co/datasets/syvb/nla-layer-diff-experiments under
  `gate_c/` — activations, diff targets, all generated texts, hybrid
  labels, arm pair files, held-out predictions. Trained checkpoints were
  not kept (~11 GB each, reproducible for ~$2 via `train_critic.py`).
- wandb: project `nla-layer-diff`, runs `arm0p_8k`, `armA_8k`, `armB_8k`,
  `armA_full`.
