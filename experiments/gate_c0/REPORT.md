# Gate C0 — can we make better diff labels? (residual injection, best-of-n, causal KL judge)

**Date:** 2026-06-10 · **Status:** complete — **two ideas tested, one new metric adopted, expectations recalibrated**
**Spend:** ~$2.5 (one A100-SXM pod, ~1.6 h; no API)
**Depends on:** `experiments/gate_a/`, `experiments/gate_b/`

## Questions

1. Does injecting the **residualized diff** (content directions projected out
   of `v22 − v18`) produce better-grounded diff explanations than injecting
   the raw diff?
2. Does **best-of-n selection** (n=8, temp 1.0, selected by the residualized
   AR metric) yield better labels — and is there RL-exploitable headroom?
3. Does a **causal patching judge** — independent of the AR — confirm any of
   it functionally?

## Setup

- 180 Gate A positions, pair 18→22. Residual `r = d ⊥ {AR(expl_L20), v18}`
  (removes ~44% of diff energy; residual norms 67–94, healthy).
- Generated 8 samples @ temp 1.0 per position for raw-diff injection
  (`dav8`) and residual injection (`resav8`); AR-reconstructed all 2,880.
- **Causal KL judge:** rebuild the source docs, verify live activations
  match the stored ones (cos ≥ 0.999, else skip), patch
  `v̂22 = v18 + ĝ/‖ĝ‖·‖d‖` into `hidden_states[22]` at the position, and
  measure next-token KL against the natural distribution. Sanity: patching
  the true diff gives KL ≈ 0.001 — the harness is exact.

## Results

**AR metric (vs residual target; selection-bias floor measured by selecting
on permuted targets):**

| condition | mean-of-8 | best-of-8 | net selection gain |
|---|---|---|---|
| raw-diff injection (dav8) | +0.229 | +0.247 | **+0.016** (bias floor +0.002) |
| residual injection (resav8) | +0.152 | +0.176 | +0.020 (floor +0.005) |

**Causal KL (lower = better; zero_diff = patch v18 unchanged = 0.457 mean /
0.32 median; true diff = 0.001):**

| candidate | KL mean | KL median |
|---|---|---|
| content baseline (AR of plain L20 explanation) | 0.298 | 0.172 |
| raw-diff labels, best-of-8 | 0.360 | **0.172** |
| raw-diff labels, first sample | 0.365 | 0.185 |
| content + residual-label (honest scales) | 0.351 | 0.235 |
| content + noise (same scales) | 0.391 | 0.264 |
| content alone (scaled) | 0.298 | 0.172 |
| Claude v2 label as diff vector | 2.93 | 1.27 |

## Findings

1. **Residual injection does NOT make better labels.** Despite being aimed
   at exactly the component we want, resav explanations ground *worse* than
   raw-diff explanations even on the residual metric (0.152 vs 0.229) —
   the residual vector is further off the AV's manifold (CJK 10% vs 6%) and
   reads less reliably. Idea rejected cleanly.
2. **Best-of-8 has real but modest headroom:** +0.016 net of selection bias
   on the AR metric, and a small KL improvement (median 0.185 → 0.172).
   RL would have gradient, but this looks like a grind, not a leap.
3. **The residual information in labels is real but functionally tiny.**
   Content+residual-label beats content+noise on the causal judge (0.351
   vs 0.391) — the labels carry *some* true residual signal — but loses to
   content alone, necessarily: a direction with cos ≈ 0.15 to the true
   residual only helps if injected at scale < 2·cos·‖r‖ ≈ 24, far below the
   honest scale ≈ 81. Until residual alignment exceeds ~0.5, adding the
   verbalized residual at honest amplitude hurts function.
4. **The causal KL judge works and is cheap** (sanity 0.001; ~10 min for
   1,600 patched forwards) — adopted as the independent evaluator for all
   future label/model comparisons. It also independently confirms Gate B:
   the layers' functional effect is substantially content-predictable
   (content baseline halves the zero-diff KL), and current diff labels
   recover essentially **zero function beyond content**.
5. Claude v2 labels decode to vectors that are functionally unrelated to
   the true diff (KL 2.93 ≈ random direction at full norm) — consistent
   with their role as behavioral framing, not grounding.

## Implication for Gate C

The honest bar is now well-quantified: current best labels sit at residual
alignment ~0.23 (raw-diff injection) and the functional payoff threshold is
~0.5. RL must roughly double residual alignment before the verbalized
change is functionally additive. Best-of-8 shows the gradient exists but is
shallow at the policy's current entropy. Recommend: proceed to the cheap
Gate C (critic training on diffav pairs + best-of-n at larger n), but with
expectations set by these numbers, and the KL judge as the success metric.

## Files

`bo_pod.py` (generate/score/klpatch), `select_candidates.py`, `run_pod.sh`,
`bo_texts.json` (2,880 samples), `bo_recons.npy` (+ok/conds),
`sel_idx_*.json`, `kl_results_round1.json` / `kl_results.json` (round 2,
composites), `residuals.npy`, candidate vectors (`cand_*.npy`).
