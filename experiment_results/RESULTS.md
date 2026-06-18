# Length penalty: FVE sacrificed per length reduction

NLA on Qwen2.5-7B, continue-RL from the released NLA with a per-token length
penalty (`NLA_LENGTH_PENALTY`), KL=0.01 anchored to the released NLA. Metrics are
**held-out** (1024 samples each, pile-10k docs disjoint from RL data). Reconstruction
is direction-only (L2-normalized activations): FVE = 1 − MSE/var, var≈0.72 (predict-
the-mean baseline); NMSE = 1 − FVE. Base = released NLA, no penalty.

## Per-model (held-out)

| model | mean len (tok) | len vs base | FVE | NMSE | FVE drop vs base | FVE retained |
|---|---|---|---|---|---|---|
| base | 144 | 100% | 0.681 | 0.319 | — | 100% |
| λ=0.002 | 50 | 35% | 0.588 | 0.412 | 0.093 | 86% |
| λ=0.006 | 34 | 23% | 0.478 | 0.522 | 0.203 | 70% |

## Marginal tradeoff (how much FVE per unit length cut)

| segment | Δlen (tok) | Δlen (%base) | ΔFVE | FVE per token saved | FVE per 10% len cut |
|---|---|---|---|---|---|
| base → 0.002  (first cut) | 94 | 65% | 0.093 | 0.0010 | 0.014 |
| 0.002 → 0.006 (further cut) | 16 | 11% | 0.110 | 0.0067 | 0.096 |
| base → 0.006  (overall) | 110 | 77% | 0.203 | 0.0018 | 0.026 |

## Takeaways

- **The first big cut is cheap, the rest is expensive (convex tradeoff).**
  Going base→λ=0.002 **shrinks length 65% (144→50 tok) for only ~0.10 FVE**
  (0.685→0.587, i.e. 86% of base FVE retained).
- Pushing further (λ=0.002→0.006, 50→34 tok) costs **another ~0.11 FVE for just**
  **16 more tokens** — the per-token cost jumps ~**6.7×** (0.0010 → 0.0069 FVE/tok).
- Practical read: **~halving the explanation (~50 tok) is the sweet spot**; below
  that, reconstruction degrades fast. Pick λ≈0.002 for max compression-per-FVE.

## Caveats
- 3 held-out points; var=0.72 is the canonical baseline (FVE is approximate to ~±0.01).
- KL anchoring kept text coherent; **KL-off runs are excluded** (λ=0.01 KL-off collapsed
  to ~30 tok / FVE~0.54 but is suspect of reward-hacking — see EXPERIMENT_STATE.md).
- Single seed (1234), 35 RL steps/run; not a converged production run.