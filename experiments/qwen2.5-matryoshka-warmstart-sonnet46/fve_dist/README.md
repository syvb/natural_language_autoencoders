# Per-rollout round-trip FVE distribution on held-out examples

The headline round-trip FVE numbers are aggregates; this experiment shows the
**distribution over individual rollouts**. 250 held-out distinct-doc eval
samples × 2 independent sampled rollouts (T=1) per model; each rollout is
reconstructed by the model's own critic at full length and at a
10-content-token prefix.

Per-rollout FVE_i = 1 − mean((pn_i − gn_i)²) / denom, where pn/gn are
L2-normalized to `mse_scale` and denom is the population variance of the 250
normalized golds (raw-mean baseline). The mean of the histogram therefore
**equals the aggregate FVE** of `eval_round_trip_fve.py` — the histogram is a
decomposition of the headline number, not a new metric. Both models score the
same 250 docs with identical denominators (0.7230), so the panels are directly
comparable. Each model uses its own eval parquet format (`v3/av_eval_v3` vs
`av_eval`) and its own critic.

Models: ours = `syvb/nla-qwen2.5-7b-L20-v3-rl` @ `iter_0000200` (AV+AR);
baseline = `kitft/nla-qwen2.5-7b-L20-{av,ar}`.

## Results (500 rollouts per model)

| metric (full length) | matryoshka NLA (ours) | kitft baseline |
|---|---|---|
| mean (= aggregate FVE) | **+0.565** | **+0.743** |
| median | +0.672 | +0.776 |
| 10th–90th pct | +0.12 … +0.80 | +0.59 … +0.85 |
| rollouts with FVE < 0 | 7.0% | 0.6% |

| metric (10-token prefix) | ours | kitft |
|---|---|---|
| mean | **+0.433** | **−0.539** |
| median | +0.470 | −0.541 |
| rollouts with FVE > 0 | 96.2% | 1.0% |

Aggregates reproduce the known headline numbers (ours full ≈0.57–0.59, p10
≈0.44; kitft full ≈0.74, p10 ≈ −0.54).

## What the distribution adds

1. **The full-length gap is a failure tail, not a shifted bulk.** The two
   full-length distributions peak in the same place (modes ≈ 0.78 vs 0.83);
   ours pays for its ordering property with a long left tail — 7% of rollouts
   reconstruct worse than the mean baseline vs 0.6% for kitft.
2. **The tail is per-rollout luck, not hard docs.** Of the docs with a
   negative full-length rollout, 29/250 fail on exactly one of their two
   rollouts and only 3/250 on both. Sampling again usually fixes it.
3. **RL made reconstruction quality stochastic.** Variance decomposition of
   full-length per-rollout FVE: for ours, **71%** of the variance is
   within-doc (sampling noise; mean |Δ| between a doc's two rollouts = 0.21).
   For kitft it is **4%** (mean |Δ| = 0.02) — its encoding is nearly invariant
   to T=1 sampling even though the texts differ (0/250 identical pairs). The
   matryoshka objective (reward under random truncation, T=1 rollouts) traded
   determinism of the code for front-loading.
4. **At a 10-token budget the distributions are disjoint.** 96% of our
   rollouts are already better than the mean baseline after 10 tokens; 99% of
   kitft rollouts are *worse* than baseline at that budget. This is the
   matryoshka property at the level of individual rollouts, not just means.
5. CJK-flagged rollouts (any single CJK char; ours 33%, kitft 6% — the known
   sprinkle caveat) are only mildly worse for ours (mean +0.50 vs +0.60 clean)
   and account for a minority of the negative tail (11.7% vs 4.7% frac<0), so
   the tail is not just the CJK leak.

## Files

- `gen_fve_dist.py` — box script: AV rollouts (T=1) + critic reconstruction,
  per-rollout err²/cos at full/p10, JSON out. Sample selection identical to
  `sweep_fve.py`; shards partition one fixed 250-doc set and the denominator
  is computed over all 250 golds in every shard.
- `driver_fve_dist.sh` — quad-GPU box driver (one model:shard per GPU).
- `plot_fve_dist.py` — renders `results/fve_dist.png` (+ zoomed full-length
  panel) from the shard JSONs.
- `recon_budget.py` / `driver_recon.sh` — critic-only re-reconstruction of the
  saved rollouts at a grid of prefix budgets (1..200 tokens); validated by
  reproducing the k=10 aggregates to 4 decimals. Outputs
  `results/fvedist_budget_{v3,kitft}.json`. At k=20: ours +0.581 (98.8% of
  rollouts >0), kitft −0.420 (5.2% >0) — still fully separated.
- `plot_fve_dist_k.py K` — zoomed histogram at any stored budget
  (`results/fve_dist_k20_zoom.png`).
- `results/fvedist_{v3,kitft}_s{0,1}.json` — per-rollout rows incl. the
  generated texts (reusable for budget-limited readout evals without a GPU).

Run: quad RTX 4090 (~$1.5/hr), ~10 min wall-clock for all 1000 rollouts;
total cost ≈ $0.40.
