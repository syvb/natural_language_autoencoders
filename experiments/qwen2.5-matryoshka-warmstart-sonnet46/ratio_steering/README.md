# Two-concept ratio test (yellow vs sycophancy)

A causal intervention designed to test the **ordering** property of the matryoshka NLA —
the axis where single-direction steering cannot differentiate models (any competent NLA
detects a large injected direction; only an ordering-trained one should *rank* concepts
by their share of the injection).

## Design

Into each of 40 neutral base activations `b` (L20 last-token, training convention) inject
**two** genuine contrast directions at once:

```
a = b + ||b|| * (r_y * v̂_yellow + r_s * v̂_syco),   sqrt(r_y² + r_s²) = R
```

- log-ratio `λ = log2(r_y / r_s)` swept over 9 points in [−3, 3]
- total strength `R ∈ {0.6, 0.9, 1.3}` (injection renormalizes to `injection_scale` as always)
- 2 sampled reps per condition at T=1 → 2160 generations per model
- models: v3 RL AV (`syvb/nla-qwen2.5-7b-L20-v3-rl` iter_0000200) vs kitft baseline AV

Every line of every generation is judged 3-way for BOTH traits in one call
(0 none / 1 related / 2 direct), same rubric style as `post_figs/judge_yellow_lines.py`.

**Primary readout:** P(yellow directly mentioned *before* sycophancy) vs λ — a
psychometric curve per model. Prediction: the matryoshka model's first-mention order
tracks the injected component ratio (steep logistic), the baseline is flatter.

## Files

- `gen_ratio.py` — box-side generation (env: `AV`, `OUT_NAME`, `SHARD/NSHARDS`, `NLA_GEN_TEMP`)
- `driver_ratio.sh` — box driver (`MODEL=v3|kitft`): deps → downloads → `build_dirs_min.py` → sweep
- `judge_ratio.py` — local, OpenRouter Haiku 4.5, cached; both traits per call
- `plot_ratio.py` — figR1 (psychometric), figR2 (presence dose-response), figR3 (first index),
  printed logistic slopes
- `results/` — raw + judged JSONs and figures
