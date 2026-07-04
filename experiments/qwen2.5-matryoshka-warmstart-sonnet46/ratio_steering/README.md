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

## Results (2026-07-04, T=1, 2160 gens/model, zero judge failures)

Logistic fits of P(yellow first) on λ, pooled over R. The pooled psychometric mixes
two effects — *which* concepts get verbalized (presence) and *in what order* — so the
decomposition is the primary result:

| metric | matryoshka (v3 iter200) | kitft baseline |
|---|---|---|
| pooled slope (presence + order) | **+1.59** | +1.03 |
| pooled midpoint λ | **+0.90** | +1.95 |
| **order-only slope** (both traits present, figR4) | **+0.88** | +0.10 ≈ 0 (CI [−0.02,+0.26]) |
| ties-excluded pooled slope | +1.68 | +1.31 |
| yellow presence @ λ=+3 (order-free) | **0.78** | 0.45 |

**The clean causal separation is the order-only row:** when both concepts appear, the
matryoshka model ranks them by their injected ratio (slope +0.88); the baseline's
order is statistically flat — it verbalizes both but does not rank. The presence gap
(0.78 vs 0.45 at 8:1 dominance) is independent and survives raw full-text keyword
matching (0.77 vs 0.41), so it lives in the generations, not the judge. Pooled-slope
gap: cluster bootstrap over bases +0.57 [+0.35, +0.78], p<0.0005; survives every
stress variant (no-exclusion, tie-exclusion, kitft-favorable tiebreaks, interior-λ
only — where it grows).

**Word-chunk control (figR5/figR6):** re-judging kitft on 10 equal-word chunks of its
full text (instead of its ~3 long lines) equalizes order resolution with v3 — tie
rate falls 10.1% → 4.4% (v3: 3.0%) and the 200-char judge clip never bites — yet its
order-only slope barely moves: **+0.165 (chunks) vs +0.105 (lines) vs v3's +0.877**
(cluster-bootstrap diff +0.72 [+0.46, +0.99], p<0.001). The baseline's flat ordering
is not a granularity artifact: it verbalizes the steered concepts without ranking
them by injected magnitude. (Its pooled slope actually drops on chunks, +0.78 —
chunking surfaces more both-present rows, shrinking the mechanical presence
component.)

Caveats (multi-agent review, 2026-07-04): the "~1.5×" pooled contrast is partly
kitft's coarser 3-line granularity (10% same-line ties vs 3%; ties-excluded ratio is
~1.3×) and its midpoint shift explains ~2/3 of the showcased λ=+2 point gap; kitft at
8:1 reaches 0.71, i.e. converging late rather than failing. `max_new_tokens=256`
censors ~69% of v3 generations mid-list (kitft 0%) — biases presence *against* v3.
The two steering dirs share a neutral-mean anchor so cos(v̂_y,v̂_s) ≠ 0 (identical for
both models; compresses the effective ratio toward 1 — λ is in nominal units; the
cosine was printed to box logs but not archived). Judge line-clip at 200 chars is
immaterial (worst-case flips move slopes by <0.11 and shrink no gap).

## Files

- `gen_ratio.py` — box-side generation (env: `AV`, `OUT_NAME`, `SHARD/NSHARDS`, `NLA_GEN_TEMP`)
- `driver_ratio.sh` — box driver (`MODEL=v3|kitft`): deps → downloads → `build_dirs_min.py` → sweep
- `judge_ratio.py` — local, OpenRouter Haiku 4.5, cached; both traits per call
- `plot_ratio.py` — figR1 (psychometric), figR2 (presence dose-response), figR3 (first index),
  printed logistic slopes
- `results/` — raw + judged JSONs and figures
