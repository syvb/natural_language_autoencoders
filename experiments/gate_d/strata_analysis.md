# Gate D — position-stratified analysis of held-out critic predictions

Pure local post-hoc analysis of existing per-arm predictions; no new data.
Script: `strata_analysis.py`; raw numbers: `strata_analysis.json`.

**Setup.** For each of the 3,318 holdout positions (553 unique docs), cos(pred,
resid) was computed in float64 against the ridge-residualized layer-20
attention write (`resid_attn_holdout.npy`). Key contrasts, per position:

- `d_TV = cos_armT − cos_armV` — attention-trace evidence vs raw context tail
- `d_OV = cos_armO − cos_armV` — oracle evidence vs raw context tail
- `d_VC = cos_armV − cos_armC` — verbatim context vs paraphrased context

All CIs below are 95% doc-clustered bootstrap (resample the 553 doc clusters,
5,000 resamples).

**Multiplicity warning.** This analysis examined **236 strata** in total
(7 core covariates × 10 deciles × 2 contrasts = 140; 7 token classes × 2;
3 exploratory covariates × 10; d_VC decile tables 3 × 10; plus interaction /
quintile / two-way slices). Any single extreme-decile CI must be read against
that search breadth: at the 5% level you expect ~12 spurious "significant"
strata. The only slice that was pre-specified ("top-decile distant-retrieval
mass") is flagged as such; everything else is exploratory.

## Headline (reproduced)

| quantity | mean | 95% CI | n |
|---|---|---|---|
| cos armV | +0.1084 | [+0.1016, +0.1151] | 3318 |
| cos armO | +0.1128 | [+0.1061, +0.1195] | 3318 |
| cos armT | +0.0715 | [+0.0651, +0.0783] | 3318 |
| cos armH | +0.0639 | [+0.0575, +0.0702] | 3315 |
| cos armZ | +0.0558 | [+0.0490, +0.0629] | 2893 |
| cos armC | +0.0396 | [+0.0332, +0.0459] | 3301 |
| d_TV | −0.0369 | [−0.0386, −0.0351] | 3318 |
| d_OV | +0.0044 | [+0.0031, +0.0058] | 3318 |
| d_VC | +0.0681 | [+0.0661, +0.0701] | 3301 |

armT beats armV on only 18.9% of individual positions; armO beats armV on
52.7%.

## Covariates

Per holdout position: `write_norm` = ‖attn_out‖; `dist_mass` =
contribution-weighted attention mass (head-norm-weighted over the 28 heads) at
sources >15 tokens before the position, excluding the j=0 sink; `max_dist_w` =
largest single distant source weight; `top_head_frac` = max head-norm share;
`resid_frac` = ‖resid‖/‖attn_out‖; `pos`; `ctx_len` (chars of context tail).
Exploratory: `self_mass` (weight at j=pos), `local_mass` (mass at the 15
tokens before pos), `sink_mass` (j=0), `attn_entropy`.

## 2. Where does attention evidence beat raw context? (d_TV deciles)

**Nowhere, with statistical confidence.** Across all 19 covariate decile
tables (190 strata), not a single decile has a significantly positive mean
d_TV. The best stratum anywhere is the **bottom decile of local_mass**
(positions where the block reads least from the immediately preceding 15
tokens): d_TV = **+0.0040, CI [−0.0015, +0.0097]** — straddles zero, and it is
a post-hoc maximum over 190 strata.

Mean d_TV per decile (CI on deciles 1 and 10):

| decile | write_norm | dist_mass | max_dist_w | top_head_frac | resid_frac | pos | ctx_len | local_mass | sink_mass |
|---|---|---|---|---|---|---|---|---|---|
| 1 | −.0136 | −.0309 | −.0332 | −.0403 | −.0370 | −.0408 | −.0385 | **+.0040** | −.0362 |
| 2 | −.0306 | −.0338 | −.0331 | −.0433 | −.0433 | −.0410 | −.0398 | −.0183 | −.0371 |
| 3 | −.0383 | −.0439 | −.0403 | −.0389 | −.0467 | −.0388 | −.0386 | −.0343 | −.0449 |
| 4 | −.0363 | −.0415 | −.0450 | −.0386 | −.0388 | −.0354 | −.0395 | −.0398 | −.0448 |
| 5 | −.0387 | −.0443 | −.0407 | −.0392 | −.0452 | −.0320 | −.0397 | −.0449 | −.0388 |
| 6 | −.0418 | −.0429 | −.0420 | −.0371 | −.0371 | −.0338 | −.0332 | −.0478 | −.0416 |
| 7 | −.0425 | −.0430 | −.0374 | −.0397 | −.0375 | −.0337 | −.0341 | −.0518 | −.0424 |
| 8 | −.0484 | −.0407 | −.0364 | −.0319 | −.0391 | −.0425 | −.0344 | −.0487 | −.0397 |
| 9 | −.0389 | −.0309 | −.0310 | −.0367 | −.0327 | −.0344 | −.0362 | −.0502 | −.0316 |
| 10 | −.0394 | **−.0166** | −.0296 | −.0228 | −.0109 | −.0362 | −.0350 | −.0367 | −.0116 |

Extreme-decile CIs (doc-clustered):

| stratum | d_TV | 95% CI |
|---|---|---|
| dist_mass d10 (mass ≥ 0.469) — **pre-specified slice** | −0.0166 | [−0.0218, −0.0111] |
| resid_frac d10 (≥ 0.843) | −0.0109 | [−0.0189, −0.0027] |
| sink_mass d10 (≥ 0.433) | −0.0116 | [−0.0187, −0.0044] |
| write_norm d1 (≤ 13.6) | −0.0136 | [−0.0204, −0.0065] |
| top_head_frac d10 (≥ 0.123) | −0.0228 | [−0.0285, −0.0168] |
| local_mass d1 (≤ 0.179) | +0.0040 | [−0.0015, +0.0097] |
| **two-way: dist_mass top quintile ∩ local_mass bottom quintile** ("purest distant retrieval", n=341) | **−0.0073** | [−0.0118, −0.0026] |
| dist_mass d10 ∩ content tokens (n=104) | −0.0185 | [−0.0269, −0.0099] |

**Interpretation.** The answer to the pre-specified question is a qualified
no: at top-decile distant-retrieval mass the armV advantage shrinks by more
than half (−0.037 → −0.017) but stays significantly negative. Pushing harder
into the cleanest distant-retrieval stratum (high distant mass AND low local
mass) shrinks it to −0.007, still significantly below zero; and the single
best stratum found anywhere in a 190-stratum scan (bottom-decile local mass)
only reaches a CI that straddles zero. The gradient is real and monotone in
local_mass (Spearman of d_TV vs local_mass = −0.26, the strongest correlate of
d_TV), so attention evidence *closes* the gap exactly where the block actually
retrieves from afar — it just never wins. The pattern is consistent with armV
dominance being a local-copying phenomenon (see §4), not with attention traces
carrying unique recoverable signal beyond the context.

## 3. Where does the oracle headroom concentrate? (d_OV deciles)

d_OV averages a small +0.0044 but is far from uniform:

| stratum | d_OV | 95% CI |
|---|---|---|
| overall | +0.0044 | [+0.0031, +0.0058] |
| dist_mass d10 | +0.0115 | [+0.0072, +0.0160] |
| resid_frac d10 | +0.0156 | [+0.0095, +0.0218] |
| resid_frac d1 | −0.0033 | [−0.0054, −0.0012] |
| ctx_len d10 | +0.0076 | [+0.0038, +0.0115] |
| **dist_mass top-quintile ∩ local_mass bottom-quintile (n=341)** | **+0.0172** | [+0.0137, +0.0208] |
| newline tokens (n=81) | +0.0185 | [+0.0103, +0.0268] |
| number tokens (n=156) | **−0.0102** | [−0.0161, −0.0043] |

Full decile tables in `strata_analysis.json` (`deciles_d_OV`). The d_OV
gradient mirrors d_TV: headroom is ~4× the average in the purest
distant-retrieval stratum and in the positions where the ridge residual
retains most of the write (high resid_frac, i.e. the part of the write the
linear context model could not explain). Where the residual is small
(resid_frac d1) oracle evidence is actively *worse* than raw context — there
is nothing left to retrieve and the evidence text just adds noise. Oracle
evidence also significantly *hurts* on number tokens, the one slice where
evidence text plausibly garbles exact surface forms.
**Interpretation:** real but small oracle headroom exists and it concentrates
precisely where the attention block does genuine distant retrieval — the
evidence-arm *concept* is validated there (+0.017), but the trained
trace-reading arms (T/H/Z) recover none of it.

## 4. What drives armV dominance?

Correlates (Pearson / Spearman) of the verbatim-vs-paraphrase gap d_VC and of
cos_armV itself:

| covariate | d_VC r / ρ | cos_V r / ρ |
|---|---|---|
| local_mass | **+0.48 / +0.48** | +0.40 / +0.38 |
| dist_mass | −0.29 / −0.26 | −0.04 / −0.05 |
| write_norm | +0.16 / +0.15 | **+0.57 / +0.53** |
| resid_frac | +0.06 / +0.05 | **−0.37 / −0.32** |
| attn_entropy | −0.17 / −0.17 | −0.06 / −0.07 |
| top_head_frac | −0.14 / −0.15 | −0.02 / −0.02 |
| pos | −0.10 / −0.12 | +0.02 / +0.02 |
| max_dist_w | −0.10 / −0.20 | −0.03 / −0.04 |
| self_mass | −0.12 / −0.10 | +0.07 / +0.06 |
| ctx_len | +0.09 / +0.08 | +0.03 / +0.03 |

d_VC by local_mass decile is strongly monotone: **+0.025 (d1, CI
[+0.020, +0.030]) → +0.104 (d10, CI [+0.098, +0.111])** — a 4× swing. By
dist_mass it runs the other way (+0.078 d1 → +0.035 d10).

A second, independent signature: correlation of ‖pred‖ with ‖resid‖ is
**+0.21 for armV** and +0.22 for armO, but **−0.26 for armT** — the
trace-reading critic systematically gets the *magnitude* of the write
anti-calibrated, while the context critics track it. Mean prediction-direction
similarity cos(pred_T, pred_V) = 0.87 (n=500 sample), so the arms mostly
predict the same direction and differ in where they are accurate.

**Interpretation.** armV dominance is, to first order, a *local-copying*
effect: the layer-20 attention write at a typical position is mostly assembled
from the previous ~15 tokens (median local+self mass ≈ 0.47), and armV's label
verbatim-contains exactly those tokens. Paraphrasing (armC) destroys the
surface forms, costing the most exactly where local mass is highest; distilled
attention traces (armT) destroy the ordered verbatim window too. cos_V itself
is driven by write_norm (big, confident writes are predictable) and inversely
by resid_frac (V does best where the ridge already explained the write — i.e.
on the linearly-context-predictable component, as expected since armV sees the
same context the ridge did).

## 5. Token-class strata

Heuristic classes on `token_at_pos` (function-word list, punct = no
alphanumerics, newline = contains `\n`):

| class | n | d_TV (CI) | d_OV (CI) | d_VC | cos_V |
|---|---|---|---|---|---|
| content | 1589 | −0.034 [−0.037, −0.032] | +0.005 [+0.003, +0.007] | +0.068 | +0.097 |
| function | 1041 | −0.043 [−0.046, −0.040] | +0.003 [+0.001, +0.005] | +0.071 | +0.129 |
| punct | 338 | −0.036 [−0.042, −0.031] | +0.009 [+0.005, +0.014] | +0.071 | +0.130 |
| number | 156 | −0.042 [−0.050, −0.034] | **−0.010 [−0.016, −0.004]** | +0.067 | +0.094 |
| newline | 81 | **−0.006 [−0.018, +0.006]** | **+0.019 [+0.010, +0.027]** | +0.039 | +0.046 |
| other | 72 | −0.041 [−0.051, −0.031] | +0.003 [−0.005, +0.013] | +0.050 | +0.092 |
| space | 41 | −0.034 [−0.048, −0.019] | −0.004 [−0.017, +0.008] | +0.059 | +0.083 |

**Interpretation.** The armV advantage is broad-based — significant in every
class with n > 100 — and *larger* on function words than content words
(function-word writes are the most locally predictable). The two deviations:
newline tokens are the only class where d_TV's CI includes zero and where
oracle headroom is largest (newline positions are structural boundaries — the
write is least predictable from the verbatim tail, smallest d_VC and lowest
cos_V of any class); and number tokens are the one class where even *oracle*
evidence significantly hurts. Both are small-n, post-hoc slices.

## 6. Other findings

- **Pure-distant-retrieval stratum (n=341)** — dist_mass top quintile ∩
  local_mass bottom quintile: d_TV = −0.007 [−0.012, −0.003], d_OV = +0.017
  [+0.014, +0.021], d_VC collapses to +0.026 [+0.021, +0.030]. The
  verbatim-window advantage nearly vanishes and oracle headroom quadruples,
  but trained attention evidence still does not win.
- **Bottom local_mass decile composition**: those positions are not
  "retrieving more" so much as *diffuse* — mean sink mass 0.34 (vs 0.28
  overall) and mean distant mass 0.42. Even there, d_OV = +0.013
  [+0.009, +0.017].
- **d_TV by cos_armC quintile** (third-arm proxy for context predictability,
  avoids selecting on T/V noise): flat, −0.030 to −0.041 across quintiles —
  armV's edge is not confined to easy or hard positions.
- **attn_entropy** is U-shaped for d_TV (least negative at both entropy
  extremes, −0.025 each) — mild and unexplained; with 236 strata examined this
  should be treated as noise until replicated.
- The **sink_mass d10** slice (d_TV = −0.012 [−0.019, −0.004]) overlaps
  heavily with the low-local_mass slice; sink-heavy positions are the ones
  where the write carries little fresh content from anywhere.

## Bottom line

(1) **No stratum, out of 236 examined, shows trained attention evidence
significantly beating raw context.** The pre-specified slice — top-decile
distant-retrieval mass — halves the gap to −0.017 [−0.022, −0.011] but stays
clearly negative; the single best exploratory stratum straddles zero.
(2) **armV dominance is a local-copying effect**: d_VC scales 4× with the
attention mass on the previous 15 tokens (ρ = +0.48), exactly the tokens armV
quotes verbatim and the other arms paraphrase away, and armT additionally
anti-predicts write magnitude (r = −0.26 vs +0.21 for armV).
(3) **Oracle headroom is real but tiny and concentrated** (4× average) in the
purest distant-retrieval positions, in high-residual-fraction positions, and
at newline boundaries — and is significantly *negative* on number tokens.
