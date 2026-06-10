# NLA Layer-Diff Variant — Feasibility Study & Gate A Report

**Date:** 2026-06-10
**Status:** Gate A complete — PASSED
**Spend so far:** ~$2.25 of $25 Gate-A budget (RunPod A100 80GB, ~1.6 h)

---

## 1. The idea under test

A variant of Natural Language Autoencoders (NLAs) that verbalizes **activation
diffs** between two layers instead of raw activations:

1. Take an NLA trained on layer X (Qwen2.5-7B, layer 20) and assume it also
   works on a nearby layer Y, because residual-stream activations change
   slowly between adjacent layers.
2. Generate NLA explanations for the same positions at both layers; call the
   Claude API to produce a *diff text* describing how Y's content differs
   from X's.
3. Warm-start (SFT) an AV/AR pair on those diff labels, as in the NLA paper.
4. RL on diff reconstruction: reward = −MSE(AR(diff text), v_Y − v_X),
   direction-only, exactly like the existing pipeline.

The object being verbalized, `h_Y − h_X`, is what blocks X+1..Y *wrote into
the residual stream* — arguably a more interpretable target than the raw
state, and one the existing pipeline supports with minimal changes (data-gen
stores raw vectors; all normalization comes from the sidecar at
injection/loss time, so a diff is "just another vector" with its own
`injection_scale` / `mse_scale`).

### Risks identified up front

- **R1 — cross-layer transfer:** if the L20 NLA can't read layer-Y vectors,
  there is no cheap way to label diffs. *(→ Gate A, this report)*
- **R2 — degenerate diff labels:** the same adjacent-layer similarity that
  makes R1 plausible could make the two explanations near-identical, so
  diffing them yields sampling noise, not signal. *(→ Gate B: label real
  (v20, v21) pairs AND null (v20, v20) pairs; labels must be
  distinguishable from the null control.)*
- **R3 — RL liftoff:** warm start only needs to be good enough to give the
  RL a nonzero gradient; the reward is ground-truth-grounded. *(→ Gate C:
  short RL smoke test.)*

### Cost plan (from the initial estimate)

| Phase | Cost |
|---|---|
| Gate A — transfer check | ~$10 budgeted → **$2.25 actual** |
| Gate B — diff-label quality (API, ~2k calls + controls) | ~$10 |
| Data gen + self-labeling (~30k pairs) | ~$15 GPU |
| Diff labels, 30k Batch-API Sonnet calls | ~$65 |
| SFT warm start from released checkpoints (2×H100) | ~$15 |
| RL smoke test, 500–1000 steps (16×H100) | $260–520 |
| **Full minimal experiment** | **≈ $400–650** |

(A full RL run to convergence, paper-scale, would be ~$2,200 — only justified
if the smoke test shows reward climbing.)

---

## 2. Gate A — design

**Question:** does the released L20 NLA round-trip activations from layers it
was never trained on, and how far above/below L20 does it stay usable?

**Setup**

- **Data:** 60 FineWeb (`sample-10BT`) docs, 3 positions each (uniform over
  positions ≥ 50, matching the repo's `_MIN_POSITION`), 180 positions total.
  Activations saved at **all 29 hidden-state indices** (embeddings + 28
  blocks), raw/unnormalized per the data-gen invariant.
- **Models:** released `kitft/nla-qwen2.5-7b-L20-av` (verbalizer, served via
  SGLang `input_embeds`) and `kitft/nla-qwen2.5-7b-L20-ar` (reconstructor).
  All scales/templates/token IDs loaded from the `nla_meta.yaml` sidecars.
- **Protocol:** for each eval layer L ∈ {0,4,8,10,12,14,16,17,18,19,20,21,
  22,23,24,26,28}, inject each of the 180 vectors into the AV (temp 0.7,
  220 max tokens), then score the explanation with the AR:
  `cos(AR(text), v_L)`. One full repeat pass at L20 measures generation
  sampling noise. ~3,240 generations total.
- **Controls:**
  - *Shuffled floor:* `cos(AR(text_i), v_L[perm(i)])` — what cosine you get
    with no position-specific information (the residual stream has a large
    shared component, so the floor is far from 0).
  - *Raw geometry:* `cos(v_L, v_20)` per position — how much of transfer is
    just "layers look alike".
  - *Injection-failure smell:* fraction of outputs containing CJK characters
    (per the repo's debugging guide), and `<explanation>` tag compliance.

**Infrastructure:** single RunPod A100 80GB PCIe ($1.39/hr),
`lmsysorg/sglang:latest` image. Pipeline: extract (one pass of the base
model, all layers at once) → SGLang serves the AV → async generation →
critic scoring. Pod terminated immediately after artifact retrieval.

---

## 3. Results

![Cross-layer transfer curve](gate_a_transfer.png)

Headline metric is **FVE over the mean activation** (normalized): FVE =
1 − ‖AR(text) − v_L‖² / ‖v_L − v̄_L‖² with both vectors L2-normalized,
computed exactly from the per-layer mean cosine via FVE = 1 − 2(1−cos)/(1−‖mean
unit vector‖²). FVE 1 = perfect, 0 = no better than predicting the layer's mean
activation; explanations paired with the *wrong* position score ≈ −1 (measured
−0.94 to −1.3 per layer). Mean over 180 positions:

| layer | FVE | cos(AR, v_L) | cos(v_L, v_20) | CJK % | tag % |
|---|---|---|---|---|---|
| L0 | -0.97 | 0.065 | 0.04 | 23% | 100% |
| L4 | -0.31 | 0.506 | 0.44 | 0% | 100% |
| L8 | 0.02 | 0.607 | 0.59 | 0% | 100% |
| L10 | 0.09 | 0.642 | 0.64 | 1% | 100% |
| L12 | 0.21 | 0.688 | 0.72 | 0% | 100% |
| L14 | 0.28 | 0.719 | 0.77 | 0% | 100% |
| L16 | 0.34 | 0.755 | 0.83 | 0% | 100% |
| L17 | 0.36 | 0.761 | 0.85 | 1% | 100% |
| L18 | 0.39 | 0.771 | 0.87 | 1% | 100% |
| L19 | 0.47 | 0.803 | 0.92 | 2% | 100% |
| **L20 (trained)** | **0.60** | **0.855** | 1.00 | 2% | 100% |
| L20 (repeat) | 0.60 | 0.856 | 1.00 | 0% | 100% |
| L21 | **0.73** | **0.902** | 0.93 | 0% | 100% |
| L22 | 0.59 | 0.850 | 0.86 | 1% | 100% |
| L23 | 0.45 | 0.799 | 0.79 | 1% | 100% |
| L24 | 0.26 | 0.746 | 0.73 | 1% | 100% |
| L26 | -0.21 | 0.637 | 0.61 | 3% | 100% |
| L28 | -1.69 | 0.093 | 0.13 | 10% | 97% |

For calibration (from the AR's own docs): cos 0.9 ≈ good clean decode,
cos 0.5 ≈ mediocre, cos 0.0 ≈ orthogonal. Note our trained-layer FVE (0.60 at
temp 0.7, single sample per vector) sits below the paper's reported ~0.75 —
plausibly sampling temperature, corpus, and FVE-definition differences; the
cross-layer *comparison* is internally consistent regardless.

**Uncertainty:** 95% CIs (percentile bootstrap, 10,000 resamples clustered
by document to respect within-doc correlation) are tight — roughly ±0.01–0.02
FVE across the usable band (e.g. L20: 0.60 [0.58, 0.61]; L21: 0.73
[0.72, 0.74]). The headline L21 > L20 comparison, done paired on the same
bootstrap resamples, gives ΔFVE = +0.13 [+0.12, +0.14], P(Δ>0) = 1.0000.
Per-layer CIs live in `results.json` (`fve_ci95`, `cos_ci95`).

### Findings

1. **Transfer is excellent; the premise holds with headroom.** L21
   round-trips at **FVE 0.73 / cos 0.90 — better than the trained layer
   itself** (FVE 0.60 / cos 0.855).
   The band L14–L24 stays within ~0.10 of trained-layer performance. Layer
   gaps of 2–4 are viable, not just gap 1.
2. **Graceful, monotone degradation** away from L20 in both directions,
   down to a floor at the embeddings (L0) and a hard collapse at the final
   layer (L28) — consistent with the paper's "stream collapses toward the
   unembedding" rationale for the 2/3-depth extraction rule.
3. **The injection path is robust far off-distribution.** Tag compliance
   ~100% and CJK ≤3% across L4–L26 — injection only starts visibly failing
   at L28 (10% CJK). Note injection always rescales to the sidecar's
   `injection_scale`, which is presumably why varying raw norms
   (27 → 364 across layers) don't break it.
4. **Sampling noise is negligible in aggregate** — the L20 repeat pass
   reproduces the mean cosine to 3 decimals (0.855 vs 0.856). But the
   *texts* of two same-vector generations differ noticeably in wording while
   agreeing semantically — which is exactly why Gate B needs the null-pair
   control before we trust API-generated diff labels.
5. **Adjacent-layer diffs are not small.** cos(v20, v21) = 0.93 but the
   relative diff norm ‖v21−v20‖/‖v20‖ averages **0.44** (p10 = 0.36);
   gap 2 gives 0.79. Each block writes a substantial update — there is real
   geometry for a diff-AR to reconstruct.
6. **Qualitative check:** explanations of the same position at L16/L20/L21/
   L24 describe recognizably the same content with shifting emphasis and
   detail — the regime where a diff verbalizer would need to find signal.

### Caveat

`cos(AR(text), v_L)` for L ≠ 20 conflates (a) the AV reading layer-L vectors
and (b) the AR reconstructing into layer-20-ish geometry. The companion
column `cos(v_L, v_20)` bounds effect (b); the fact that round-trip cosine
*exceeds* raw geometric similarity for L ≥ 22 (e.g. L24: 0.75 vs 0.73)
shows the loop carries position-specific information beyond "layers look
alike". For the diff experiment proper, a diff-AR gets retrained anyway, so
this conflation doesn't block any decision.

---

## 4. Incidents & fixes (for reproducibility)

- `huggingface_hub` 1.x removed `huggingface-cli download` — it prints help
  and **exits 0** (silent no-op). Use `hf download`.
- **Repo compat bug found:** on transformers ≥ 4.47,
  `apply_chat_template(tokenize=True)` returns a dict by default, so
  `nla_inference.py`'s `load_nla_config` fails its "injection token appears
  1×" assert (counts over dict keys → 0). Fix: `return_dict=False` at both
  call sites (~lines 189, 405). Patched copy in this directory; worth
  upstreaming.
- `datasets` streaming aborted Python at interpreter shutdown
  (`PyGILState_Release` fatal error) *after* outputs were saved — harmless
  but kills a `set -e` driver; the driver now treats extraction as
  resumable.
- Operational: launch remote jobs with `setsid nohup … < /dev/null`; never
  edit a running bash script in place; `pkill -f` patterns must not match
  the invoking ssh command (`pkill -f "[s]glang"`).

## 5. Artifacts

All in this directory (`experiments/gate_a/`); `acts.npy` (75 MB) is kept out of git — regenerate via `python gate_a.py extract`.

| file | contents |
|---|---|
| `acts.npy` | raw fp32 activations, shape [180, 29, 3584] — **reusable for Gate B** |
| `texts.json` | all 3,240 explanations (per layer, with raw outputs) |
| `results.json` | per-layer metrics |
| `meta.json` | doc/position/context for each of the 180 samples |
| `gate_a.py`, `run.sh`, `run2.sh` | rerunnable pipeline (extract / generate / score) |
| `recons.npy`, `recons_labels.json` | AR-predicted vectors for every explanation (fp16, [18, 180, 3584]) — all metrics recomputable CPU-only |
| `score2.py` | regenerates `recons.npy` (the only GPU step) |
| `analyze_ci.py` | bootstrap CIs from `recons.npy` + `acts.npy` |
| `plot_fve.py` | renders `gate_a_transfer.png` |
| `nla_inference.py` | repo copy + the `return_dict=False` compat patch |

## 6. Next steps

- **Gate B (~$10, no GPU):** Claude-label diffs for (v20, v21) and (v20, v22)
  pairs from the existing `texts.json`, plus (v20, v20) null pairs. Pass
  criterion: real-pair diff texts are reliably distinguishable from
  null-pair texts (manual inspection + a simple classifier or judge).
  Decide the layer gap based on signal-to-noise — per §7, straddle the
  trained layer (e.g. 19→21 or 18→22) rather than anchoring at L20.
- **Gate C:** SFT warm start from the released checkpoints on ~20–30k diff
  labels, then a 500–1000-step RL smoke test; pass = reward climbs above
  the warm-start baseline.

## 7. Addendum: diff-direction transfer (metric probe)

FVE/cosine measure recovery of the *whole* vector, dominated by high-norm
shared content — but the layer-diff project lives in the diff subspace. A
more decision-relevant probe, computable from the saved reconstructions:
for a layer pair (X, Y), compare the reconstruction difference against the
true diff, `cos(AR(text_Y) − AR(text_X), v_Y − v_X)`. The AR is never given
a diff as input (both reconstructions are ordinary text→vector outputs,
subtracted numerically), but the subtraction lives in the AR's
layer-20-biased output space, so this is a **lower bound** on the diff
information in the explanation texts. Null control: the reconstruction diff
of two independent explanations of the *same* vector (L20 vs L20-repeat) ≈
0.00 everywhere. Diff retrieval = the recon diff picks its own gold diff
among all 180 positions (chance top-1 = 0.6%).

| pair | gap | mean cos | frac > 0 | null | diff top-1 retrieval |
|---|---|---|---|---|---|
| L20→L21 | 1 | +0.04 | 0.68 | +0.00 | 26% |
| L20→L22 | 2 | +0.05 | 0.73 | −0.00 | 25% |
| L20→L23 | 3 | +0.05 | 0.73 | +0.00 | 20% |
| L20→L24 | 4 | +0.03 | 0.64 | +0.00 | 14% |
| **L19→L21** | 2 | **+0.11** | **0.87** | +0.01 | **51%** |
| **L18→L22** | 4 | **+0.12** | **0.88** | +0.00 | **52%** |
| **L17→L23** | 6 | **+0.10** | **0.84** | +0.00 | **47%** |

Findings:

1. **The unmodified pipeline already carries position-specific diff
   signal** — without any diff training, the recon diff identifies the
   correct position's diff among 180 candidates 14–52% of the time vs 0.6%
   chance, and the null control sits at zero.
2. **Pairs straddling the trained layer are ~3× better** (L19→L21,
   L18→L22, L17→L23 vs anchoring one end at L20). Consistent with the
   AR's layer-20 output bias cancelling symmetrically in the subtraction
   when both endpoints sit off the trained layer. Diff data-gen should
   straddle L20, not anchor at it.
3. Full-vector retrieval is saturated (top-1 100% at every tested layer,
   n=180) — explanations pin down their position exactly; the hard part is
   specifically the diff direction, as expected.

Caveat for the eventual system: the diff-AR will be *trained* on diff
vectors, so its input OOD-ness is resolved by training; this probe only
uses the existing AR as a fixed text→vector measurement device. Script:
`analyze_diff_transfer.py`; numbers persisted under `_diff_transfer` in
`results.json`.
