# Gate B — diff-label quality for the layer-diff NLA

**Date:** 2026-06-10 · **Status:** complete — **pass with a redesign**
**Spend:** ~$6.1 (API $4.07 via OpenRouter/Sonnet 4.6; 3 short GPU pods ~$2)
**Depends on:** `experiments/gate_a/` (all explanations, activations, and AR
reconstructions reused from there)

## Question

Can we produce *diff labels* — text describing how the activation changed
between two layers — that are distinguishable from hallucinated noise, good
enough to warm-start a diff-NLA?

## Design

Three label sources, all evaluated by AR-grounding (reconstruct the label
text with the released L20 AR, compare to the true diff `v_Y − v_X`) against
matched controls:

1. **Claude text-diff labels (v1):** give Claude the Gate A explanations of
   the same position at layers X and Y (blind to condition); it returns a
   same/different verdict plus difference snippets. Conditions: real pairs
   L19→L21 and L18→L22 (straddling L20, per Gate A §7), and a **null** —
   two independent explanations of the *same* vector (L20 vs L20-repeat),
   so any "difference" is text-sampling noise. 180 positions each.
2. **Claude v2 (delta-only prompt):** same, but the prompt forbids
   restating content shared by both descriptions ("subtract A from B,
   describe only the remainder"). L18→L22 + null.
3. **Diff-injection AV (no API):** inject the raw diff vector itself,
   rescaled to `injection_scale` like any activation, into the released
   L20 AV and take its explanation. Zero marginal label cost.

## Results

**Text-level detection (verdict rates, v1):** real pairs are flagged
"different" significantly more than null — 57.2% (18→22) and 42.8% (19→21)
vs 30.6% null (z ≈ 5 for the wider straddle). The v2 prompt inflated all
rates (90.6% vs 74.4%) and is a worse detector; detection and delta-writing
trade off.

**AR-grounding, cos(AR(label), v_Y − v_X), 18→22:**

| label source | matched | shuffled pos. | matched control | paired Δ over control | diff retrieval top-1 |
|---|---|---|---|---|---|
| Claude v1 | +0.43 | +0.24 | null texts +0.40 | **+0.028** (65% pos.) | 34% |
| Claude v2 delta-only | +0.45 | — | null texts +0.41 | **+0.046 ± 0.016** (t=5.7) | 47% |
| diff-injection AV | +0.61 | +0.21 | pos-content baseline +0.63 | −0.02 | **96%** |
| recon subtraction (Gate A §7) | +0.115 | 0.00 | exact-zero null | +0.115 | 52% |

(19→21 numbers are uniformly weaker; chance retrieval = 0.6%.)

**The structural finding that reframes everything:** the position-content
baseline — the AR reconstruction of a plain L20 explanation, containing *no
layer information whatsoever* — matches the gap-4 gold diff at cos 0.63 and
retrieves it at 99% top-1. **Most of the layer-to-layer diff is predictable
from position content alone**: what layers 18→22 write is heavily
conditioned on what the position is about. The genuinely layer-specific,
content-unpredictable residue is the small remainder, and *no* current
label source captures much of it (best: v2's +0.046 paired increment).

## Interpretation

- **Gate B's literal question: passed.** Claude detects real layer change
  at the text level (z ≈ 5), and its labels carry a statistically solid
  grounded increment over hallucinated-noise labels (+0.046 ± 0.016).
  Prompt iteration (delta-only) measurably helped (+0.028 → +0.046,
  retrieval 34% → 47%).
- **But the increment is small**, because the explanations the labels are
  diffed from describe position content, and position content already
  predicts most of the diff.
- **Diff-injection is the sleeper result.** The released AV reads raw diff
  vectors zero-shot — 100% format compliance, 0–2% CJK, and its
  explanations ground at the position-content ceiling (96% diff
  retrieval). It doesn't verbalize *change* (it describes the content
  encoded in the diff vector), but it's free, in-format, and grounded.

## Implications for the training plan

1. **The diff-AV likely needs no expensive warm start for injection
   competence** — it already reads diff vectors zero-shot. What it lacks is
   the *behavior* of describing change rather than content.
2. **Warm-start the diff-AR (critic) on self-generated pairs** —
   (diff-injection explanation, gold diff vector) — generated entirely
   without API spend.
3. **Use Claude v2 delta-only labels as a small behavioral seasoning** for
   the AV (teach the "describe what changed" framing), not as the bulk
   grounding source. Filter by verdict (verdict-different labels ground
   better: +0.036 vs +0.016 overall in v1).
4. **RL has clear headroom and a meaningful target:** a policy that only
   describes position content caps at the content-predictable ceiling
   (cos ≈ 0.6 for gap 4); reward beyond that requires verbalizing the
   unpredictable residue — which is exactly the interesting,
   layer-specific information. The reward floor is high enough for liftoff
   and the gap to 1.0 is the scientific payload.
5. **Prefer wider straddles** (gap 4 over gap 2) — better on every metric,
   again.

## Incidents

One RunPod H100-HBM3 host died with two distinct low-level CUDA errors
(fused_add_rmsnorm illegal memory access; cuBLAS GEMM execution failure) —
flaky hardware, not workload; the identical job ran clean on A100-SXM.

## Files

| file | what |
|---|---|
| `gate_b_label.py` / `gate_b_label_v2.py` | OpenRouter labeling (v1 3×180, v2 2×180 pairs) |
| `gate_b_labels.json` / `gate_b_labels_v2.json` | all 900 Claude labels with verdicts |
| `gate_b_pod.py`, `run_pod.sh` | pod job: diff-injection generation + AR scoring |
| `diffav_texts.json` | 360 explanations from directly-injected diff vectors |
| `score_diffs.py` | AR-reconstruct v1 labels (pod) |
| `diff_recons*.npy/json`, `round2_*.npy/json` | AR vectors for every label (fp16) — analysis is CPU-only |
| `analyze_grounding.py` | v1 grounding analysis |
| `grounding_results.json` | v1 metrics |
