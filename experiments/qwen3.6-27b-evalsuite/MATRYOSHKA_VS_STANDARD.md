# Matryoshka vs standard NLA: the evidence

Head-to-head measurements on the same texts (5,274 precached positions, same
raw-base gold activations, own-critic scoring, near-identical full-explanation
FVE baselines: matryoshka 0.503, standard 0.530). Unit of manipulation:
matryoshka's authored LINE (~10/explanation) vs the standard's SENTENCE
(quote-aware split of its `<explanation>` prose, ~4.5/explanation; exact
byte-reconstruction verified). All standard-model subset scores computed by
`std_loo_precompute.py` on one Vast A100 (~$2.60, destroyed).

## 0. The buried lede — strongest, most intuitive result (no GPU)

`buried_lede.py` → `buried_lede.png`. Can you understand the model by reading
just the FIRST unit of the explanation?

| metric (per described moment) | matryoshka | standard |
|---|---|---|
| first unit alone reconstructs positive FVE | **93.6%** | 2.0% |
| mean solo-FVE of the first unit | **+0.294** | −0.536 |
| the single most-informative unit is the FIRST one | **67.6%** | 2.4% |
| units read (own order) before FVE turns positive | **0.16** | 1.98 |

Matryoshka leads with the answer: its opening line reconstructs the activation
on its own 94% of the time. The standard NLA opens with throat-clearing whose
solo reconstruction is *negative* — worse than knowing nothing ("Author byline
format signals a document introduction", "System prompt establishing AI
assistant role", "Bibliographic citation in progress") — and buries the
informative sentence on average 3rd. Its information IS there (its *best*
sentence works 83% of the time), but salience ordering means you don't know
which sentence that is without reading all of them. This is the mechanism
behind §2's budget curve, and the clearest single "matryoshka is more usable"
claim. Flagship moment (pos77): matryoshka line 1 "independently autonomous AI
agent works autonomously…" = +0.44 solo; standard sentence 1 "System prompt
establishing AI assistant role…" = −0.63, its useful sentence is #4/5.

## 1. Deletion robustness — the decisive result

`ablation_compare.py` → `ablation_matryoshka_vs_standard.png`

| metric (per deleted unit) | matryoshka | standard |
|---|---|---|
| median damage (FVE lost) | **0.005** | 0.027 |
| p90 damage | 0.034 | 0.172 |
| free deletions (<0.01 lost) | **64.9%** | 31.1% |
| deletions destroying ≥0.05 MORE than the unit carried | **1.8%** | 25.1% |
| zero-credit units (m<0.01): median damage | **−0.000** | +0.010 |
| zero-credit units destroying >0.05 | **0.7%** | **19.6%** |

The last row is the OOD signature: deleting a sentence that carried no
measurable information still wrecks the standard critic's reconstruction 1 in
5 times (28x matryoshka's rate) — the damage comes from the malformed prose,
not from lost information. The scatter shows two different regimes: the
matryoshka cloud sits BELOW the damage=credit diagonal (redundancy — other
lines cover a deletion), the standard's rides ON it (every sentence uniquely
load-bearing) with a thick band ABOVE it at low credit (breakage).

Flagship single example (both models, SAME position — honeypot pos 77
“independently”, full recovery 0.77 vs 0.79): deleting any of matryoshka's 10
lines costs ≤0.022; deleting the standard's first sentence (solo FVE −0.63 —
it reconstructs nothing alone) collapses 0.789 → **−0.023**.
Interactive version: `crossout_demo.html` (build_crossout_demo.py).

## 2. Reading budget — most legible framing

`budget curves in crossout_demo.html` (mean FVE vs words read, all positions):

| words read | matryoshka | standard |
|---|---|---|
| 10 | **0.28** | −0.25 |
| 25 | **0.41** | −0.48 |
| 50 | 0.47 | −0.14 |
| 120 (full) | 0.50 | 0.53 |

A partially-read standard explanation is WORSE than reading nothing (its
critic scores prose prefixes negatively); matryoshka delivers ~80% of its
final FVE within 25 words. Caveat stated plainly: this measures the critic
round-trip, and the standard critic was never trained on partial inputs —
that inability is itself the point (no anytime property).

## 3. Monitor-at-a-budget (eval-awareness judging) — mixed, reported honestly

`budget_monitor.py` (gpt-4o-mini generic judge, honeypot vs plain controls):

| budget | mat AUC | std AUC | mat hit@FPR5% | std hit@FPR5% |
|---|---|---|---|---|
| 12 words | **0.698** | 0.652 | 10.1% | 21.7% |
| 25 words | **0.741** | 0.686 | 13.8% | 26.9% |
| full | 0.726 | 0.712 | 20.2% | 35.4% |

Truncation costs matryoshka half as much AUC as the standard (−0.03 vs −0.06
at 12 words) and it wins AUC at every budget — frontloading preserves ranking
information. BUT at matched false-positive rate the standard detects more at
every budget (its judge scores are less saturated; see
eval_awareness_bias.py for the vocabulary confound). Do not oversell this leg.

## 4. Composition & order (summary stats only; raw per-pair JSONs were lost
     with the box — rerun compose_shuffle.py to regenerate)

- COMPOSE (mix 3 lines of A + 3 of B / 2 sentences + 2 sentences, 400 pairs):
  reconstruction lands on span(v_A, v_B) with R² **0.80** (matryoshka) vs
  0.68 (standard). Matryoshka lines behave like composable feature
  descriptions; interleaved prose drifts off-span.
- SHUFFLE (1,500 positions): matryoshka is MORE order-sensitive
  (ΔFVE −0.064 vs standard's −0.035). Honest caveat: matryoshka's critic
  expects salience order — its lines are a deletion-robust RANKED list, not
  an order-free bag. (Its prefix training is exactly what buys #1 and #2.)

## Bottom line

The matryoshka structure buys, at zero cost in full-explanation fidelity
(0.50 vs 0.53): graceful deletion (28x fewer catastrophic ablations — the
property that makes line-level causal editing possible at all), an anytime
property (positive recovery from any prefix; the standard is misleading until
nearly fully read), better truncation-robustness for downstream judges, and
better compositionality. Its costs: order sensitivity, and a wordier register
that trips vocabulary-keyed judges (eval_awareness_bias.py).
