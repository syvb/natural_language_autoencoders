# Hallucination vs marginal FVE — matryoshka vs standard 27B NLA

**Question.** When one of a matryoshka NLA explanation's top-3 items has a
**negative marginal FVE** (adding it makes the critic's reconstruction *worse*),
(a) is that item actually unfaithful to the source text, and (b) what marginal
FVE does the *corresponding* claim get in the standard (non-matryoshka) NLA?

**Models.** `ceselder/nla-qwen36-27b-matryoshka` (`rl_av_lora_iter400`) and
`ceselder/qwen3.6-27b-nla-L42` (standard baseline), each scored by its **own**
co-trained `rl_critic_step400`. Layer 42, d_model 5120, mse_scale √5120 ≈ 71.55.

**Data.** The suffix-eval corpus (`../qwen3.6-27b-suffix-eval`): 250 clean
held-out Ultra-FineWeb-en contexts (tail shard, far outside training range),
one L42 activation each at token position t ∈ log-uniform[96, 600]. The two
models' explanations were already generated on the *same* activations (4 sampled
rollouts each). This experiment adds no generation — only critic scoring of
every explanation subset + an LLM faithfulness/matching pass.

**Unit.** Matryoshka items = its authored bullet lines (~10/expl). Standard items
= its `<explanation>` prose split into sentences (quote-parity splitter, byte-exact
reassembly — same as `std_loo_precompute.py`), ~4/expl. Marginal of item k =
FVE(items 1..k) − FVE(items 1..k−1). We also record **solo** (item scored alone)
and **LOO damage** (full − leave-one-out; > 0 means the item is load-bearing).

**Judge.** `nex-agi/nex-n2-mini` via OpenRouter. Each item is classified against
the source passage + true continuation as SUPPORTED / CONTRADICTED / FABRICATED /
META; hallucinated ≔ CONTRADICTED or FABRICATED. 0 unparsed across 8,275 calls.

## 1. Where the matryoshka critic penalizes a claim, the standard critic rewards it

![paired marginal](results/fig_paired_marginal.png)

Mining the matryoshka's top-3 items with marginal < 0 yields **272 items** (167
with marginal ≤ −0.01) across **149 of the 250 contexts**. For **252** of them the
standard model asserts the same claim in ≥1 of its 4 rollouts (matched by the
judge). Collapsing to one value per mined case (mean over matched standard
rollouts):

- **85%** of matched standard sentences have **positive** marginal FVE (mean
  **+0.42**) — the exact claims that hurt the matryoshka's reconstruction *help*
  the standard's.
- 20 mined items (7%) have **no** corresponding standard claim at all — the
  standard NLA simply never says it.

## 2. Same claim, opposite reconstruction role

![reconstruction role](results/fig_reconstruction_role.png)

The marginal gap is partly an artifact: the standard critic's cumulative FVE
climbs out of a deeply negative start (≈ −0.88 at 1 token; see the evalsuite
marginal-per-token figure), which inflates its early per-item marginals. So we
also compare the **order-independent** quantities on the 252 paired cases:

| metric | matryoshka mined item | matched standard sentence |
|---|---|---|
| marginal FVE | −0.026 | **+0.422** |
| solo FVE (item alone) | −0.126 | **+0.247** |
| LOO damage (load-bearing if > 0) | −0.003 | **+0.087** |

Even ignoring order: the matched standard sentence reconstructs well *by itself*
(+0.25) and is *load-bearing* (removing it costs +0.09 FVE, 82% of the time
positive), while the matryoshka's item does neither — it reconstructs poorly
alone (−0.13, negative 67% of the time) and removing it typically *helps*
(LOO damage ≈ 0, positive only 39%). The two critics assign the same claim
opposite roles. This is the per-claim face of the cross-critic co-adaptation
already seen in the evalsuite (matryoshka explanations are critic-portable;
the standard AV/critic share a private co-adapted code).

## 3. But negative marginal FVE is a *weak* hallucination signal

![hallucination detector](results/fig_halluc_detector.png)

The premise that "negative marginal ≈ critic-detected hallucination" only weakly
holds, and **not at all significantly for the matryoshka**:

| | P(halluc \| marginal<0) | P(halluc \| marginal≥0) | odds ratio | Fisher p |
|---|---|---|---|---|
| matryoshka (top-3) | 0.43 | 0.39 | 1.18 | **0.19 (n.s.)** |
| standard (all sents) | 0.58 | 0.52 | 1.24 | 0.001 |

Most hallucinated items — in *both* models — still carry positive marginal FVE,
and most negative-marginal items are faithful. A negative marginal means the
critic couldn't reconstruct better *given the item*, which happens for redundant,
vague, or off-base items alike; unfaithfulness is only one cause. Base hallucination
rates differ sharply though: **39% of matryoshka top-3 items** are judged
hallucinated vs **54% of standard sentences** (and of the matched standard
sentences specifically, **57%**). So the standard critic is not just tolerating
the occasional fabrication — it is systematically rewarding a claim pool that is
hallucinated more than half the time.

**Takeaway.** The requested comparison is clean and one-directional: matryoshka
items that damage its own reconstruction map to standard sentences that *improve*
the standard reconstruction, are load-bearing there, and are themselves
hallucinated a majority of the time. The standard AV/critic pair reconstructs
activations from surface features of a sentence largely independent of the
sentence's faithfulness; the matryoshka pair at least *sometimes* down-weights an
off-base late item to zero/negative marginal — though not reliably enough to use
the marginal sign as a hallucination detector.

## Case studies

`results/cases.md` — the 20 strongest negative-marginal matryoshka items (CJK-
tainted items excluded from the case list; 4/272 mined items contain CJK and are
tallied separately). Each shows the source tail, the full matryoshka explanation
with per-item marginals, the judge verdict, and every matched standard sentence
with its scores. E.g. **ci=166**: on a passage about deep-brain stimulation for
Parkinson's, the matryoshka's item 2 is "AI-generated misleading medical article
about knee surgery" (FABRICATED, marginal −0.222, solo −0.445, removing it *helps*
by 0.036) — and the standard NLA makes no such claim there.

## Files / repro

- `score_subsets.py` — GPU: raw-base L42 extraction (norms cross-checked against
  the generation-time `act_norm`, median rel-diff 7e-8) + own-critic scoring of
  every pfx/solo/loo subset. Ran on 1× RunPod H200, ~30k + ~12.6k scorings,
  fla-accelerated (~67 jobs/s), resumable. → `results/subset_scores_{mat,std}.json`
- `analyze.py` — local: marginals, mining, the nex-n2-mini faithfulness sweep
  (all mat top-3 + all std sentences) and cross-model matching, report assembly.
  Cached per call. → `results/{analysis.json, judged_faithfulness.json, cases.md}`
- `plot_figs.py` → the three figures above.
- Own-critic scoring caveat carries over from the evalsuite: cross-model FVE
  *levels* are co-adapted; the comparison here is of each claim's role *within*
  its own model's reconstruction, which is the fair unit.
