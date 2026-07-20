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

| metric (mean over 252 paired cases) | matryoshka mined item | matched standard sentence |
|---|---|---|
| marginal FVE | −0.026 | **+0.422** |
| solo FVE (item alone) | −0.126 | **+0.247** |
| LOO damage (load-bearing if > 0) | −0.003 | **+0.087** |

Even ignoring order: the matched standard sentence reconstructs well *by itself*
(+0.25) and is *load-bearing* (removing it costs +0.09 FVE; positive in 91% of
paired cases), while the matryoshka's item does neither — it reconstructs poorly
alone (−0.13; negative in 65% of paired cases) and removing it typically *helps*
(LOO damage ≈ 0). The two critics assign the same claim opposite roles. This is
the per-claim face of the cross-critic co-adaptation already seen in the
evalsuite (matryoshka explanations are critic-portable; the standard AV/critic
share a private co-adapted code). (All figures collapse each mined case to one
value, averaging over its matched standard rollouts; the 881 raw matches cover
708 distinct standard sentences, since one sentence can match several mined
items.)

## 3. Negative marginal FVE is *not* a hallucination detector (in either model)

![hallucination detector](results/fig_halluc_detector.png)

The premise that "negative marginal ≈ critic-detected hallucination" does not
hold. The naive pooled comparison looks like a weak signal:

| pooled | P(halluc \| marginal<0) | P(halluc \| marginal≥0) | odds ratio | Fisher p |
|---|---|---|---|---|
| matryoshka (top-3) | 0.43 | 0.39 | 1.18 | 0.19 (n.s.) |
| standard (all sents) | 0.58 | 0.52 | 1.24 | 0.001 |

but the one apparently-significant cell — the standard model's p=0.001 — is a
**Simpson's-paradox position artifact**, not a critic signal. Two facts collide:
(i) the standard critic scores a *lone first sentence* at ≈**−0.58 FVE, 94%
negative** (its deeply-negative-start), so **67% of all standard "negative
marginals" are simply first sentences**; and (ii) P(hallucinated) falls
monotonically with sentence position (k=0: 0.68 → k=1: 0.60 → k=2: 0.51 → k=3:
0.42) for reasons unrelated to the marginal. Pooling the two manufactures a
correlation. Stratifying by position dissolves it — and for the standard model it
**reverses**:

| standard, by position | P(halluc \| marginal<0) | P(halluc \| marginal≥0) | odds ratio | Fisher p |
|---|---|---|---|---|
| first item (k=0) | 0.68 (n=939) | 0.70 (n=61) | 0.88 | 0.78 (n.s.) |
| later items (k≥1) | 0.37 (n=464) | 0.52 (n=2723) | **0.54** | **2.6e-9 (reversed)** |

A *genuine mid-explanation* negative marginal is, if anything, **less** likely to
be a hallucination. The matryoshka shows no lift at any position (odds ratio 1.15
at k=0, 1.06 at k≥1, both n.s.), so its pooled 1.18 is also compositional —
consistent with the n.s. label. A negative marginal means the critic couldn't
reconstruct better *given the item*, which happens for redundant, vague, or
position-0 items far more often than for unfaithful ones; the per-item FVE sign
carries no usable hallucination signal in either model.

What *does* differ is the base rate, independent of the marginal: **39% of
matryoshka top-3 items** are judged hallucinated vs **54% of standard sentences**
(and of the matched standard sentences specifically, **57%**). The standard
critic is not just tolerating the occasional fabrication — it rewards (§2) a claim
pool that is hallucinated more than half the time.

### 3a. Do hallucinated items have *low* marginal FVE vs the average? (all items)

![marginal by faithfulness](results/fig_marginal_by_halluc.png)

Extending the faithfulness judge to **every** item (all 9,957 matryoshka lines —
not just top-3 — and all 4,187 standard sentences) and comparing each item's
marginal FVE to the average line-item marginal:

| | avg line-item marginal | hallucinated mean | SUPPORTED mean | Mann-Whitney (halluc<rest), pooled | position-controlled |
|---|---|---|---|---|---|
| matryoshka (lines) | +0.067 | **+0.052** | +0.108 | **p = 5e-10** | p = 0.99 (n.s.) |
| standard (sentences) | +0.175 | +0.164 | +0.355 | p = 0.41 (n.s.) | p = 1.0 (n.s.) |

**Matryoshka: yes, pooled — but it's entirely position.** Hallucinated lines
average +0.052 marginal, below the +0.067 line-item average and well below
SUPPORTED lines (+0.108), a highly significant gap. But it vanishes completely
once position is controlled (detrended p = 0.99): faithful and hallucinated lines
trace the *same* steep position curve (line 1 ≈ +0.48, decaying to ≈0 by line 5;
right panel). The apparent effect is because SUPPORTED content concentrates in the
high-marginal first line while hallucinations spread across the low-marginal tail
— not because a hallucinated line reconstructs worse *at its position*.

**Standard: no, not even pooled.** Hallucinated sentences (+0.164) sit essentially
at the average (+0.175); CONTRADICTED is actually *above* it (+0.201), and the
halluc<rest test is n.s. (p = 0.41). The genuinely low-marginal category is
**META** (−0.179) — pure genre/tone/structure commentary, which the standard
critic can barely reconstruct from — not hallucination. Ordering by mean marginal:
SUPPORTED (+0.36) > CONTRADICTED (+0.20) > FABRICATED (+0.15) ≫ META (−0.18).

So the answer to "do hallucinations have low marginal FVE" is **no** in the sense
that matters: within a position they don't, and for the standard model they aren't
low even pooled. The only robust *low-marginal* signal is meta-commentary, and the
only robust *high-marginal* signal is faithful (SUPPORTED) content — the marginal
tracks reconstruction usefulness and position, not unfaithfulness.

**Takeaway.** The requested comparison is clean and one-directional: matryoshka
items that damage its own reconstruction map to standard sentences that *improve*
the standard reconstruction, are load-bearing there (order-independent solo/LOO,
§2), and are themselves hallucinated a majority of the time. The standard
AV/critic pair reconstructs activations from surface features of a sentence
largely independent of the sentence's faithfulness. But the marginal *sign* is
not a hallucination detector for either model (§3) — the only significant-looking
cell was sentence position, not faithfulness — so this is a statement about the
two critics' reconstruction behavior on matched claims, not a lie-detector.

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

## Method audit (three independent reviewers)

The pipeline was re-audited after the first draft. Verified correct against the
raw data: the standard sentence splitter is byte-exact on all 1000 rollouts
(`''.join(units)==line`, `rebuild(all)=='\n'.join(lines)`), prefix nesting holds,
`solo[0]==pfx[0]` and `loo[n-1]==pfx[n-2]` identities pass on the shipped scores
(proving marginal/gold indexing is aligned end-to-end), the cross-model matcher
has zero indexing errors across all 881 matched rows (marginal, solo, and
sentence text all reconcile), and every headline number recomputes from scratch.
One real defect was found and **fixed here**: the earlier draft reported the
standard model's pooled P(halluc\|neg) lift (0.58 vs 0.52, p=0.001) as a "modest
real signal" — it is a Simpson's-paradox position artifact (§3, now corrected).
Guards added to `score_subsets.py` (finite-score assert, suffix-anchor cap
assert) since it carries no external FVE reference of its own.
