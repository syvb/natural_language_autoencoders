# NLA Suffix-Prediction eval — Qwen3.6-27B matryoshka vs standard

Reimplementation of the **Suffix Prediction** evaluation from Anthropic's
*Natural Language Autoencoders* (transformer-circuits.pub/2026/nla), run against
the released Qwen3.6-27B NLAs:

- **matryoshka** — `ceselder/nla-qwen36-27b-matryoshka` (`rl_av_lora_iter400`)
- **standard** — `ceselder/qwen3.6-27b-nla-L42` (`av_rl_lora_step400`), for comparison

**The metric.** Given the activation verbalizer's (AV) explanation of a
layer-42 activation at the *last token* of a truncated pretraining passage, a
blind LLM grader picks the true **next 32 tokens** from **10 options** (1 true +
9 off-document distractors). The grader sees only the explanation and the
options — never the source passage. Chance = 10%.

**What it does and does not measure.** Because the 9 distractors are drawn from
*other* documents (as in the paper), the true continuation is the only on-topic
option — so the grader needs the passage's **topic/domain**, not the specific
next 32 tokens. This is a topic/domain-separability task, not verbatim next-token
recovery: a hand-written 3–6 word topic label (no next-token content) already
scores ~100% on these items. Read the accuracies below as "how well the
explanation conveys the discriminative topic," and see *Validity* for a
specificity test that would go further.

## Method

- **Corpus:** `openbmb/Ultra-FineWeb` (en) — the models' *training* corpus, so
  the eval is in-distribution — drawn from the **tail shard** (`part-2001-of-2048`),
  far from the front where the 7B datagen config drew (`corpus_slice start=0`).
  Held-out by shard position only, not exact doc-dedup: the 27B (ceselder) corpus
  config is out-of-repo, so "training drew from the front" is an *assumption*
  carried from the 7B pipeline, the trained Ultra-FineWeb subset is no longer
  hosted, and the base model's own pretraining plausibly overlaps FineWeb. Any
  residual contamination would inflate **both** arms equally, so it does not
  threaten the matryoshka-vs-standard comparison — but treat the absolute ~99%/
  skyline as "in-distribution, shard-held-out," not a clean generalization number.
- **Layer 42** (the 27B extraction layer; layer 20 is the 7B pipeline).
- **N = 250** contexts. Truncation position `t ~ log-uniform[96, 600]` capped to
  leave ≥32 tokens (median `t ≈ 244`). This **rejects the paper's [512,1536]
  window**: the 27B is meant to operate over short prefixes (the 7B pipeline's
  training prefixes were short, median a few hundred tokens — the exact 27B
  distribution isn't in-repo), so a training-realistic short window is the right
  operating regime rather than the sparse long tail. Applied identically to both
  arms. `prefix_ids` are stored verbatim so GPU-side extraction is immune to
  tokenizer-version drift.
- **Distractors:** random 32-token windows from other tail-shard docs — **off-
  document** by construction (as in the paper), which is what makes the task
  topic-separable (see *Validity*).
- **One fixed 10-way option set per context**, reused across rollouts and both
  models (it depends only on the true suffix + distractor pool, not any model),
  so the skyline validates items for both arms at once.
- **K = 4 rollouts** per activation (T=1 sampling); activations extracted once
  from the clean base and shared by both arms (bit-identical).
- **Inference protocols** pinned from the deployed Spaces: matryoshka = raw base
  + adapter, no prefill, plain bullet lines; standard = base + `av_sft` merged +
  RL adapter, `<explanation>\n` prefill, cut at `</explanation>`.
- **Grader:** Claude Haiku 4.5 via OpenRouter, paper-verbatim prompt.
- **Controls:** skyline (passage in place of explanation), shuffled-pairing
  (each explanation vs a deranged context's options, scored on the donor key),
  high-norm-outlier flag.

## Results — full explanation (saturated)

| | accuracy (95% Wilson CI) | majority-vote | inter-rollout agree | shuffled ctrl |
|---|---|---|---|---|
| **Skyline** | 100.0% [98.5, 100] | — | — | — |
| **Matryoshka** | **99.1%** [98.3, 99.5] | 98.8% | 99.5% | 6.7% |
| **Standard** | **99.0%** [98.2, 99.5] | 99.2% | 99.5% | 6.4% |

Chance = 10%; 0 norm-outliers. Both NLAs are **near-perfect** at full length —
the explanations convey the passage's topic well enough that the blind grader
picks the true continuation out of the off-topic distractors ~99% of the time.
The shuffled controls sit at/below chance, so the ~99% is real signal, not an
option-length/format cue; the 100% skyline confirms items are solvable (and, with
off-document distractors, trivially topic-separable). Because full-length is at
ceiling for both, it **can't differentiate** the models — that's in *how few
tokens* each needs.

*On the CIs:* the Wilson intervals above use n = 1000 (250 contexts × 4
rollouts), but the 4 rollouts share one activation and one option set and are
~perfectly correlated (design effect ≈ 3), so those intervals are ~1.7× too
narrow. The **honest unit is the majority-vote column (n = 250)** — e.g. the
matryoshka primary is really ≈ [96.5, 99.6], not [98.3, 99.5]. This widening
changes no conclusion (both arms are at ceiling; the budget gap below dwarfs it).

## Results — token budget (front-loading)

Accuracy vs. the number of **content tokens** of the explanation the grader may
read (first N). Tokens, not lines: the matryoshka is RL-trained on random-length
truncation of **U[1,120] content tokens**, so tokens are the unit the matryoshka
property is defined in, and "first N tokens" is comparable across the two arms
(which use different line granularities). n = 500/point (250×2 rollouts).

| content tokens | 4 | 8 | 16 | 32 | 64 | 120 |
|---|---|---|---|---|---|---|
| **Matryoshka** | **86.4%** | **95.8%** | 96.4% | 97.0% | 99.0% | 99.0% |
| **Standard** | 52.4% | 67.2% | 87.2% | 93.2% | 98.0% | 99.4% |

**The matryoshka front-loads the discriminative content.** With just **4 content
tokens** it is already at 86% (vs the standard's 52%); by **8 tokens** it is at
96% — effectively saturated — where the standard is at 67% and needs ~32–64
tokens to catch up. Both explanations are ~170 tokens long (median 171 mat / 163
std), but the matryoshka names the topic in its first few tokens while the
standard spends them on structural preamble ("Comparative educational
structure…", "Academic research paper…") before naming it. The two converge to
~99% by 64–120 tokens (the top of the training truncation range).

Given the topic-separability caveat, the precise claim is: **the matryoshka
conveys the passage's discriminative topic in far fewer tokens than the
standard** — not that it recovers the exact continuation sooner. The comparison
is fair (same token unit, identical grading, same fixed option sets), and it
independently confirms the matryoshka's designed front-loading through an
**external-grader** metric — orthogonal to the co-trained critic's FVE that the
model was optimised against, and consistent with the "buried lede" FVE result
(matryoshka leads with the informative unit; the standard opens with preamble).

![suffix_eval.png](suffix_eval.png)

## Validity

Reviewed adversarially (four independent passes over design, generation, grading,
and interpretation). The computation is sound — every number reproduces from the
cache, the Wilson formula is exact, extraction is the correct causal layer-42
convention with bit-identical activations across arms, and injection/protocols
match the deployed Spaces. Two things bound the *interpretation*:

- **Topic-separability, not next-token recovery (the main caveat).** Off-document
  distractors mean the true continuation is the only on-topic option, so the eval
  rewards conveying the passage's *domain*, not the exact 32 tokens. Direct probe:
  replacing every explanation with a bare 3–6 word topic label scores ~100%; the
  standard's register-only first 4 tokens still score 52%. So the accuracies (and
  the front-loading curve) measure **topic/register conveyance**. A **specificity
  test** would swap the off-document distractors for **same-document / same-topic
  near-misses** (other 32-token windows from the *same* doc) — this forces
  fine-grained next-token prediction and would show whether the matryoshka's
  early-token advantage survives when topic alone is not enough. Constructible
  from the existing explanations + manifest, no GPU. **Recommended next step.**
- **Rollout clustering.** See the CI note above — lead with the n=250
  majority-vote intervals.

## Caveats

- Held-out by shard position, not exact training-doc dedup (see *Corpus*).
- Grader is Haiku 4.5; a stronger/weaker grader would shift absolutes but not the
  matryoshka-vs-standard gap, which is a within-grader comparison.
- ~12.7% of matryoshka explanations (2.6% of standard) contain a stray CJK char —
  benign inline code-switching (a single on-topic Chinese word in otherwise-correct
  English), verified not to be injection failure; kept in, so the numbers include
  that realistic noise.
- The token-budget axis re-encodes the cleaned (stripped, blank-dropped) explanation
  text, so "first N tokens" is an ~faithful proxy for the model's emitted token
  boundaries, applied identically to both arms.

## Reproduce

```bash
# tokenizer (not committed): hf download Qwen/Qwen3.6-27B tokenizer* -> data/qwen_tok
python build_manifest.py                                    # local, CPU
# on a single H200 (transformers==5.5.4, peft==0.19.1, flash-linear-attention,
# torch 2.7.1+cu126): rsync repo + Space-vendored nla/, then
python suffix_gen.py --model mat --out results/explanations_mat.json
python suffix_gen.py --model std --out results/explanations_std.json
# local grading (needs ~/.openrouter_key):
python  grade_suffix.py --explanations results/explanations_{mat,std}.json
.venv-cpu/bin/python grade_budget.py --explanations results/explanations_{mat,std}.json
python  plot_suffix.py
```
