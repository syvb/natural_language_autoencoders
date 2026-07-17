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
options — never the source passage. Chance = 10%. An informative explanation
lets the grader recover the continuation the target model was about to produce.

## Method

- **Corpus:** `openbmb/Ultra-FineWeb` (en) — the models' *training* corpus, so
  the eval is in-distribution — drawn from the **tail shard** (`part-2001-of-2048`),
  far from where training drew (`corpus_slice start=0`), for held-out-ness.
  (Held-out by shard position, not exact doc-dedup — the warmstart→FineWeb row
  mapping isn't public. A limitation, stated honestly.)
- **Layer 42** (the 27B extraction layer; layer 20 is the 7B pipeline).
- **N = 250** contexts. Truncation position `t ~ log-uniform[96, 600]` capped to
  leave ≥32 tokens. This **rejects the paper's [512,1536] window**: these models
  trained on a much shorter prefix distribution (median `n_raw_tokens ≈ 243`), so
  [512,1536] tests the sparse long tail, not the operating regime. Our sample's
  `t` median is **245**, matching training. `prefix_ids` are stored verbatim so
  GPU-side extraction is immune to tokenizer-version drift.
- **Distractors:** random 32-token windows from other tail-shard docs (off-domain
  by construction, as in the paper).
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
| **Matryoshka** | **99.1%** [98.3, 99.5] | 99.2% | 99.7% | 7.7% |
| **Standard** | **99.0%** [98.2, 99.5] | 99.6% | 99.4% | 5.8% |

Chance = 10%; n = 1000 gradings/arm (250×4); 0 norm-outliers; 0 grader parse
failures on the primary. Both NLAs are **near-perfect** at full length — the
explanations are so informative that the blind grader recovers the true 32-token
continuation ~99% of the time. The shuffled controls sit at/below chance, so the
~99% is real signal, not an option-length/format cue. The 100% skyline confirms
items are solvable and distractors aren't accidentally plausible.

Because full-length is at ceiling for both, it **can't differentiate** the
models. The differentiation is in *how few tokens* each needs.

## Results — token budget (front-loading)

Accuracy vs. the number of **content tokens** of the explanation the grader may
read (first N). Tokens, not lines: the matryoshka is RL-trained on random-length
truncation of **U[1,120] content tokens**, so tokens are the unit the matryoshka
property is defined in, and "first N tokens" is comparable across the two arms
(which use different line granularities). n = 500/point (250×2 rollouts).

| content tokens | 4 | 8 | 16 | 32 | 64 | 120 |
|---|---|---|---|---|---|---|
| **Matryoshka** | **86.4%** | **95.8%** | 96.4% | 97.0% | 99.0% | 99.0% |
| **Standard** | 52.2% | 67.2% | 87.2% | 93.2% | 98.0% | 99.4% |

**The matryoshka front-loads the answer.** With just **4 content tokens** it is
already at 86% (vs the standard's 52%); by **8 tokens** it is at 96% — effectively
saturated — where the standard is at 67% and needs ~32–64 tokens to catch up.
Both explanations are ~170 tokens long (median 171 mat / 163 std), but the
matryoshka packs the predictive content into the first handful of tokens while
the standard spreads it out. The two converge to ~99% by 64–120 tokens (the top
of the training truncation range).

This independently confirms the matryoshka's designed behaviour through an
**external-grader** metric — orthogonal to the co-trained critic's FVE that the
model was optimised against, and consistent with the "buried lede" FVE result
(matryoshka leads with the informative unit; the standard opens with preamble).

![suffix_eval.png](suffix_eval.png)

## Caveats

- Held-out by shard position, not exact training-doc dedup (mapping unavailable).
- Grader is Haiku 4.5; a stronger/weaker grader would shift absolutes but not the
  matryoshka-vs-standard gap, which is a within-grader comparison.
- ~12.7% of matryoshka explanations (2.6% of standard) contain a stray CJK char
  (the model's known minor injection leak); these are kept, so the numbers
  include that realistic noise.

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
