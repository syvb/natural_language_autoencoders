# Gate G — decoder-conditioned diff verbalization (Stage 0)

**Idea (user's):** take the released L20 NLA and reconstruct **h₂₂** from *both*
the injected earlier state **h₂₀** *and* the AV's text. Conditioning the critic
on h₂₀ moves "subtract the redundant part" into the decoder's representation
space — so the AV reads an **in-distribution state** (h₂₂), and the text only has
to carry what h₂₀ doesn't already determine (the layer-20→22 diff). This repairs
Gate E's two failure modes (it fed the AV an OOD δ vector, and froze the reader).

**Verdict: MIXED — mechanism validated, diff-signal tiny.** The conditioned-AR
setup gives a robust, significant, position-specific reconstruction gain
(text > none / cat / shuffled, all p<1e-4). But the s20 control decomposes that
gain: **~91% is state-legibility** (the AV description helps the critic *read the
injected h₂₀* — it works almost as well describing the *before* state) and only
**~9% (+0.0013 cos) is genuinely after-state/diff-specific**. Zero-shot, the
frozen AV does **not** meaningfully verbalize the layer-diff.

## Setup

- Target **h₂₂**; condition on **h₂₀** injected at a marker token (AV's
  injection_char, scaled to injection_scale=150). Critic = released L20 AR +
  LoRA(r=16, q/k/v/o) + trained value head. Direction-only MSE = 2(1−cos).
  15k FineWeb positions (gate_c), doc-level split 12030 train / 2970 holdout.
- Arms differ ONLY in the explanation text alongside the injected h₂₀:
  **none** (empty) · **cat** (raw 60-tok context) · **text** (AV of h₂₂) ·
  **textshuf** (text mismatched to wrong positions) · **s20** (AV of h₂₀).

## Pre-check (free, GPU): headroom 0.106

Best **linear** h₂₀→h₂₂ leaves 10.6% of h₂₂'s direction unrecovered (raw
cos=0.866; diff is 53% of h₂₂'s norm, only 34% aligned with h₂₀). Above the 0.08
bar → room for the text to add value. *(This headroom is mostly closed by
state-legibility, not diff content — see below.)*

## Results (held-out docs, eval cos to h₂₂)

| arm | mean cos |
|---|---|
| **text** (AV of h₂₂, after) | **0.8812** |
| s20 (AV of h₂₀, before) | 0.8798 |
| none (h₂₀ only) | 0.8673 |
| cat (raw context) | 0.8624 |
| textshuf (wrong-position text) | 0.8412 |

Doc-clustered paired bootstrap (495 docs, 10k resamples):

| contrast | Δcos | 95% CI | p | win % | reads as |
|---|---|---|---|---|---|
| text − none | +0.0138 | [+0.0131,+0.0146] | <1e-4 | 84.7% | total description value |
| text − cat | +0.0187 | [+0.0175,+0.0200] | <1e-4 | 78.7% | beats raw-context control |
| text − textshuf | +0.0400 | [+0.0390,+0.0410] | <1e-4 | 98.5% | position-specific (not a prior) |
| **text − s20** | **+0.0013** | [+0.0008,+0.0018] | <1e-4 | 54.2% | **diff-specific (after vs before)** |

Decomposition (exact): text−none (+0.0138) = s20−none (+0.0125, state-legibility)
+ text−s20 (+0.0013, diff content).

## What the controls establish

- **Beats raw context (cat).** Not "any text helps" — raw context text *hurts*
  vs h₂₀ alone (0.862 < 0.867); the critic already has context in h₂₀.
- **Position-specific (textshuf).** A *mismatched* description reconstructs h₂₂
  **below** the h₂₀-only baseline (0.841 < 0.867) — the critic genuinely *reads*
  the text, so a wrong one misleads it. +0.0400, 98.5% of positions: the
  contribution is real h₂₂ content tied to *this* position, not a generic prior.
- **Mostly state-legibility, not diff (s20).** Describing the *before* state h₂₀
  — which is already injected — still lifts reconstruction +0.0125. The AV
  functions as a **translator that makes the lossy single-token injection
  legible** to the critic. Only +0.0013 is specifically the after-state content.
- **Not an identical-text artifact.** AV(h₂₀) vs AV(h₂₂) descriptions are
  genuinely different (token-Jaccard 0.37, 0% identical; different content and
  next-token guesses). The frozen AV *does* describe the two layers differently
  — that difference is just barely reconstruction-relevant (+0.0013). So the
  finding is "frozen AV doesn't usefully verbalize the diff," not "the texts are
  the same."

## Interpretation

Two genuine findings:

1. **NLA descriptions are effective decoding aids for injected activations.**
   A natural-language description of a state makes the critic's read of that
   state's single-token injection materially better (+0.0125), even when the
   description is of a neighboring layer. This is a *reusable* positive: the
   bottleneck in reading an injected vector is partly legibility, and text fixes
   it. (It also means injection is a lossier channel than assumed.)
2. **The frozen AV barely verbalizes the layer-diff.** With the cleanest
   possible setup — in-distribution AV read, decoder-side subtraction, trained
   reader — the diff-specific, reconstruction-relevant, text-expressible signal
   for a 2-layer gap is **+0.0013 cos**. This reinforces the program's
   plumbing-dominance finding: even a substantial-norm diff (53% of ‖h₂₂‖) is
   mostly re-encoding the critic recovers from h₂₀, with a sliver of nameable
   novelty.

## Bearing on Stage 1 (RL)

The case for RL is now narrow. +0.0013 is a *frozen-AV lower bound* — the AV
describes h₂₂ holistically (mostly re-describing h₂₀-shared content), not
diff-targeted; an AV RL'd to "tell the decoder only what's new" might surface
more. But the **upside is capped**: the nonlinear none-baseline already sits at
0.867, and the total description headroom (text−none) is only +0.0138, of which
the diff-specific part is +0.0013. RL would chase a small ceiling against the
program's consistent prior that adjacent-layer diffs carry little text-expressible
novelty. A higher-EV variant before any RL spend: test a **larger layer gap**
(e.g. 20→28) — if the diff-specific component grows with gap, diff verbalization
may be a big-gap phenomenon; if it stays ~+0.001, the frozen-reader ceiling is
the story and RL is not worth it.

## Costs / artifacts

Single H100 ($3.29/hr), pre-check + gen(×2) + 5 critic trains ≈ **$13**.
Descriptions (av_out, av_h20_out), arm JSONs, per-position eval cos, and the
bootstrap published under gate_g/. wandb project `nla-layer-diff`
(runs g2_*; the first none/cat/text run was pre-key-fix offline, synced after).
