# Hybrid diff labels — smoke test

**What:** `hybrid_label.py` has Claude (Sonnet 4.6 via OpenRouter) write a
change-framed diff label per position from *grounded* inputs: the true
source context, the L18 and L22 explanations, and all 8 independent AV
reads of the diff vector. The prompt encodes the generation-review
findings as rules: keep only claims supported by ≥2 independent pieces of
evidence, no named entities absent from the source, no quotation marks at
all, change register, no meta-language. ~$0.018/label, resumable.

**Why:** Gate B/C0 showed the two existing label sources have complementary
failures — diff-injection is vector-grounded but content-framed and
entity-confabulating; Claude-only text diffs are change-framed but
groundless (it never saw the source or the vector). The hybrid gives the
model the evidence to vote away confabulation and verify against reality.

**Smoke results (12 positions, 2 prompt iterations, $0.22):**

| check | v1 | v2 (quotes banned) |
|---|---|---|
| format parse | 12/12 | 12/12 |
| entity violations (names not in source) | 0 | **0** |
| fabricated quoted spans | 14 | **2** |

The labels also independently converge on *continuation-expectation
sharpening* as the dominant L18→L22 change — consistent with the measured
diff geometry (diff ≈ 0.86-aligned with v22; the layers mostly consolidate
the position's prediction).

## Examples

**Position 113** — source: `…said Brett Connolly. "The Finns were a great
team, the Czechs gave us a good battle, too. We're`

Raw diff reads confabulate freely (different entities each sample, CJK):

> read s0: …Canadian Basketball player **Jonnie Davis** speaking after a
> loss … on their **Chicago Bulls** rookie season …
> read s3: …young Basketball player sharing his**感受** after opening game
> against **Toronto** …

Hybrid label — entities gone, change-framed, matches the real context:

> Sharpened expectation that the incomplete closing clause will continue
> with a positive team-status phrase, such as a declaration of readiness or
> collective confidence, rather than a remark about process or
> imperfection. ¶ Strengthened framing of the token as part of an
> enumerated list of team-mindset statements within a post-game quote
> context …

**Position 0** — source: `…"SAMANTHA GENE!!" Marlena Death-Stares out of
it. And "newfound" feelings. Please. If only`

> reads s0/s3: …character spoiler context around **Beth's pregnancy** …
> "**Kaniet** seemed to…" / …referencing **Bethany and MLB** …

> Hybrid: Sharpened anticipation of a continuation clause following the
> closing incomplete conditional, with the expected completion shifting
> from a frustrated enumeration about absent story elements toward a more
> specific longing for emotional resolution or reconciliation in a
> character's arc. …

**Position 74** — source: `…You are signed up for email and/or mobile
alerts`

> reads s0/s3: …**NFL club branding** … "Your **ESPN帐户** i…" /
> …presenting **track listing** and notification preferences …

> Hybrid: Sharpened expectation that the closing token is part of a
> subscription confirmation structure, with the anticipated continuation
> shifting from a generic account-settings clause toward a more explicit
> opt-in completion phrase …

## Known limitations

- The verdict field collapses to all-"different" (like the v2 prompt) —
  irrelevant for critic training data; null calibration must come from
  null *inputs*, not verdict trust.
- Local source context is only the last ~30 tokens; full Gate C data-gen
  will carry longer context, which should remove most of the residual
  near-quote errors (some flagged "fabrications" may simply be verbatim
  from parts of the document we don't have locally).
- The labels inherit the information ceiling of their inputs (~0.235
  residual content in the diff reads, per the ensemble probe) — the hybrid
  buys fidelity and register, not new information.

## Next

Full 180 hybrid labels (~$3), then the Gate C critic experiment with three
data arms — diffav / hybrid / both — evaluated on held-out documents with
the causal-KL judge.
