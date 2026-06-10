# Gate C0 — Generation quality review (AV explanations)

Reviewed: `gate_a/texts.json` (L20, L20_rep full-activation), `gate_b/diffav_texts.json`
(raw-diff, temp 0.7), `gate_c0/bo_texts.json` (dav8 = raw diff, resav8 = residualized
diff; 180×8 each, temp 1.0), against `gate_a/meta.json` ground truth. ~60 explanations
read closely, stats over all 3,240. Caveat used throughout: `context_tail` is only the
last ~130 chars of true context, so "not found in context_tail" overstates confabulation
somewhat — but snippet-2 quotes purport to describe the *immediately preceding* text, so
the tail is the right reference for most checks.

## Findings

### F1. Quoted "sentences" are essentially never real text
The snippet-2 quote — formatted as if quoting the document — is verbatim in the source
tail in **4/180** L20 explanations. Mean content-word overlap with the true tail is
**0.29** (median 0.27); **34/180 have zero** content-word overlap, **122/180 < 25%**.
The quotes are style artifacts: the stage-2 Claude labels quoted real text it could see;
the AV imitates the quoting register without verbatim access.

> doc 0 pos 95 (true tail: `...Marlena Death-Stares... Please. If only`):
> `The sentence structure "I wish Daisy would die or get better. Just pretend... Even if only" suggests...` — invented wholesale, presented as a quote.

### F2. Specifics churn per sample; genre/topic is the stable, grounded part
Within one position, entity sets across the 8 dav8 samples have mean Jaccard **0.21** —
names/numbers are re-rolled every sample. Snippet-1 *word* similarity within-position is
**0.248** vs **0.076** cross-position: topic+genre is position-specific signal, entities
are noise. Words stable in ≥6/8 samples are exactly topic/genre ("atheist, religious,
scientific"; "chocolate, company, marketing") plus template scaffolding — never names.
Explanations average **5.7 proper nouns**, only **17%** findable in the context tail.
Hedging is nearly absent: "likely" appears often (51/180 L20) but hedges *continuations*;
"uncertain/unclear/perhaps" ≈ 0–1% — confabulated entities are stated as fact.

> Same position, two dav8 samples: "a famous atheist declared 'I believe'... with a picture of his pet" vs "I found a Muslim scientist ranting 'I'm Allah'". (True text: an apology email about animation use.)

### F3. The template is perfectly rigid; snippet 3 is the most formulaic and ~60–75% reliable at best
**100%** of 3,240 explanations are exactly 3 paragraphs; **100%** contain "Final token";
**0/2,880** end mid-word (max_new_tokens 220 is adequate — no truncation problem).
Length shares: snippet1 23%, snippet2 38%, snippet3 40%. Snippet-3 cross-position word
Jaccard is **0.167** vs 0.07 for snippet-1 — it is mostly stock phrasing ("ends an
incomplete clause... strongly expecting..."). Its core claim (final-token identity)
matches ground truth: **L20 74%**, dav8 67%, diffav_19_21 58%, **resav8 7%**.

### F4. Final-token errors are systematic, not stochastic — ensembling doesn't fix them
Per-sample dav8 final-token accuracy is 66–68% for all 8 samples; **majority vote over 8
= 69%** (vs 67% single). Errors repeat across samples, consistent with the prior
per-sample-grounding finding. But *agreement* is still informative as a confidence
signal even though *voting* can't correct.

### F5. Diff explanations contain no change framing
Fraction containing any change language (shift/instead of/rather than/compared to/...):
content baseline L20 **14%**, dav8 16%, diffav_19_21 19%, resav8 20%. Diff vectors are
verbalized as if they were content vectors — the model has no trained register for
"describe a difference", so it narrates the diff's dominant content.

### F6. CJK is a brief mid-text code-switch, and it marks bad samples
CJK onset is at median **58%** through the text (never the opening); **57% of runs are
≤3 chars** (mean 4.4, max 31) — a lapse, not full degeneration. But it correlates with
quality: dav8 final-token accuracy is **57% in CJK-containing samples vs 68% clean**.
Rates: L20 2%, dav8 11%, resav8 22%.

> `...expecting a specific institution or信仰related concept like "God's Creation"...`

### F7. Residual-injection (resav8) output is OOD-degraded at the token level — an injection problem, not a sampling problem
resav8: 22% CJK, **8%** of samples contain garbled tokens (vs 3% dav8), final-token
accuracy **7%**. Outputs include non-words ("figurese", "seriesesequenceion",
"from_assigned_text"). Topic signal sometimes survives (an unemployment-stats position
still yields "unemployment rate" mentions), but no generation-side fix recovers this;
the residual vector's scale/direction is outside the AV's training distribution.

## Recommendations

| # | Change | Helps | Cost | Risk |
|---|--------|-------|------|------|
| 1 | n-sample + automatic filters: drop CJK/garble samples, keep positions with high cross-sample final-token agreement; flag (don't trust) snippet-2 quotes | (a) critic | low — regexes + 8 samples already exist | low |
| 2 | Two-stage rewrite: small LLM strips entities/quotes, keeps genre+structure+final-token core | (a) critic, (b) RL | 1 cheap LLM call/sample | medium — distribution shift; rewriter must be conservative |
| 3 | Lower temp (~0.3–0.5) for critic-data generation | (a) critic | free | low — diversity loss irrelevant per F4 |
| 4 | Fix stage-2 label style upstream: ban verbatim-quote formatting, require hedged specifics, add a diff-description register | (a)+(b), next AV train | retrain | medium |
| 5 | Custom `<INJECT>` prompt A/B for diff vectors ("describe how the text changes / what this vector adds") on ~20 positions | (b) RL framing | hours | medium-high — template drift; AV only saw canonical prompt |
| 6 | Don't scale residual-injection generation until the residual is renormalized/rescaled into the AV's input distribution | both | analysis first | n/a |

### Rationale

**R1 — filter, don't vote (top priority for critic data).** F6 gives a free quality
signal: a CJK/garble regex removes samples that are 11pt worse on the one verifiable
claim. F4 says majority voting can't *correct* the final token, but cross-sample
agreement is a usable per-position confidence weight: train the critic preferentially on
high-agreement positions, or at least record agreement as a feature. With 8 samples
already generated, this is pure post-processing. Quote-verification against source text
is only possible where source text is stored — worth adding `context_tail`-style fields
to generation metadata so the filter is even feasible later.

**R2 — rewrite preserves the signal F2 identified.** The grounded payload of an
explanation is: genre/format/register, topic nouns, structural claims (mid-quote,
mid-list, clause state), and the final-token claim. The confabulated payload is: proper
nouns, the fake snippet-2 quote, and the specific "expecting X or Y" phrases. A rewrite
prompt should: keep snippet-1 minus entities; convert snippet-2 from quote to paraphrase
("the preceding sentence appears to be a frustrated wish-list about plot events" — no
quote marks); keep the final-token identity and its structural role; drop the invented
continuation examples or mark them "e.g., plausibly". For critic training this removes
exactly the per-sample noise that makes two explanations of the same vector look
different (F2's 0.21 entity Jaccard); the critic stops being trained to embed fabricated
names. Risk: the critic must then be trained and evaluated on rewritten text only — a
mixed distribution is worse than either pure one.

**R3 — temperature.** Per-sample grounding is noise-free and errors are systematic
(F4), so temp 1.0 buys nothing for critic data while inflating entity churn and likely
the CJK lapse rate (L20 at 0.7 → 2% CJK; dav8 at 1.0 → 11%, though condition also
differs — worth a quick 0.3 vs 1.0 A/B on 20 positions to separate the two). For the RL
warm start keep moderate temp: the policy needs output diversity.

**R4 — the root cause is in the training labels, not decoding.** F1 is structural: the
stage-2 labeler (Claude, seeing real text) writes verbatim quotes; the AV learns the
format but cannot ground it, so every quote is a confident fabrication. Next label pass:
forbid quote marks unless reproducing ≤5 tokens; require uncertainty marking on
entities; and generate a *diff-labeled* subset (pairs of positions, label = what
changed) so the model has a change-description register (fixes F5 properly — prompt
hacks at inference only approximate it).

**R5 — prompt steering is the only near-term lever for F5, use it cautiously.** The AV
generalized zero-shot to diff vectors; small prompt edits ("the vector encodes a
difference between two texts; describe what changed") may steer framing the same way.
But the model was trained on one template, so validate on a small grid before spending
GPU-hours; expect possible degradation to CJK/garble (the resav8 lesson: OOD inputs →
token-level breakdown).

### (a) critic data vs (b) RL warm start

- **Critic (grounding is everything):** R1 + R2 + R3. Style, hedging, and diff framing
  are irrelevant; what matters is that the text contains only vector-recoverable content.
  Fabricated quotes/entities are actively harmful — the critic learns to score noise.
- **RL warm start (behavior matters):** R4 + R5. The policy's initial *register* should
  already hedge entities and describe diffs as changes; that comes from training labels
  (R4) or prompt steering (R5), not from filtering. Filtering/rewriting matter less:
  RL can fix content, but it explores from the warm-start's framing.
