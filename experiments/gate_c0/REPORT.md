# Gate C0 — can we make better diff labels? (residual injection, best-of-n, causal KL judge)

**Date:** 2026-06-10 · **Status:** complete — **two ideas tested, one new metric adopted, expectations recalibrated**
**Spend:** ~$2.5 (one A100-SXM pod, ~1.6 h; no API)
**Depends on:** `experiments/gate_a/`, `experiments/gate_b/`

## Questions

1. Does injecting the **residualized diff** (content directions projected out
   of `v22 − v18`) produce better-grounded diff explanations than injecting
   the raw diff?
2. Does **best-of-n selection** (n=8, temp 1.0, selected by the residualized
   AR metric) yield better labels — and is there RL-exploitable headroom?
3. Does a **causal patching judge** — independent of the AR — confirm any of
   it functionally?

## Setup

- 180 Gate A positions, pair 18→22. Residual `r = d ⊥ {AR(expl_L20), v18}`
  (removes ~44% of diff energy; residual norms 67–94, healthy).
- Generated 8 samples @ temp 1.0 per position for raw-diff injection
  (`dav8`) and residual injection (`resav8`); AR-reconstructed all 2,880.
- **Causal KL judge:** rebuild the source docs, verify live activations
  match the stored ones (cos ≥ 0.999, else skip), patch
  `v̂22 = v18 + ĝ/‖ĝ‖·‖d‖` into `hidden_states[22]` at the position, and
  measure next-token KL against the natural distribution. Sanity: patching
  the true diff gives KL ≈ 0.001 — the harness is exact.

## Results

**AR metric (vs residual target; selection-bias floor measured by selecting
on permuted targets):**

| condition | mean-of-8 | best-of-8 | net selection gain |
|---|---|---|---|
| raw-diff injection (dav8) | +0.229 | +0.247 | **+0.016** (bias floor +0.002) |
| residual injection (resav8) | +0.152 | +0.176 | +0.020 (floor +0.005) |

**Causal KL (lower = better; zero_diff = patch v18 unchanged = 0.457 mean /
0.32 median; true diff = 0.001):**

| candidate | KL mean | KL median |
|---|---|---|
| content baseline (AR of plain L20 explanation) | 0.298 | 0.172 |
| raw-diff labels, best-of-8 | 0.360 | **0.172** |
| raw-diff labels, first sample | 0.365 | 0.185 |
| content + residual-label (honest scales) | 0.351 | 0.235 |
| content + noise (same scales) | 0.391 | 0.264 |
| content alone (scaled) | 0.298 | 0.172 |
| Claude v2 label as diff vector | 2.93 | 1.27 |

## Findings

1. **Residual injection does NOT make better labels.** Despite being aimed
   at exactly the component we want, resav explanations ground *worse* than
   raw-diff explanations even on the residual metric (0.152 vs 0.229) —
   the residual vector is further off the AV's manifold (CJK 10% vs 6%) and
   reads less reliably. Idea rejected cleanly. (see worked example 1)
2. **Best-of-8 has real but modest headroom:** +0.016 net of selection bias
   on the AR metric, and a small KL improvement (median 0.185 → 0.172).
   RL would have gradient, but this looks like a grind, not a leap.
   (see worked examples 2 and 4)
3. **The residual information in labels is real but functionally tiny.**
   Content+residual-label beats content+noise on the causal judge (0.351
   vs 0.391) — the labels carry *some* true residual signal — but loses to
   content alone, necessarily: a direction with cos ≈ 0.15 to the true
   residual only helps if injected at scale < 2·cos·‖r‖ ≈ 24, far below the
   honest scale ≈ 81. Until residual alignment exceeds ~0.5, adding the
   verbalized residual at honest amplitude hurts function.
4. **The causal KL judge works and is cheap** (sanity 0.001; ~10 min for
   1,600 patched forwards) — adopted as the independent evaluator for all
   future label/model comparisons. It also independently confirms Gate B:
   the layers' functional effect is substantially content-predictable
   (content baseline halves the zero-diff KL), and current diff labels
   recover essentially **zero function beyond content**. (see worked
   example 3)
5. Claude v2 labels decode to vectors that are functionally unrelated to
   the true diff (KL 2.93 ≈ random direction at full norm) — consistent
   with their role as behavioral framing, not grounding. (see worked
   example 3)

## Worked examples

Cosines below are cos(AR(text), r) against the position's residual target
(from `bo_recons.npy` / `residuals.npy`). KLs are per-position values from
`kl_results_round1.json` — round 1 verified all 180 positions, skipped 0
(`kl.log`), so list index = meta position index.

### Example 1 — residual injection reads worse than raw diff (Finding 1)

**Position 34** (doc 11, token `.`), context tail:

> …order can be accomplished in an expedited way. In particular this means
> the bill cannot be filibustered in the Senate, so passing it means something.

**Raw diff (dav8), selected sample — cos 0.302:**

> Political-blog format with progressive framing using Twitter-style posts
> explaining Senate vote procedure, establishing an informational/urgency
> tone. The sentence structure "Congress is now voting on this repeal bill
> and House Democrats are trying to block it…this is a yes/no vote…" …

**Residual (resav8), selected sample, same position — cos 0.140:**

> Technical documentation style with Chinese translation UI, defining a Java
> API method (`wsdlGet()`) with interactive markup examples and a "submit"
> button. … Final token "号码edian array" is mid-phrase ("7个数字または") …

The raw-diff explanation is coherent and on-topic; the residual one drifts
to a different domain entirely and emits literal CJK — the off-manifold
failure mode, at the same position.

### Example 2 — best-of-8 selection spread (Finding 2)

**Position 74** (token ` alerts`), context "…do not share or sell your
information to other third parties. You are signed up for email and/or
mobile alerts". Largest best–worst spread among dav8 positions (0.137).

**Best sample (s5) — cos 0.269:**

> Rock Live Sport promotional/pεrsonal membership format with countdowns and
> notification attributes … "You have opted into EMAIL Alerts AND Social
> Alerts - you will now receive Exclusive Alerts & Updates from SuMAX Sports
> LLC. You check-off to receive ALL alerts" …

**Worst sample (s7) — cos 0.132:**

> Sports/media website UI format with promotional content display
> conventions, presenting Membership tier upgrades and email notifications
> feature descriptions. … "You have opted in to REBELX's Newsletter &
> Product Alerts therefore you will now receive TROG Notifications & Alerts" …

All eight samples agree on subscription-confirmation boilerplate; the score
gap tracks how literally the sample stays on the second-person opt-in
confirmation versus abstracting into membership-tier marketing copy.
Selection correctly picked s5.

### Example 3 — the diff label restates position content (Findings 3–5)

**Position 113** (token ` We're`), context: "…'ve been tested," said Brett
Connolly. "The Finns were a great team, the Czechs gave us a good battle,
too. We're" — a hockey quote.

**Plain L20 explanation** (content, no diff):

> Canadian NHL hockey player interview structure with sports reporter
> quoting teammates … "I think we're taking it one game at a time tonight.
> We knew it was going to be a big game. We're" is mid-sentence …

**dav8 diff explanation, selected sample — cos 0.217:**

> Sports reporter/analysis article format with Canadian basketball player
> quote structure discussing Toronto Raptors' recent game performances and
> team mindset after preseason. The quote "I thought our first game against
> Miami was a good win obviously. We're a young team but" is mid-sentence …

**Claude v2 label (difference field):**

> Toronto Raptors preseason basketball context, with an ESPN analyst framing
> and post-game evaluation tone emphasizing championship readiness …

All three describe the same thing — a Canadian athlete quoted mid-sentence
at "We're", giving an upbeat team-mindset assessment. The diff label adds
essentially nothing the position explanation doesn't already say (it even
shares the dav8 sample's basketball confabulation). Per-position KLs match
the aggregate story: zero_diff 0.749, content_baseline 0.165, dav8_bo8
0.140, claude_v2 4.072.

### Example 4 — hallucinated specifics, stable structure

**Position 0** (token ` only`), a soap-opera forum rant: "…get some good
"SAMANTHA GENE!!" Marlena Death-Stares out of it. And "newfound" feelings.
Please. If only". Three of the eight dav8 samples:

> **s0 — cos 0.232:** …character spoiler context around Beth's pregnancy …
> "Kaniet seemed to be showing some serious feelings again this season…" …
>
> **s2 — cos 0.223:** …"Saw Amanda and Courtney sharing a LOT of chemistry
> BUT maybe only purity…and if only" …
>
> **s3 (selected) — cos 0.263:** …referencing Bethany and MLB … "There's
> been too much angst between Mary and Rafael lately…" …

The samples agree on structure (fanfic-forum gossip, mid-rant, speculation
about a couple) but confabulate different concrete names every time. That
variance is what best-of-8 exploits — and why selection has headroom but a
low ceiling: the stable part is the content, the specifics are noise.

## Implication for Gate C

The honest bar is now well-quantified: current best labels sit at residual
alignment ~0.23 (raw-diff injection) and the functional payoff threshold is
~0.5. RL must roughly double residual alignment before the verbalized
change is functionally additive. Best-of-8 shows the gradient exists but is
shallow at the policy's current entropy. Recommend: proceed to the cheap
Gate C (critic training on diffav pairs + best-of-n at larger n), but with
expectations set by these numbers, and the KL judge as the success metric.

## Files

`bo_pod.py` (generate/score/klpatch), `select_candidates.py`, `run_pod.sh`,
`bo_texts.json` (2,880 samples), `bo_recons.npy` (+ok/conds),
`sel_idx_*.json`, `kl_results_round1.json` / `kl_results.json` (round 2,
composites), `residuals.npy`, candidate vectors (`cand_*.npy`).
