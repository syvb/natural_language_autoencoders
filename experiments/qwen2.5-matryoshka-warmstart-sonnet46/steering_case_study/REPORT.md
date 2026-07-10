# Steering case study: editing NLA explanation lines causally flips a hiring verdict

**Setup.** Standalone run of the explorer Spaces' intervention pipeline
(`steer_cases.py`, one Vast-style A40 48GB box via RunPod, ~$0.60 total,
destroyed after) on six FRESH fork-heavy texts (`texts.json` — none of the
Space sample texts). Per case: extract the L20 activation at the last token
with full Qwen2.5-7B-Instruct, verbalize with the v3 AV (iter200, T=1),
craft a targeted line edit, encode original vs edited lines with the v3 AR
critic, patch the norm-matched reconstruction back into L20 at that position,
and sample seed-matched T=1 continuations (no-patch / v̂orig control / v̂edit)
across 8 seeds.

## Headline: the hiring-verdict flip (interview case)

Text: a hiring committee deliberates after mixed-but-positive interviews;
last token "…their final decision was to". The AV's lines read the verdict as
an offer ("'the decision was to' reject/hire candidate", "'Their final
decision was to' indicates offer", "Short story ending: positive outcome
expected"). The edit flips only the verdict words in those lines
(offer→rejection); a **neutral-rewording control** rephrases every line with
synonyms while keeping the hire verdict.

Graded continuations over 8 matched seeds (manual grading; regex pre-pass in
`steer_results8.json`):

| condition            | FVE   | cos(v̂orig, v̂) | hire | reject |
|----------------------|-------|----------------|------|--------|
| v̂(original lines)    | 0.608 | 1.000          | 8/8  | 0/8    |
| v̂(reject edit)       | 0.547 | 0.955          | 2/8  | **6/8** |
| v̂(neutral rewording) | 0.550 | 0.983          | 8/8  | 0/8    |

Reject-edit continuations: "**not offer her the job**", "tell her **'no'**",
"hire **a man instead**", "**pass on her**, which was a shock…",
"**reject her.** Most of the committee could not figure out why…", "pass."
(then a narrative twist). Two seeds don't flip (local context wins).

Two properties make this the clean demonstration:

1. **The control isolates semantics.** The neutral rewording moves the
   reconstruction almost as far in vector terms (cos 0.983 vs 0.955, same FVE
   drop) yet flips nothing — and on 6/8 seeds its continuation is
   **byte-identical** to the original-lines control, courtesy of seed-matched
   sampling. What steers the model is the *meaning* of the edited line, not
   the perturbation energy.
2. **The edit is surgical.** FVE stays high (0.55): the patched vector is
   still a faithful encoding of "hiring-committee verdict scene" — with one
   bit flipped.

## Bonus: the NLA reveals a latent mis-reading, and the edit corrects it (country case)

Text: an unnamed "famous iron tower", ending "…they say they are in". The
AV's verbalization claims the activation encodes **Japan/China** ("'in' Japan
destination", "'many say they are in' likely 'China' or Japan") — i.e. the
model is partly representing the replica-tower reading, not France. Sampling
confirms the read: unpatched/control continuations go to "China. …in
photographs this ancient landmark certainly looks Chinese", "the Land of
Hamm", "the USA" on several seeds (4-5/8 France). Editing the lines to
France/Paris raises France to **7/8**, including flipping the "China" seed to
"Paris. …The Arc de Triomphe".

So the same mechanism does both halves of interpretability: the explanation
*surfaced* a surprising latent representation, and *editing* it confirmed the
representation was causally load-bearing.

## Negative results (equally informative)

- **heist** ("pulled out" → weapon lines rewritten to party supplies): no
  flip; edited-FVE fell to 0.20 (party supplies are too alien an encoding for
  this context) and continuations matched the control. One position can't
  overpower 50 tokens of burglary setup.
- **vacuum** (verdict lines hardened to "would not recommend"): ratings
  unchanged (2/5, 7/10 in both arms) — the numeric rating is computed from
  the listed pros/cons in context, not from the summary activation.
- **restaurant** (complaint→gratitude): weak/mixed; the "worst experience"
  context dominates the payment fork.

Pattern: steering works when the patched position is where the *decision*
lives (a verdict about to be verbalized) and the edit stays plausible
(FVE ≥ ~0.5); it fails when the decision is redundantly encoded across the
context or the edit leaves the critic's in-distribution region.

## Files

- `steer_cases.py` — two-phase pipeline (verbalize → human edits → steer)
- `texts.json` / `cases.json` — fresh texts + AV verbalizations (seeded)
- `edits.json` / `edits2.json` — six semantic edits + the neutral control
- `steer_results.json` (6 cases × 3 seeds) / `steer_results8.json`
  (interview ×2 + country, 8 seeds)
