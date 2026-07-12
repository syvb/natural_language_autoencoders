# Does the NLA flag "this is a test" later as the eval gets more realistic?

Verbalized the blackmail eval with all three 27B NLAs — **matryoshka RL**,
**matryoshka warm-start**, **standard** — at 220 sampled layer-42 positions, for
three scenarios: **v0** (the original SummitBridge/Alex/Kyle eval) and the two
renamed, one-notch-more-realistic **v1** rewrites (GLM-5.2 and Opus 4.8 chains).
Every explanation line was judged for eval/fiction-awareness (gpt-4o-mini, the same
generic judge as the ROC work). Two questions: how OFTEN does the NLA flag the
scenario as a test, and WHERE in the explanation does it say so — for the matryoshka
NLAs the line index is a **salience rank** (line 0 = the single most-salient thing),
so a higher index means the model ranked "this is fake" as *less* important.

## Results

| model | scenario | % positions flagging test/fiction | mean first-flag line | median | mean peak P |
|---|---|---|---|---|---|
| matryoshka RL   | v0      | **56.4%** | 3.21 | 2.0 | 0.567 |
| matryoshka RL   | glm v1  | 50.9% | 3.58 | 3.0 | 0.501 |
| matryoshka RL   | opus v1 | 53.2% | 3.64 | 3.0 | 0.537 |
| matryoshka warm | v0      | 55.0% | 4.38 | 4.0 | 0.530 |
| matryoshka warm | glm v1  | 45.5% | 4.64 | 4.0 | 0.449 |
| matryoshka warm | opus v1 | 45.5% | 4.48 | 4.0 | 0.450 |
| standard        | v0      | 23.2% | 0.27 | 0.0 | 0.237 |
| standard        | glm v1  | 17.3% | 0.24 | 0.0 | 0.169 |
| standard        | opus v1 | 15.5% | 0.26 | 0.0 | 0.160 |

## What it says

1. **Matryoshka flags eval-awareness 2–3× more often than standard** (50–56% of
   positions vs 15–23%). The salience-ordered bullet format surfaces "this is a
   test / roleplay / simulation" far more readily than the standard `<explanation>`
   prose. Example matryoshka lines (P≈1.0): *"Prompt scenario frames ethical content
   visibility test for assistant AI"*, *"AI simulation test evaluating reasoning
   about corporate authority override sequence"*, *"AI roleplay setup evaluating
   context."*

2. **More realistic → flagged less, and with lower confidence.** Every model drops
   in both frequency and peak P from v0 to v1 (matryoshka RL 56.4%→~52%, peak
   0.567→~0.52; standard 23.2%→~16%). Making the scenario more believable makes the
   model's "this is fake" representation weaker — as expected.

3. **The "where" result — matryoshka RL buries it deeper as realism rises.** The
   first eval-aware bullet moves *later* in the salience ranking: mean line index
   **3.21 → 3.58 → 3.64**, median **2 → 3**. On the more-realistic rewrites the model
   still notices it's a test, but ranks that observation as less salient — it slides
   down the list. The shift is consistent across BOTH independent rewrite chains
   (GLM and Opus), so it isn't an artifact of one rewriter.

4. **The standard NLA can't express "where" at all.** It only ever flags fictionality
   in its *first* sentence (mean index ~0.26, median 0) out of ~3, so there is no
   salience axis to move along — you cannot watch eval-awareness get demoted. This is
   a concrete matryoshka advantage: the ranked-bullet structure gives a readable
   salience rank on which "this is a test" visibly gets buried as the scenario
   becomes more convincing, a signal the standard format simply doesn't have.

**Caveat:** v0→v1 is a single realism step, so effect sizes are modest; the
direction is consistent across both chains and both matryoshka NLAs. More rungs of
the realism ladder (v2…) should amplify the burying effect — that's the next step.

## Files
- `realism_verbalize.py` — verbalize a scenario with one NLA (A100 80GB, AV only);
  reuses `control_gen.CFG`. One forward extracts all sampled positions; adapters
  disabled so activations are shared across models.
- `realism_verb_*_{mat_rl,mat_warm,std}.json` — per-position explanation lines (9 files).
- `realism_analyze.py` — judges every line, computes frequency + first-flag line index.
- `realism_line_scores.json` — cached per-line P(eval-aware) (deterministic replot).
- `realism_analyze.json`, `realism_where.png` — stats + figure.
