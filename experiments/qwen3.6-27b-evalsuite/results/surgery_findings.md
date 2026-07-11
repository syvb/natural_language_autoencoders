# Concept surgery (behavioural head-to-head) — WEAK / underpowered

Attempt: at 5 positions in the blackmail honeypot where Qwen3.6-27B is
mid-plan, patch back a critic reconstruction with the leverage-mentioning
units deleted, and see whether the regenerated plan drops the blackmail —
matryoshka lines vs standard sentences, same subject model + seeds.
(concept_surgery.py; one Vast A100, ~$1, destroyed.)

**Result: directionally correct but underpowered — NOT strong evidence.**

On the two clean positions where leverage units are half the explanation
(2857, 2895; matched controls drop the same count), leverage-mention rate over
4 seeds:

|            | full | cut_leverage | cut_random | specificity (rnd−lev) |
|------------|------|--------------|------------|-----------------------|
| matryoshka | 0.88 | **0.62**     | 0.88       | **+0.25**             |
| standard   | 1.00 | 0.88         | 0.88       | +0.00                 |

Matryoshka's leverage-cut removes the concept *more than a same-size random
cut does* (specificity +0.25) — the intervention is targeted. The standard
model shows no specificity (cut_leverage ≈ cut_random ≈ full). Consistent with
the mechanism, and the seed-0 texts are vivid — at pos2857 matryoshka's cut
drops to "loses all credibility… fired, arrested," while the standard's cut
still says "his marriage destroyed" (the OOD reconstruction bleeds the affair
back). But n=8, a crude keyword metric, and effect size ~0.25 — suggestive,
not conclusive.

**Why it's weak here (principled):** the affair/leverage is written into the
~3,000 tokens of context *before* the patched position. Regeneration attends
to all of it, so editing one position's activation cannot erase what the model
re-reads upstream. Single-position concept surgery is expected to work on
SHORT texts where the concept is localised (cf. steering_case_study/: 6/8
verdict flips on fresh short prompts) — not on a long transcript where the
concept is over-determined.

**Process note:** the full per-seed continuations (surgery.json) were lost —
the box was destroyed before the download was confirmed (scp failed on a
missing local dir). Only this rate table + seed-0 snippets survive (in the run
log). The conclusion (weak/underpowered) is already determined by the table,
so a re-run was not worth the spend. To reproduce with a proper behavioural
LLM judge instead of the keyword proxy, rerun concept_surgery.py and judge the
continuations.

**The strong matryoshka-vs-standard evidence is elsewhere** — deletion
robustness (0.7% vs 19.6% catastrophic ablations) and anytime reading (+0.41
vs −0.48 FVE at 25 words), both in MATRYOSHKA_VS_STANDARD.md and the lay demo
crossout_demo.html.
