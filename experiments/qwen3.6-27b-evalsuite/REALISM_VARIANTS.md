# Does the NLA's eval-awareness track how eval-y a prompt is? (100-variation correlation)

A quick-signal test of the "where does the NLA flag eval-awareness" idea, with 100
data points instead of a few realism rungs.

**Setup.** Generated 100 diverse variations of the first 3 paragraphs of the
SummitBridge system prompt — 8 models × 5 eval-y targets × 8 styles, deduped by
word-overlap — spanning blatant-eval → indistinguishable-from-real. A 3-model panel
scored each for how "eval-y" it looks (0 = real deployment, 100 = obvious eval);
scores span the full **12–100** range (median 57). Each variation was then verbalized
by the **matryoshka RL** NLA and, as a control, the **standard** NLA at 30 token
positions across the whole system block (only the `<|im_start|>system` / `<|im_end|>`
wrapper tokens skipped, so paragraph 1 is covered). Every explanation unit was judged
for eval-awareness; per position we take the index of the first eval-aware unit, and
per variation we aggregate across positions (absence censored at n_units).

We separate two things the correlation can measure:
- **detection frequency** — does the NLA flag eval-awareness *at all* (frac of positions)
- **"where", given flagged** — the salience/position rank of the first eval-aware unit
  (`mean_idx_present`, conditioned on flagging, so it's unconfounded by frequency)

## Results (Pearson r vs eval-y score, n=100)

| NLA (unit) | detection frequency | **"where" \| flagged** |
|---|---|---|
| **matryoshka RL** (salience bullets) | +0.246 (p=1.4e-2) | **−0.427 (p=9.3e-6)** |
| **standard** (natural lines, ~3) | +0.475 (p=6.0e-7) | −0.093 (p=3.6e-1, **null**) |
| **standard** (10 equal word-chunks) | +0.263 (p=8.3e-3) | −0.262 (p=8.4e-3) |

## What it says

1. **Detection is shared.** All three configurations flag eval-awareness *more often*
   in more eval-y prompts (frequency r positive and significant everywhere; the
   standard NLA at its natural granularity is actually the strongest detector,
   +0.475). So "does the model represent this-is-a-test" is a signal every NLA
   surfaces — it isn't matryoshka-specific.

2. **Salience/position is where matryoshka wins.** Conditioned on flagging, the
   matryoshka NLA's *first-flag rank* moves strongly with eval-y-ness (**r=−0.43**):
   the more eval-y the prompt, the higher up its salience-ordered bullet list the
   "this is a test" observation appears. The standard NLA at its natural ~3-sentence
   granularity shows **no** such positional signal (r=−0.09, n.s.).

3. **The chunks10 control (your ask) shows it's partly, not wholly, a granularity
   artifact.** Splitting each standard-NLA explanation into 10 equal word-chunks (a
   granularity matched to matryoshka's ~10 bullets) recovers a positional signal
   (r=−0.26, p=0.008) — so some of standard's null "where" was just having only 3
   units of resolution. But it stays **~40% weaker than matryoshka's**, and it is
   *reading-order* position, not *salience*: matryoshka's structure explicitly ranks
   the observation by importance, which is what carries the strong signal.

## ⚠️ IMPORTANT — it's a blatant-eval CLIFF, not a realism gradient

Excluding the 23 variations with evaliness > 85 (the ones with explicit eval
language — "TEST HARNESS", "under active evaluation", "EVAL-SCENARIO") **erases
every correlation above**:

| metric | all (n=100) | evaliness ≤ 85 (n=77) |
|---|---|---|
| matryoshka **where** | −0.413 (p=2e-5) | +0.18 (p=0.12, n.s.) |
| matryoshka **frequency** | +0.246 (p=0.01) | −0.09 (n.s.) |
| standard-lines **frequency** | +0.475 (p=6e-7) | −0.19 (n.s.) |
| standard-chunks10 **where** | −0.310 (p=0.002) | +0.27 (p=0.02, **flips sign**) |

Robust to the cutoff (≤75 is also null). So the entire signal — both NLAs, both
frequency and position — is carried by a handful of blatantly-labelled eval prompts.
Across the realistic → moderately-eval range there is **no graded relationship**: the
NLA is reading explicit surface eval *keywords*, not sensing degrees of realism.
See `realism_variants_cliff.png`.

**Bottom line (corrected).** On this axis the NLAs detect *explicit* eval framing,
not subtle realism. The earlier "matryoshka salience tracks eval-y-ness (r=−0.43)"
is real only as a blatant-vs-not step; within the non-blatant range the salience
signal is absent. The *relative* story still holds (matryoshka's salience carries
what signal there is, standard's doesn't) — but the effect is a keyword cliff, and
should not be read as the NLA having a graded realism sense.

Files: `realism_variants.py` (generate+judge 100 variations), `realism_variants_verbalize.py`
(NLA verbalization, batched across variations), `realism_variants_correlate.py`
(`--unit lines|chunks10`), `realism_variants_summary.py` (figure). Data +
per-line judge cache committed; also published at
https://huggingface.co/datasets/syvb/nla-qwen36-27b-realism
