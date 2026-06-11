# Pair-injection smoke review (50 zero-shot reads)

**Setup.** Two adjacent injected vectors into the frozen single-state AV: token 1 = v_pre
(residual entering L20), token 2 = v_post (v_pre + L20 attention write). Hypothesis: the
model might verbalize the *change* (what attention retrieved). Files reviewed:
`gend_out/pair_smoke.jsonl` (i=0–49), `meta_d.json`, `attn_evidence2.jsonl`, and two
baselines at the same positions: `gend_out/attn_read.jsonl` (raw diff vector alone, s=0)
and `gend_out/cpre.jsonl` (v_pre alone).

## 1. Coherence — 50/50 pass

All 50 are well-formed three-paragraph AV-style readings (gist → mid-context paraphrase →
final-token continuation). No garble, no degenerate loops, no CJK leakage. Minor local
stutter in a few ("Researchers affiliated affiliated", i=17; "quickrecrec", i=27) — normal
AV artifacts, not injection failure.

## 2. State vs. change — 0/50 describe a change

Every read follows the standard single-state template. A regex sweep for change language
(change/added/shift/compar/differ/retriev/...) hits 9 reads, but on inspection all are
incidental ("Weekly Manga Updates", "as cool as" comparison quoted from the context itself,
"comparative clause"). **Not one read frames its content as a delta, a comparison of two
states, or "what was just retrieved."** They read as descriptions of a single state.

## 3. Grounding (manual check on 16 positions: i=0,4,5,6,7,10,12,14,18,21,26,28,33,36,42,49)

| criterion | count |
|---|---|
| (a) names real context content (topic/genre + a real local phrase) | 16/16 |
| (b) names some actually-attended source token from evidence | ~14/16, but almost all **local** (final token + immediate phrase, where 60–90% of attention mass sits); **long-range** retrieved content named in only ~3/16 (i=4/5 header "Spoilers for the Week of…", i=49 "Big Five" from 6 lines earlier) — and those are inferable from the state |
| (c) contains at least one confabulated specific (names, brands, numbers) | ~12/16 |

Aggregate stats over all 50: content-word overlap with `context_tail` mean **0.063** (pair)
vs 0.018 (old diff read); pair beats old at 43/50 positions. The quoted "Final token" in the
read matches the actual `token_at_pos` at **42/50** positions (old diff read: **1/50**).

## 4. Pair reads vs. old single-diff reads — far more grounded, but that's the state

Old diff-vector reads at these positions were mostly off-topic confabulation (Arabic/Korean
formats, Indian politicians, math puzzles — topic roughly right in only ~5/50). Pair reads get
topic/genre right in ~48/50 and nail the final token. But this improvement is exactly what
restoring the full state buys, not evidence of change-reading.

## 5. The decisive control: pair ≈ v_pre-alone

Comparing each pair read to the **single-vector v_pre read at the same position**
(`cpre.jsonl`): content-word Jaccard **0.451** same-position vs 0.101 cross-position.
At several positions the openings are near-verbatim identical (i=49 below). The second
vector produces no visible behavioral change — the model treats the pair as one state.

Adjacent same-doc positions (i, i+1): mean Jaccard 0.193 vs 0.102 random — similar in topic
(same doc) but distinct at final-token specifics, so the reads do track position; no
copy-paste suspicion. The variation tracks the *position*, not the *change*.

## Worked examples

**i=0** (soap forum, tok " really"). Evidence: heads attend locally to "care about Sami,
really". Read: fan-forum rant tone, "I don't much care about Faith, really, really" —
genre + clause structure + final token correct; character names confabulated (Sarah/Faith
for Sami). State read; no change framing.

**i=4** (tok " for" in "|Spoilers for"). Evidence: strongest heads retrieve **long-range**
from the page header ("Week of February 11th", j=12–13). Read: "mirrors the header,
signaling a second post block… expecting 'The Week of March 12'". The one family of cases
that *looks* change-aware — but the v_pre-alone read style infers header repetition from
state alone, so not attributable to the second vector.

**i=6** (tok " wont", Harvard animation apology blog). Pair read: "quoted testimony about a
professor's apology… expecting 'do it again'". The **v_pre-alone read at i=6 says nearly
the same thing**: "quoted apology clause… expecting 'do it again'". Second vector added
nothing.

**i=33** (tok "Bow", cross-country recap). Read invents "freshman Andrew Bowens" — the real
doc has "Matt Bowman" (which two evidence heads actually attend to at j=27). Half-grounded:
right surname stem, confabulated rest. Still a state read.

**i=38** (tok " in" after "2,000 tonnes", Kraft Philadelphia-with-Cadbury launch). Read:
correct trade-news genre, correct numeric-target structure, expects "annual sales" — but
names "PepsiCo's Innocent Drinks" instead of Kraft. Real attended content ("tonnes") named;
brand confabulated.

**i=49** (tok " Big" in `(see "The Big`). Evidence: heads retrieve "Big Five illnesses" from
the prior clause. Pair read: "directly quoting the program's acronym… expecting 'Five'".
Correct — but the **v_pre-only read opens with the identical sentence** ("Public
health/educational resource document format with professional food service guidance…") and
also completes "The Big Five". The pair output is indistinguishable from a one-vector state
read.

## Verdict: **NO-GO** (for the 30k pair-generation run as designed)

- Coherence criterion: **pass** (50/50).
- Change-sensitivity criterion: **fail**. 0/50 change descriptions; outputs are ordinary
  single-state readings, near-duplicates of v_pre-alone reads at the same positions
  (Jaccard 0.45 vs 0.10 baseline). The frozen AV was trained on one injected state and
  evidently averages/ignores the extra token rather than contrasting the two.

Spending ~$5 to generate 30k more of what `cpre.jsonl` already contains has no expected
information gain. If the two-state idea is to be tested further, it needs a model change,
not more sampling — e.g., light fine-tuning with two injection slots and diff-targeted
labels, or injecting (v_pre, v_post − v_pre) so the second slot carries only the change.
One genuinely positive finding worth keeping: dual injection does not confuse the frozen
AV — reads stay coherent and become highly grounded (42/50 exact final-token hits), so the
two-slot input format itself is viable for training.
