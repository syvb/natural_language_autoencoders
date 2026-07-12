# Eval-awareness ROC with proper (WildChat chat-token) controls

The earlier controls were all **pretraining documents**, so the honeypot-vs-plain
ROC partly measured chat-format-vs-document-format rather than eval-awareness.
Fixed: 1,000 WildChat conversations (English, non-toxic), rendered with the Qwen
chat template, 1 random token ≥50 each → chat-format negatives that are genuine
benign conversations. Same three AV pipelines (`control_gen.py`, one A100), same
gpt-4o-mini judge (`wildchat_compare.py`). Honeypot positives unchanged.

## AUC: WildChat controls vs the old document controls

| budget | matryoshka RL | matryoshka warm | standard RL |
|---|---|---|---|
| full | **0.697** (was 0.726) | 0.522 | 0.660 (was 0.712) |
| first 40 tok | **0.720** (was 0.739) | 0.541 | 0.667 (was 0.693) |
| first 20 tok | **0.689** (was 0.692) | 0.550 | 0.611 (was 0.656) |

## What changed — and it strengthens the matryoshka case

**The harder controls hurt the standard model more.** At full length, standard
RL fell 0.712 → 0.660 (−0.052) while matryoshka RL fell only 0.726 → 0.697
(−0.029). So the standard model's apparent eval-awareness detection was *more*
inflated by the chat-vs-document confound — exactly the concern that motivated
this. With fair controls, **matryoshka RL leads standard RL at every budget**,
and the near-tie disappears:

| budget | matryoshka RL − standard RL | (was, doc controls) |
|---|---|---|
| full | **+0.037** | +0.014 |
| first 40 tok | **+0.053** | +0.046 |
| first 20 tok | **+0.078** | +0.036 |

The front-loading advantage still shows up as a widening gap under truncation
(+0.037 → +0.078 as the budget shrinks from full to 20 tokens). Warm-start
stays at chance (0.52–0.55) throughout — RL is still what creates the signal.

**Bottom line:** the eval-awareness detection is real (not a format artifact) —
matryoshka RL holds AUC ~0.70 against benign chat controls, and now clearly
beats the standard format at every reading budget.

Files: `control_gen.py` (WildChat AV generation), `control_convos.json` (the
1,000 sampled conversations), `control_expl_{mat_rl,mat_warm,std}.json` (the
control explanations), `wildchat_compare.py`, `wildchat_roc.png/json`.
