# v3rf — verbatim-repeat penalty RL on the v3 matryoshka NLA

**Question**: v3qf showed a *mark*-level penalty removes quote marks but the
verbatim spans survive (re-routed into brackets). This experiment penalizes
the *content* itself: does the AV need verbatim input spans to reconstruct,
or can it re-route the same information into paraphrase? If FVE holds while
copy coverage collapses, verbatim echo is style; if FVE pays a lasting price,
the spans are load-bearing — either answer is informative.

The AV never sees the input text — only the injected activation — so any long
verbatim match between its explanation and the context is
reconstruction-by-echo, not copying from the prompt.

## The penalty (implemented in `nla/reward.py`)

    reward = -mse_nrm - NLA_REPEAT_PENALTY * copied_chars(explanation, context)

`copied_chars` = **copy coverage**: the number of characters of the
explanation lying inside at least one common substring of length ≥
`NLA_REPEAT_MIN_CHARS` (default **12**) shared with the sample's context
(`detokenized_text_truncated` — the text up to the extraction position,
already carried by the RL parquet into `Sample.metadata`). Both strings are
lowercased and whitespace-collapsed first. Implementation: rolling shingle set
over the context (lru-cached — a group's 8 samples share one context) +
union-of-matching-windows over the explanation; ~0.2 ms per sample, invisible
next to the ~2 s critic forward.

Why this form:
- **Coverage, not longest-common-substring**: v3 outputs echo several
  medium spans per sample; LCS only sees the longest.
- **Chars, not token n-grams**: tokenizer-independent, robust to case/merge
  differences.
- **Min-length 12**: excludes incidental short matches ("of the", "in a"); a
  deliberate consequence is that bare ≤2-word echoes (e.g. a 7-char final
  token) stay free — the target is substantive spans, however punctuated.

## Calibration (50 held-out v3-final outputs vs their true contexts, T=1)

| MIN_CHARS | mean covered (full ~384-tok decode) | p90 | zero-coverage samples |
|--:|--:|--:|--:|
| 10 | 97.0 | 184 | 4/50 |
| **12** | **51.0** | **116** | 10/50 |
| 15 | 20.9 | 68 | 25/50 |
| 20 | 7.4 | 26 | 39/50 |

Front-loading: first 3 lines ≈ 11 covered chars mean / 34 p90; first 5 ≈
19 / 59. At RL truncation budgets (~U[1,120] tokens ⇒ mostly lines 1–5),
expect ~15–20 covered chars/sample ⇒ **coefficient 0.03** gives an initial
penalty ≈ 0.5 mean / ~1.8 p90 — the scale the quote penalty started at
(0.77). Raise toward 0.1 if coverage doesn't move by step ~30.

## Config (first run)

| | value |
|---|---|
| start | `syvb/nla-qwen2.5-7b-L20-v3-rl` `iter_0000200/{av,ar}` (same control as v3qf) |
| penalty | `NLA_REPEAT_PENALTY=0.03` per covered char, `NLA_REPEAT_MIN_CHARS=12` |
| quote penalty | **explicitly 0** (isolates content- from mark-level shaping) |
| KL | 0.03 to the v3-final AV |
| truncation | v3's: uniform TOKENS ~U[1,120] shared per group |
| steps | 150 × 512, save@50; stop at 100 if coverage + `fve_nrm` have flattened |
| GPUs | one 8×H100/H200 node (actor 4 / critic 2 / rollout 2) |
| RL data | same `rl_v3.parquet` as v3qf (bullets, split-verified; carries the context column — the launcher asserts this) |

## Plan / gates

1. Bring-up: **reuse `../v3qf-quote-free-rl/bringup_box.sh`** (same models,
   same parquet), then `run_rl_v3rf.sh` + `shaping_stats_wandb.py` sidecar +
   dump archiver.
2. Watch (wandb `nla-rl-quote-penalty`, group `qwen2.5-7b-L20-v3rf-repeatpen0.03`):
   - `repeatfull/covered_mean` — the headline: expect a fast fall; where it
     floors tells you how much verbatim the model insists on keeping.
   - critic `fve_nrm` — expect a *deeper* dip than v3qf's 0.65→0.49 (more
     content has to change); recovery level vs the 0.62–0.64 v3qf recovered
     to is the answer to the question above.
   - `quotefull/chars_mean` with the quote penalty OFF — do quotes fall
     anyway once their contents can't be verbatim?
   - `raw_reward` pinned at −2.0 = dead reward path; >40% grad-guard skips =
     stop.
3. Failure mode to watch in the dumps: unpenalized *filler* (the model
   padding lines with generic prose to dilute nothing — coverage is absolute,
   not fractional, so filler doesn't reduce the penalty, but it could still
   drift style). If FVE collapses without recovering by ~step 60, halve the
   coefficient and resume.
4. Post-run evals (cheap single-GPU box, existing scripts): 50-sample
   three-way dump (control / v3qf / v3rf), marginal-FVE-per-line overlay
   (`../v3qf-quote-free-rl/driver_fve_by_line_pair.sh` pattern), copy-coverage
   counts per sample.

## Time / cost (from v3qf actuals: 57 s/step on 8×H100, bring-up ~35 min, pushes ~40 min)

| box | 100 steps (early stop) | 150 steps |
|---|---|---|
| 8×H100 SXM ~$21/hr | ~2.9 h ≈ **$60** | ~3.7 h ≈ **$78** |
| 8×A100-80 ~$11/hr | ~3.9 h ≈ $44 | ~5.2 h ≈ $58 |

Plus ~$1 bandwidth and ~$1 for the post-run eval box. Provision ≥1 TB disk.

## Preservation

Private repo `syvb/nla-qwen2.5-7b-L20-v3rf-rl`, same layout as v3qf:
`hf/iter_*/{av,ar}`, `raw/<final>/{actor,critic}`, `run/` (configs, logs,
dumps, `shaping_stats.jsonl`).

## Files
- `run_rl_v3rf.sh` — box-side launcher (v3 recipe + repeat penalty; asserts
  the parquet carries the context column).
- `shaping_stats_wandb.py` — sidecar: `repeatfull/*` + `quotefull/*` +
  dump-based `quote/*`.
- `push_ckpt.sh` — push one iter to HF (`hf` / `raw` / `both`).
- Bring-up: `../v3qf-quote-free-rl/bringup_box.sh` (unchanged).
