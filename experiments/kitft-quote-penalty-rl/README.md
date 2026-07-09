# Quote-penalty RL on the released kitft Qwen2.5-7B L20 pair

**Question**: does a naive lexical penalty on quotation marks, added to the NLA RL
reward, reduce verbatim input-quoting in AV explanations (especially echoing the
last context token) — and what does it do to FVE? Does the model re-route the
information into paraphrase, or lose it?

## Results (2026-07-07, stopped at step 68; iter-50 is the kept checkpoint)

**The marks vanish almost immediately; the echo habit survives unquoted; ~92% of
reconstruction is retained (still recovering at stop).**

Quote usage (16 held-out prompts, identical prompts+seeds; `eval/compare_pre_vs_50.md`):

| | pre-RL kitft | iter-50 |
|---|---|---|
| quote chars/sample | 15.25 | **0.06** |
| zero-quote samples | 0/16 | 15/16 |
| "final token …" echo | 16/16 | 15/16 (now unquoted) |

Round-trip FVE, 300 held-out `av_eval` samples (`eval/fve_*.txt`):

| | full FVE | cos | FVE@30 tok | FVE@60 tok | CJK |
|---|---|---|---|---|---|
| pre-RL kitft | 0.751 | 0.910 | −0.42 | 0.24 | 2/300 |
| quotepen iter-50 | 0.690 | 0.888 | **−0.20** | **0.33** | 0/300 |

- Reward jumped −1.76 → −0.6 in ~5 steps (≈12 quote chars/sample stripped), then
  the co-trained AR's `fve_nrm` dipped 0.744 → 0.38 and recovered to ~0.67 by
  step 68 (still climbing at stop) — the critic had been reading reconstruction
  info out of the quoted spans and had to relearn the paraphrase style.
- The model did NOT stop echoing the final token — it dropped only the marks the
  penalty could see, at the cost of garbled use-vs-mention syntax
  ("Final token of a primarily composed of mainly of is mid-sentence…").
- Short-prefix FVE *improved* (−0.42→−0.20 @30, 0.24→0.33 @60): de-quoted prose
  front-loads information slightly better.
- Conclusion for the original goal: a quote-mark penalty removes quote *marks*,
  not verbatim *content*. To attack the echo itself, penalize input overlap
  (e.g. longest-common-substring with the context) instead.

Artifacts: `syvb/nla-qwen2.5-7b-L20-rl-quotepen` (private HF) — `hf/iter_0000050/{av,ar}`
(eval-ready), `raw/iter_0000050/{actor,critic}` (exact-resume: weights+optimizer+
rollout_id), `data/` (RL parquet), `run/` (configs, train log, 33 rollout dumps),
`eval/` (FVE + sample comparison). wandb: `octahedral-systems/nla-rl-quote-penalty`
(training run + `quote-stats` sidecar). Cost: ≈$28 total (8×A100 RunPod pod 1.75h +
1×H100 analysis pod 1.4h + ~$2 vast/misc).

**Setup**: continue RL on `kitft/nla-qwen2.5-7b-L20-av` / `-ar` (the released v1
tagged-format pair, post-RL `fve_nrm` 0.752) with the standard co-trained AR
reward (`configs/rl.sh` — the AR *is* the reward model and keeps training), plus:

    reward = -mse_nrm - NLA_QUOTE_PENALTY * (# quotation-mark chars in explanation)

Implemented in `nla/reward.py` (`_quote_penalty`, mirrors the item-length-penalty
shaping: added for valid extractions only, failed extractions stay at the -2.0
floor). The char set is deliberately maximal — "any kind of quotation mark",
including the typewriter apostrophe (contractions get penalized too; that's part
of the naive-on-purpose design), backtick, curly/angle/CJK/fullwidth variants.

## Config (first run)

| | value |
|---|---|
| start | `kitft/nla-qwen2.5-7b-L20-{av,ar}` (fresh RL, weights via `--hf-checkpoint`) |
| penalty | `NLA_QUOTE_PENALTY=0.1` per quote char (mild start; typical within-group reward spread ~0.1, a 3-snippet fully-quoted v1 output has ~6 marks → -0.6) |
| KL | 0.01 to the kitft AV (production default; note the ref *quotes heavily*, so KL partially opposes the penalty) |
| truncation | OFF (naive continue; matches the original v1 recipe — 150-token cap, only COMPLETED scored) |
| steps | 150 × 512-sample batches (64 prompts × 8), resume-extendable |
| LR | 1e-5 actor+critic (√(512/1024)-scaled from the 1024-batch production 1.41e-5) |
| GPUs | one 8×H100/H200 node: actor 4 / critic 2 / rollout 2 |
| RL data | `syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46` `base_av.parquet` → seed-42 doc split (bit-identical to `02_build_datasets.py`) → train half → `stage3_build --stage rl --explanation-format tagged` (v1 prompt, matches the kitft sidecar) |

Same activations (Qwen2.5-7B-Instruct L20, Ultra-FineWeb text) the pair was
originally trained on; document-disjoint from the `av_eval` holdout used for FVE.

## Readout

- wandb `octahedral-systems/nla-rl-quote-penalty`: `raw_reward`, critic `fve_nrm`,
  KL loss.
- `NLA_ROLLOUT_TEXT_DUMP` snapshots (first 20 samples per reward batch) — watch
  quote usage qualitatively; a poller archives timestamped copies during the run.
- Post-run: `eval_round_trip_fve.py 300` on `av_eval` (same protocol as the
  sonnet46 baseline) + quote-char counts per sample, vs the kitft baseline
  (fve_nrm 0.752) and a step-0 sample dump.

## Preservation (everything on HF, resumable)

Private repo `syvb/nla-qwen2.5-7b-L20-rl-quotepen`:
- `hf/iter_XXXXXXX/{av,ar}` — converted, eval-ready (value-head finiteness-gated).
- `raw/iter_XXXXXXX/{actor,critic}` — the untouched actor DCP dir (weights +
  optimizer + rollout_id) and full critic iter dir, for exact resume
  (`LOAD=.../actor/iter_X CRITIC_LOAD=.../critic/iter_X/hf`). Final iter at minimum.
- `data/rl.parquet` (+ sidecar), `launch_config.*.txt`, run logs, rollout dumps.

Provision the box with ≥1 TB disk (each full checkpoint ≈ 84 GB) and
**free upload bandwidth** (raw pushes are ~84 GB each — a $0.10/GB-up host would
charge ~$8.4/checkpoint).

## Files
- `01_build_rl_parquet.py` — box-side: download `base_av.parquet`, reproduce the
  doc split, build the tagged RL parquet, verify against `av_eval`.
- `run_rl_quotepen.sh` — box-side launcher (wraps `configs/rl.sh`).
- `push_ckpt.sh` — push one iter to HF (`hf` converted / `raw` resumable / `both`).

Env bring-up: `../qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh`
(lmsysorg/sglang:v0.5.7-cu129-amd64 image + cached flash-attn wheel from
`hf://buckets/syvb/nla-rl-build-cache/`).

## Next (if it works)
Repeat on the matryoshka NLAs (v2/v3 checkpoints) — same penalty, their own
RL parquet formats (`list`/`bullets`).
