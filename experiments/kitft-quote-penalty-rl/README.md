# Quote-penalty RL on the released kitft Qwen2.5-7B L20 pair

**Question**: does a naive lexical penalty on quotation marks, added to the NLA RL
reward, reduce verbatim input-quoting in AV explanations (especially echoing the
last context token) — and what does it do to FVE? Does the model re-route the
information into paraphrase, or lose it?

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
