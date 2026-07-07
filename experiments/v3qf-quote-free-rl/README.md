# v3qf — quote-free RL continuation of the v3 matryoshka NLA

**Question**: same as `../kitft-quote-penalty-rl/` but on the v3 matryoshka
pair: does a lexical penalty on quotation marks, added to the RL reward,
push verbatim input-quoting out of AV explanations — and does the content
re-route into paraphrase (FVE holds) or get lost (FVE drops)? v3's outputs
lean on quoted snippets; this tests whether the matryoshka NLA needs them.

**Setup**: continue RL from the FINAL v3 RL checkpoint
(`syvb/nla-qwen2.5-7b-L20-v3-rl` `iter_0000200/{av,ar}`, co-trained AR) with
the full v3 recipe unchanged, plus:

    reward = -mse_nrm - NLA_QUOTE_PENALTY * (# quotation-mark chars in explanation)

Implemented in `nla/reward.py` (`_quote_penalty`; valid extractions only,
failed extractions stay at the -2.0 floor). Char set is maximal — every
quotation-mark kind incl. apostrophes, backticks, curly/angle/CJK, fullwidth
and halfwidth, ornamental (see `_QUOTE_CHARS`).

## Config (first run)

| | value |
|---|---|
| start | `syvb/nla-qwen2.5-7b-L20-v3-rl` `iter_0000200/{av,ar}` (fresh RL, weights via `--hf-checkpoint`) |
| penalty | `NLA_QUOTE_PENALTY=0.1` per quote char |
| KL | 0.03 to the v3-final AV (the v3 coefficient; the ref quotes, so KL partially opposes the penalty — intentional) |
| truncation | v3's: uniform TOKENS ~U[1,120] shared per group, max_new_tokens cap, offset 0 |
| steps | 150 × 512-sample batches (64 prompts × 8), save@50, resume-extendable |
| LR | 1e-5 actor+critic |
| GPUs | one 8×H100/H200 node: actor 4 / critic 2 / rollout 2 |
| RL data | `base_av.parquet` → seed-42 doc split (bit-identical to `02_build_datasets.py`, verified vs published `av_eval`) → train half → `stage3_build --stage rl --explanation-format bullets` (the v3 prompt) |

## Readout

- wandb `octahedral-systems/nla-rl-quote-penalty`, group
  `qwen2.5-7b-L20-v3qf-quotepen0.1`: `raw_reward`, critic `fve_nrm`, KL loss;
  plus the `quote-stats` sidecar run
  (`quote_stats_wandb.py`): `quotefull/*` (exact full-batch curves from the
  reward-path JSONL — chars_mean, frac_zero, est_penalty) and `quote/*`
  (20-sample dump metrics incl. double/single family and `final_echo_frac`,
  comparable to the kitft run's).
- `NLA_QUOTE_STATS_JSONL=/workspace/out/quote_stats.jsonl` — full-batch
  per-drain quote stats from the reward path (source of truth).
- `NLA_ROLLOUT_TEXT_DUMP` snapshots (first 20 samples per reward batch),
  archived timestamped by the monitor poller.
- Post-run: round-trip FVE on `av_eval_v3` (full-length + 10/20-token
  prefixes — the matryoshka signature must survive) + quote-char counts per
  sample, vs the v3 iter_0000200 baseline.

## Preservation (everything on HF, resumable)

Private repo `syvb/nla-qwen2.5-7b-L20-v3qf-rl`:
- `hf/iter_XXXXXXX/{av,ar}` — converted, eval-ready (value-head gated).
- `raw/iter_XXXXXXX/{actor,critic}` — exact-resume dirs (final iter minimum).
- `data/rl_v3.parquet` (+ sidecar), `launch_config.*.txt`, logs, dumps,
  `quote_stats.jsonl`.

Provision ≥1 TB disk (a full checkpoint ≈ 84 GB) and **free bandwidth in BOTH
directions** (raw pushes ~84 GB).

## Files
- `01_build_rl_parquet.py` — box-side: rebuild the v3 bullets RL parquet,
  split-verified against `av_eval`.
- `bringup_box.sh` — one-shot env + models (v3 iter_0000200) + parquet.
- `run_rl_v3qf.sh` — box-side launcher (wraps `configs/rl.sh`; v3 recipe +
  quote penalty).
- `push_ckpt.sh` — push one iter to HF (`hf` / `raw` / `both`).

Env bring-up: `../qwen2.5-matryoshka-warmstart-sonnet46/setup_rl_box_lmsys.sh`
(lmsysorg/sglang:v0.5.7-cu129-amd64 image + cached flash-attn wheel from
`hf://buckets/syvb/nla-rl-build-cache/`).
