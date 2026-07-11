# Copied-bits (overlap) RL on the released kitft Qwen2.5-7B L20 pair

The principled successor to `../kitft-quote-penalty-rl/`. That run showed a
quote-mark penalty removes quote *marks*, not verbatim *content* (copied input
bits 70.3 → 66.9 while marks went 15.25 → 0.06/sample). This one penalizes the
content itself:

    reward −= λ · Σ_spans max(0, bits(span) − B₀)

- **spans**: maximal runs of ≥2 consecutive words shared between the AV
  explanation and the sample's input context (lowercase `\w+`, punctuation-blind).
- **bits(span)** = Σ −log₂ p(word), corpus unigram counts (train docs, deduped
  per doc). Rarity-priced: copying is charged by how document-identifying it is,
  not by length. Deliberately NOT a contextual LM — predictable text copied from
  the input is still copying.
- **B₀ = 15 bits** ≈ log₂(context words × generation words): the chance-collision
  (birthday) bound. Shuffled-pairing control prices at ≈0; charged bits are
  log-likelihood evidence of copying (BLAST-style significance).
- **λ = 0.01/bit**: step-0 mean charged bits ≈ 19 (measured on pre-RL outputs)
  ⇒ mean penalty ≈ 0.19, ~2× the within-group reward spread. Monitored live
  (`jsonl/overlap_bits_mean` in wandb); raised via resume if copying doesn't move.
- One-word matches always free: naming input content is the AV's job, and the
  1-word final-token echo is the activation's most faithful readable content.

Units rationale (words > BPE tokens > chars), gaming audit, and the interactive
worked example: see the design artifact ("Price copying in bits") and the
conversation of 2026-07-07. Implementation: `nla/reward.py` `_overlap_penalty`
(`NLA_OVERLAP_PENALTY` / `_DEDUCTIBLE` / `_MIN_SPAN` / `_UNIGRAMS`).

## Run design

Identical to the quote-penalty run except the penalty: fresh RL from
`kitft/nla-qwen2.5-7b-L20-{av,ar}` (AR co-trained — it IS the reward model),
KL 0.01 to the kitft AV, no truncation, 100 × 512-sample steps (save every 25,
stop-by-eye, resume-capable), one 8×H100 node (actor 4 / critic 2 / rollout 2).

Data: `data/rl_tagged_ctx.parquet` — same seed-42 doc-split train half
(verified against the published `av_eval` row count), built with
`--keep-debug-metadata` so each row carries `detokenized_text_truncated`
(NLADataSource forwards it to `sample.metadata`; the reward scorer matches
against it). `data/unigrams.json` = the pricing table. Both prebuilt on the dev
box by `01_build_rl_parquet_ctx.py` and downloaded by the launcher — no
data-building on GPU boxes.

Baselines already measured (2026-07-07 session): kitft round-trip FVE 0.751
(300 held-out), copied-bits 19.2 mean / 14 of 16 nonzero, quote-pen iter-50
comparison numbers in `../kitft-quote-penalty-rl/eval/`.

## Files
- `01_build_rl_parquet_ctx.py` — dev-box build: split + context column +
  unigram table + upload to `syvb/nla-qwen2.5-7b-L20-rl-overlappen/data`.
- `02_offline_check.py` — pre-launch validation: scorer vs the 16-sample
  calibration set with the real unigram table; prints the λ implied by a
  target step-0 penalty.
- `run_rl_overlappen.sh` — box launcher (wraps `configs/rl.sh`).
- Push/monitor helpers reused from `../kitft-quote-penalty-rl/` with
  `HF_REPO=syvb/nla-qwen2.5-7b-L20-rl-overlappen`.
