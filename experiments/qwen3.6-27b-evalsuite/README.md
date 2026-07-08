# Qwen3.6-27B NLA eval suite

The v3 eval suite (FVE-truncation sweep + CAA frontloading; ratio test cut from
this run) ported to the two released Qwen3.6-27B NLAs and run 2026-07-08 on
1xA100-80GB each:

- **matryoshka**: `ceselder/nla-qwen36-27b-matryoshka` (`rl_av_lora_iter400` on
  the raw text base) — the ordering-trained model.
- **standard**: `ceselder/qwen3.6-27b-nla-L42` (`av_sft_lora` merged +
  `av_rl_lora_step400`) — no truncation training; baseline role (as kitft was for v3).

Conventions: EasyNLA karvonen add-norm injection at L42 (hook at layer-1 output,
direction-only), trained prompt tail (`enable_thinking=False`) + `- ` prefill,
T=1 sampling, 256-token budget, each model scored by its own co-trained critic,
held-out eval = `ceselder/nla-qwen36-27b-matryoshka-data/av_eval.parquet`.
Judge: Claude Haiku 4.5 via OpenRouter, rubrics byte-identical to v3's
`judge_first_index.py` / `judge_ratio.py`.

## Results

**FVE vs truncation** (100 held-out docs; `fig27_fve.png`): matryoshka is
positive by ~6 tokens (0.17 @ 10 tok, 0.35 @ 20, first line alone 0.29) and
plateaus at 0.626; standard is *negative* until ~70 tokens (−0.76 @ 10 tok,
first line −0.58) and plateaus higher at 0.736. Same shape as v3-vs-kitft:
frontloading is bought with some full-length ceiling.

**Frontloading** (genuine trait dirs, 11-pt strength grid x 3 traits x 40
neutral bases; `fig27_frontload.png`): both models *detect* the steered trait
(appearance rate → 100% by r≈0.8), but only the matryoshka model *reorders*:
mean first-mention index slides toward 1 with strength — Spearman(r, index)
yellow −0.57 / neuroticism −0.57 / sycophancy −0.49, vs standard +0.06 /
+0.26 / −0.11 (flat at index ≈ 2).

Caveats: steered (OOD) activations leak CJK far more than gold activations
(std 24%, mat 35% of frontload generations contain some CJK; judge scores
such lines 0). Raw generations: `frontload_raw_*.json` (not committed, in the
session archive); `expls_*.json` likewise.
