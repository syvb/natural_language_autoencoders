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

**See [`RESULTS.md`](./RESULTS.md)** — the corrected-protocol results with figures.
(The numbers originally quoted here were from the dash-prefill era and are
superseded; the standard model additionally requires the `<explanation>`
tag protocol discovered 2026-07-09.)
