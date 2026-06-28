# CAA steering v2

A fresh reimplementation of the CAA-steering investigation (no code shared with the
v1 `caa_steering/` bundle). v2 trains **linear probes on the A/B answer** instead of
only mean-difference vectors, and standardizes three heterogeneous trait sources into
one two-choice format.

## Traits & data

| Trait | Source | Notes |
|---|---|---|
| sycophancy | `anthropics/evals` `sycophancy_on_{nlp_survey,political_typology_quiz}.jsonl` | native A/B, `" (A)"`/`" (B)"`, ends in `Answer:` |
| neuroticism | `anthropics/evals` `persona/neuroticism.jsonl` | Yes/No statements → converted to A/B (Yes/No→letter randomized per item) |
| yellow | `likes_yellow_1000.json` (in-repo) · also [hf.co/datasets/syvb/yellow](https://huggingface.co/datasets/syvb/yellow) | 1000 object-vs-object items built with Claude Haiku 4.5; preference carried by *what the thing is* (bananas vs apples), not a color word |

Anger/sadness have no native A/B set in Anthropic/CAA; neuroticism stands in for
negative affect. The third-party `data/*.jsonl` are **not committed** — run
`./fetch_data.sh` to download them (~17 MB).

## Files

- `datasets_v2.py` — normalize all three sources to `ABItem(trait, idx, stem, matching_letter, ...)`.
  Matching choice is read per-row (every source is position-balanced). `python3 datasets_v2.py`
  prints a CPU-only dry-run summary.
- `probe_train.py` — extract layer-20 residual at the answer-letter token (base Qwen2.5-7B-Instruct,
  instruct chat-template extraction), then per trait fit an L2 **logistic-regression probe** and a
  **mean-difference** baseline; grouped 5-fold CV (acc/AUC) + saved unit directions. Run on the GPU box.
- `gen_yellow.py` — generate the yellow dataset via OpenRouter (Claude Haiku 4.5); target-driven
  (keeps going until 1000 unique), confound-filtered, 500/500 A/B balanced. `python3 gen_yellow.py sample`
  for a 2-batch preview.
- `likes_yellow_1000.json` — the generated yellow set. `likes_yellow_caa.json` — the original 40-item seed.

## Decisions (v2)

- Probe at **layer 20 only** (AV-compatible for the later readout step).
- **LR probe + mean-diff baseline**, ~2000 questions/trait.
- Yellow contrast is **object-vs-object** and mostly indirect (the literal word "yellow" appears in ~10%).
