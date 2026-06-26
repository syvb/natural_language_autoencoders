# Qwen2.5-7B (L20) re-warm-start on Sonnet-4.6 "matryoshka" data

**Status: COMPLETE** (2026-06-26). Both checkpoints trained, evaluated, and published.

Re-does the NLA **warm-start (SFT) phase** for the Qwen2.5-7B layer-20 AV/AR pair —
starting from the *already-RLed* released checkpoints — on a new explanation dataset that
makes the models **write in a different style** (Claude Sonnet-4.6 "matryoshka" multi-snippet
predictions). This is **not** RL; it's a fresh warm-start on new targets, intended as the
starting point for a later RL phase with a modified reward.

## Artifacts

| | link |
|---|---|
| Warm-start data (vectors + Sonnet text, +eval holdout) | `syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46` (HF dataset) |
| Actor (AV) checkpoint | `syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46` (HF model) |
| Critic (AR) checkpoint | `syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46` (HF model) |
| Training curves | wandb `octahedral-systems/nla-warmstart-sonnet46` |
| Source explanations | `ceselder/nla-matryoshka-warmstart-sonnet46` (HF dataset, text-only) |
| Started from | `kitft/nla-qwen2.5-7b-L20-av` / `-ar` (released RLed pair) |

## Results

Round-trip FVE on **300 held-out `av_eval` samples** (document-level holdout, greedy decode),
`mse_scale=59.87`, raw-mean baseline (denominator) `=0.7235`:

| | FVE | dir-MSE | cos | CJK |
|---|---|---|---|---|
| **Round-trip** (orig activation → AV text → AR reconstruction) | **0.488** | 0.370 | 0.815 | 0/300 |
| Critic-only (gold Sonnet text → AR) | 0.491 | 0.368 | 0.816 | — |

- **The actor is not the bottleneck**: round-trip FVE (AV's own generated text) ≈ critic-only
  FVE (gold Sonnet text). The warm-started AV verbalizes activations about as informatively
  as Sonnet did; the ceiling here is the critic's reconstruction.
- **Injection healthy**: zero CJK across all generations (the canonical NLA smoke test).
- **New writing style transferred**: outputs are the multi-snippet "matryoshka" format, not
  the old 2–3 snippet `<explanation>` style.
- This is the **pre-RL** baseline. For reference, the released pair reached `fve_nrm` 0.752
  *after* RL (on the original style); RL from this 0.49 base should close much of that gap.

Training (final): AV loss ≈ 0.38–0.40; AR critic `fve_nrm` 0.478 (recovered from a warmup
dip — it relearns the new text→activation mapping). AV = 859 steps, AR = 857 steps, 1 epoch.

## How it works (the key insight)

The source dataset is **text-only** — Sonnet-4.6 explanations keyed by
`custom_id = {av,ar}-{doc_id}-{position}`, with an `input_text` column but **no activation
vectors**. NLA SFT needs the vectors. The original Ultra-FineWeb slice has drifted / is no
longer hosted, so regenerating from `doc_id` won't reproduce the exact docs.

But `input_text` **is** the truncated document text fed to the API, and it is bit-exactly
`token_ids[:position]` — verified: it retokenizes to exactly `position` Qwen tokens for 100%
of sampled rows. So the activation we need is **Qwen2.5-7B-Instruct's layer-20 hidden state
at the last token of `input_text`** — fully reproducible from the dataset's own text, no
corpus required. (A preflight against current `openbmb/Ultra-FineWeb` confirmed the corpus
had drifted: 0% text match — hence this approach.)

We use only the `av-*` rows for the actor and the `ar-*` rows for the critic (the original
document-level split), with `~5k` whole-document samples held out per split for eval.

## Reproduction

Run on a single H200/H100 (the SFT step needs ~125 GB peak for the 28-layer actor at
micro-batch 16; the H100-80GB works for the eval but use an H200 for training). All paths
assume `/workspace`. `/root/.hf_token` and `/root/.wandb_key` must be present.

```bash
# 0. provision a GPU box (image: pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel), rsync this
#    repo to /workspace/nla, then:
bash experiments/qwen2.5-matryoshka-warmstart-sonnet46/setup_box.sh   # deps + miles + env fixes + downloads

cd /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46
python 01_extract_activations.py     # matryoshka text -> Qwen L20 activations -> base_{av,ar}.parquet  (~1h GPU)
python 02_build_datasets.py          # doc-level train/eval split + stage3_build -> av_sft/av_eval/ar_sft/ar_eval
python 03_upload_datasets.py         # publish the dataset (optional)

cp run_*.sh /workspace/ && cd /workspace
bash run_both.sh                     # AV then AR warm-start SFT, with wandb (~2h + ~2h on 1xH200)

cd /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46
python 04_convert_upload_checkpoints.py both   # actor DCP->HF + publish both checkpoints

# eval (round-trip FVE). download the two checkpoints to /workspace/av, /workspace/ar and
# av_eval.parquet to /workspace/av_eval.parquet first; see ENV_FIXES §6 re torchvision/av:
cd /root && PYTHONPATH=/workspace/nla python /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/eval_round_trip_fve.py 300
```

See **ENV_FIXES.md** for the non-obvious environment patches (SGLang SFT stubs, torch-2.5.1
FSDP2 import, Ray-worker `.pth`, the `av`-dir-vs-PyAV eval gotcha). `setup_box.sh` applies
them automatically.

## Hyperparameters (Qwen2.5 recipe, unchanged from the released run)

| | value |
|---|---|
| global batch | 256 (rollout 256 × 1 sample/prompt) |
| micro-batch | 16 |
| lr | 2e-5 → 2e-6 cosine, warmup 50 |
| epochs | 1 |
| injection_scale (AV) | 150 |
| mse_scale | 59.87 (√d_model) |
| attn | FA2 (actor) / sdpa (critic) |
| layer | 20, d_model 3584 |

## Files
- `01_extract_activations.py` — text → Qwen L20 last-token activations → `base_{av,ar}.parquet`
- `02_build_datasets.py` — doc-level train/eval split + `stage3_build`
- `03_upload_datasets.py` — publish the joined dataset
- `train_sft.py` — SGLang-stub launcher for miles `train.py`
- `run_av_sft.sh` / `run_ar_sft.sh` / `run_both.sh` — the SFT runs (continue from RLed)
- `04_convert_upload_checkpoints.py` — actor DCP→HF + publish both checkpoints
- `eval_round_trip_fve.py` — end-to-end round-trip FVE on held-out samples
- `eval_av_samples.py` — dump N held-out AV samples (context + gold + model) to a file
- `setup_box.sh` — turnkey env setup (applies ENV_FIXES)
- `ENV_FIXES.md` — the environment patches and why

## Cost
~6 h on one H200 @ ~$2.65/hr ≈ **~$16** for the full pipeline (extraction + both SFTs +
uploads). The round-trip FVE eval added ~15 min on a $1.67/hr H100 ≈ **~$0.40**.

## Next
RL with a modified reward, starting from these two checkpoints (use `configs/rl.sh` with
`--hf-checkpoint` = base + `--load`/`--critic-load` pointing at these, per the repo's RL docs).
