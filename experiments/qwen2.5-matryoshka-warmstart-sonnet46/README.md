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
- `02_build_datasets.py` — doc-level train/eval split + `stage3_build` (writes `base_*_train.parquet`)
- `03_upload_datasets.py` — publish the joined dataset
- `train_sft.py` — SGLang-stub launcher for miles `train.py`
- `run_av_sft.sh` / `run_ar_sft.sh` / `run_both.sh` — the SFT runs (continue from RLed)
- `04_convert_upload_checkpoints.py` — actor DCP→HF + publish both checkpoints
- `05_build_rl_parquet.py` — stage-`rl` parquet (prompts + activations) from `base_av_train.parquet`
- `run_rl_truncated.sh` — **RL phase** with random-length truncation (this dir's launcher)
- `eval_round_trip_fve.py` — round-trip FVE + **short-prefix (info-upfront) FVE** by content length
- `eval_av_samples.py` — dump N held-out AV samples (context + gold + model) to a file
- `setup_box.sh` — turnkey env setup (applies ENV_FIXES)
- `ENV_FIXES.md` — the environment patches and why

## Cost (warm-start)
~6 h on one H200 @ ~$2.65/hr ≈ **~$16** for the full pipeline (extraction + both SFTs +
uploads). The round-trip FVE eval added ~15 min on a $1.67/hr H100 ≈ **~$0.40**.

---

## RL phase — random-length truncation ("information upfront")

RL these two checkpoints with a modified training process: each AV generation is **capped
at a random number of explanation CONTENT tokens** (uniform in [16, 130] — min 16 keeps the
online critic's targets learnable; **shared across a
prompt's 8 GRPO samples**), so the reward is computed on a random-length prefix and the model
is pushed to **put the most important information first**. The token-limit penalty is removed.
Mechanism + rationale: `nla/truncation.py`; the feature is OFF unless `NLA_TRUNC_MAX_TOKENS>0`.

**Tests** (`tests/`, run with `pytest`): truncation RNG (shared-per-group, range, hashseed-
independence), the close-tag-tolerant extractor, and wiring tests for `reward._prep_batch`
and `nla_generate.generate()` (budget override, group-sharing, penalty removal).

### RL env setup (heavier than SFT — real SGLang stack)
Unlike the SFT warm-start (which stubbed SGLang via `--debug-train-only`), RL runs the real
SGLang rollout. `setup_rl_box.sh` builds the full stack on a stock pytorch image and applies
the fixes that were needed to run it there (validated 2026-06-26, reaches steady-state training):
- **torch 2.9.1/cu129 + SGLang 0.5.7** (commit 24c91001) per miles' `build_conda`;
- **Ray pinned to 2.47.1** — latest Ray (2.55) changed `CUDA_VISIBLE_DEVICES` semantics and
  breaks colocated rollout-engine GPU assignment;
- **`libnuma-dev`** — sgl-kernel's fp8 ops need `libnuma.so.1` (absent on the stock image);
- two `miles/ray/rollout.py` engine-`runtime_env` edits (drop `NOSET_VISIBLE_DEVICES` so each
  engine takes its own Ray GPU; drop the `LD_LIBRARY_PATH` override that forced CUDA-12.4 libs
  and broke torch-2.9 CUDA init). These assume miles' own Docker image's CUDA layout.

The launcher also passes `--wandb-group` (miles requires it) and starts RL from the **HF**
actor via `--hf-checkpoint` (no DCP `--load`; the warm-start DCP is gone).

### Run the 300-step signal run
One **8×H100 node**, 512-batch (64 prompts × 8), actor 4 / critic 2 / rollout 2,
**~32s/step** (truncation shortens generations) ⇒ 300 steps ≈ **~2.7 h**.
**Provision the box at ≤ $20/hr total** (≤ $2.50/GPU/hr) ⇒ run ≈ **~$55**; setup/debug overhead
on a fresh box is extra.

```bash
# box already set up via setup_box.sh, with the warm-start outputs present:
export ACTOR_SFT_CKPT=/workspace/ckpt/av/iter_0000859   # AV DCP iter dir (+ nla_meta.yaml)
export CRITIC_SL_CKPT=/workspace/ckpt/ar/iter_0000857/hf # AR HF dir
export RUN_DIR=/workspace/rl_trunc
bash experiments/qwen2.5-matryoshka-warmstart-sonnet46/run_rl_truncated.sh
# builds the RL parquet if missing, dumps launch_config to $RUN_DIR, wandb on,
# checkpoints every 50 steps. Knobs: NUM_ROLLOUT, NLA_TRUNC_MAX_TOKENS, ACTOR_LR.
```

### Read the signal (success metric)
```bash
# download step-N actor→/workspace/av, critic→/workspace/ar, av_eval.parquet, then:
cd /root && PYTHONPATH=/workspace/nla python \
  /workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/eval_round_trip_fve.py 300
```
Reports round-trip FVE at content-prefix lengths 10/30/60/130 + full. **Short-prefix FVE
rising across checkpoints (and the short↔full gap shrinking) = information is moving upfront.**
Also gate on: full `fve_nrm` recovering past the warm-start 0.49, `resp_len` pinned to the
cap (no drift), `k3`≈0.001, CJK=0.

### Continue later (resume) — "save everything"
The run saves all checkpoints (every 50 steps) and a `launch_config.*.txt` (git commit + every
env knob) to `$RUN_DIR`. To extend/resume, just **re-run the same command**: it auto-detects
`$RUN_DIR/actor/iter_*`, loads the latest (weights + optimizer + rollout_id), keeps the SFT as
the fixed KL reference, and continues. Raise the budget with `NUM_ROLLOUT=1000`. (For
off-box durability, set `NLA_BACKUP_REMOTE` + `NLA_BACKUP_STORAGE_CLS` to sync checkpoints to
your own storage — see `nla/data_source.py`.)

### Decision at 300
- short-prefix FVE climbing + full FVE recovered → extend (`NUM_ROLLOUT=1000`, resume).
- flat / not recovering → stop; revisit LR, the truncation range, or add strict close-tag
  loss-masking (currently we train on the rare naturally-completed endings).
