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

### RL outcome (2026-06-27) — the critic NaN, root cause, and the fix

The first RL attempts **NaN'd within a few steps** at the 512-batch profile — batch-size
sensitive (clean at 128, died at 512), critic-side (`fve_nrm`→nan), LR-independent.

**Root cause (two layers):**
1. The online AR critic (which *is* the reward model) is co-trained on truncated prefixes →
   occasionally tiny-norm / degenerate predictions. The direction-only MSE loss
   differentiates through `normalize_activation`; its backward was numerically unsafe at
   small norm. Hardened by moving the eps **inside the sqrt** (`nla/schema.py`, commit
   `834c2b2`) — necessary but **not sufficient**.
2. The real killer on the **canonical critic-sharded-across-2-GPUs config**: a step with a
   **finite loss but a NaN gradient** (a bf16 backbone-backward overflow on Qwen "massive
   activations" — preds were normal-norm, so this is *not* the `normalize_activation`
   singularity). The in-loss finiteness backstop only checks the *loss*, and miles steps
   the optimizer **unconditionally** after `clip_grad_norm_` (NaN norm → NaN clip coef →
   every shard's grads NaN → poisoned critic → all rewards collapse to the `-2` penalty).

**Fix:** `nla/grad_guard.py` (`install_grad_finiteness_guard`, wired into
`NLAFSDPActor.init`, commit `26d9484`) wraps `optimizer.step` to **skip the step + zero
grads when any gradient is non-finite**, for both roles. Globally consistent across ranks
(clip already homogenized the NaN). Tests: `tests/test_grad_guard.py`.

**Validated** on the exact failing config (8×H100 actor4/critic2/rollout2, batch 512):
NaN grad on ~33% of steps (intermittent), each skipped, `loss_nonfinite=0` throughout,
`fve_nrm` 0.35→0.56, reward never hit `-2` — where the unguarded run died at step 0.
(Caveat learned the hard way: a **critic1** smoke test does NOT reproduce this — only the
FSDP-sharded critic2 does. Reproduce stability bugs on the canonical sharding.)

**PoC run (KL=0.01, the "real" config):** matches the original RL's KL strength (ref =
warm-start actor), same 512-batch/1e-5 profile, run to **200 steps**: stable throughout
(`loss_nonfinite=0`, `fve_nrm` 0.33→~0.68, `kl_loss` 0→~2.5, entropy 1.0→0.71).

**Checkpoints:** HF `syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard` (private) —
`kl0.01/iter_{100,200}/{av,ar}` (KL-on PoC; iter_200 final) and `kl0/iter_*` (KL-off
reference). `av/` = full HF actor + `nla_meta.yaml`; `ar/` = HF critic + `value_head`.

**Reproduction artifacts:** `setup_rl_box_lmsys.sh` (RL env on the
`lmsysorg/sglang:v0.5.7-cu129-amd64` image), `push_checkpoint_to_hf.sh` (convert actor
DCP→HF + upload per-iter + free disk). Open optional lever: fp32 value-head / upstream-grad
clamp to eliminate the ~33% grad-skips at the source (efficiency only; not needed for
stability or learning).

## v3 — bullets prompt, uniform 1–120-token truncation, KL 0.03

Three changes over v2, all implemented (scripts `*_v3.sh`, builder
`02c_build_datasets_v3.py`):

1. **Prompt actually fixed.** v2 shipped with the v1 "2-3 text snippets in
   `<explanation>` tags" prompt through TWO independent legs, both fixed:
   (a) `05_build_rl_parquet.py` didn't pass `--explanation-format`, so the RL
   parquet kept stage3's tagged default — it now REQUIRES `EXPLANATION_FORMAT`
   (no default) and verifies a pre-existing output's sidecar template before
   its skip-if-exists early exit; (b) the SFT scripts passed no
   `--nla-sidecar-source`, and `resolve_sidecar_source` prefers the
   hf-checkpoint's sidecar over the training parquet's — so the warm-start
   loaded the kitft base's TAGGED sidecar and `_write_sidecar` baked it into
   every exported checkpoint (that's why even the v2 *warm-start* ships the
   tagged prompt). `run_{av,ar}_sft_v3.sh` now pass `--nla-sidecar-source` =
   the training parquet. Belt-and-suspenders: `run_rl_v3.sh` aborts if the
   warm-start sidecar contains `<explanation>`, and `nla_generate` asserts at
   the first rollout that the RL parquet's prompt equals the sidecar's
   template (`NLA_SKIP_TEMPLATE_CHECK=1` to bypass). v3 adds
   `--explanation-format bullets` (`stage3_build`): the prompt says *"a list
   of bullet points, one per line"*, while the SFT targets/critic inputs stay
   plain one-item-per-line text (same as v2's list — literal `- ` markers
   would eat ~1.2 budget tokens/item with zero reconstruction information and
   break per-token comparability with v1/v2). `RL_PARQUET` defaults to
   `rl_v3.parquet` so a stale tagged `rl.parquet` on a reused box can't sneak
   back in. The `<concept>{injection_char}</concept>` context is unchanged →
   same injection token (`㈎`/149705, now pinned in
   `injection_token_cache.yaml`) and neighbors (`>`/29, `</`/522).
2. **Back to uniform token truncation, [1, 120]** (v1 mechanism —
   `max_new_tokens` cap, no post-truncation — but full-range uniform instead of
   v1's [16, 150] and v2's tapered item counts). min=1 was what NaN'd v1's
   critic; two defenses make it viable now: the AR warm-start is pre-calibrated
   on `K ~ U[1,120]`-token prefixes (`stage3_build --ar-truncate-max-tokens`,
   same uniform draw RL uses) so step-0 short prefixes are in-distribution, and
   the grad-finiteness guard (`26d9484`) skips any non-finite critic step.
   2% of AR warm-start rows are left untruncated
   (`--ar-truncate-keep-full-frac`): RL prefixes are capped at 120 too, so
   without this the critic would never see longer text at ANY stage and
   full-length round-trip FVE would read artificially low. Still compare
   "full-length" FVE across versions at a common 120-token cap.
   Tokens-mode's `+"<explanation>\n"` budget offset is now format-aware
   (`nla_generate`): 0 when the sidecar template has no tag, so a 1-token
   budget really is 1 content token — `run_rl_v3.sh` also pins
   `NLA_TRUNC_OPENING_OFFSET=0` and zeroes lingering v2 item-mode env vars.
   The item-length penalty is gated to items mode in `reward.py` (a stale
   `NLA_ITEM_LEN_PENALTY` export can't shape v3 rewards). Known cost: K=1
   groups (~0.8%) likely produce near-identical first tokens → ~zero
   advantage; wasted but harmless (`NLA_TRUNC_MIN_TOKENS=2` is the knob if
   the waste bothers you).
3. **KL 0.03** (v1: 0.01, v2: 0.02) — anchor the AV harder to the warm-start
   reference.

```bash
# SFT box (same setup_box.sh flow as v1/v2; base parquets already built by 02):
python 02c_build_datasets_v3.py          # av/ar_sft_v3 + av/ar_eval_v3 (bullets, AR token-truncated)
AV_HF_CKPT=... bash run_av_sft_v3.sh     # from kitft base AV; NLA_NO_TRAIN_EOS=1
AR_HF_CKPT=... bash run_ar_sft_v3.sh     # from kitft base AR; U[1,120]-token critic inputs
# RL box (setup_rl_box_lmsys.sh):
ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... bash run_rl_v3.sh   # KL=0.03, tokens ~U[1,120]
```

### Read the signal (v3)

- **Before RL**: run the FVE sweep + front-loading eval on the v3 *warm-start*
  — it's a fresh SFT from the kitft base with new targets, so the v1/v2
  baseline ladder (line-1 ΔFVE 0.205 ws → 0.433 v1 → 0.596 v2) does not
  transfer; without a v3-ws baseline the RL delta is uninterpretable. Use
  prefix lens `{1,2,5,10,30,60,120}` (v3 uniquely trains the 1-5 regime; 130
  exceeds the trained max).
- **During RL (wandb)**: grad-guard skip rate (reference: ~33% intermittent
  was healthy on v1's config; sustained higher or `fve_nrm` stalling ≈0.3 →
  raise `NLA_TRUNC_MIN_TOKENS` instead of burning steps); line structure —
  the newline separators are the only format tokens and pure reward prefers
  shedding even those (KL is the counterweight), so grep the
  `NLA_ROLLOUT_TEXT_DUMP` samples for degeneration into one giant unbroken
  item; CJK in outputs = injection failure, as always.
- **Comparisons**: v3 changes prompt wording, truncation scheme, KL, and
  warm-start lineage at once — but the output format is byte-compatible with
  v2's list, so per-token FVE curves are directly comparable across v1/v2/v3.
  Budget one control (e.g. `KL_LOSS_COEF=0.02` rerun) before tearing the box
  down if attribution matters.

### v3 RL outcome (2026-07-02) — 200 steps, complete

Run: 8×H100 (actor4/critic2/rollout2), 512-batch, KL 0.03, tokens ~U[1,120],
offset 0, ~41 s/step, **zero grad-guard skips across all 200 steps** (v1's
config skipped ~33% — the U[1,120] AR warm-start pre-calibration + offset-0
fix removed the instability at the source). raw_reward −0.507→−0.261,
critic fve_nrm 0.30→0.64, plateau ≈ step 110.

Round-trip FVE at iter_0000200 (150 held-out, evaluated FROM the HF upload;
warm-start baseline in parens): 1tok −0.31 (−0.31) | 2 −0.00 (−0.22) |
5 0.29 (0.00) | 10 **0.45** (0.11) | 30 0.65 (0.35) | 60 0.68 (0.44) |
120 0.69 (0.47) | full 0.67 (0.44). A 10-token prefix now beats the
warm-start's full output. Full-length sits near its own peak (0.667 vs
0.693@120) — v2's over-extension dip is largely gone. Critic-on-gold fell to
0.44 (critic specialized to the AV's phrasing — expected under co-training).
Caveat: 8/150 outputs contain a stray CJK char (minor leak, up from 0/150
at warm-start) — check at higher KL-drift before drawing conclusions.

Artifacts (all public): per-50-step inference-ready checkpoints in
`syvb/nla-qwen2.5-7b-L20-v3-rl` (iter_00000{50,100,150,200}/{av,ar}),
resumable iter_0000200 (actor DCP + optimizer, critic hf) in
`syvb/nla-qwen2.5-7b-L20-v3-rl-checkpoints`, eval in
`v3_warmstart_results/v3_rl200_fve.txt`.

### v3 faithfulness probes (2026-07-02) — paraphrase robustness + order optimality

`eval_paraphrase_order.py` on 100 held-out samples (fig:
`v3_faithfulness_results/fig_paraphrase_order.png`):

- **Paraphrase robustness — the v3 RL gains are largely NOT paraphrase-robust.**
  Line-wise meaning-preserving paraphrase (Haiku): v3 RL pair full-length FVE
  0.650 → 0.165 (**25% retention**) vs the pre-RL warm-start pair 0.431 → 0.314
  (**73% retention**). RL made the AV↔AR channel brittle to wording: much of the
  added FVE rides on exact surface forms — partly verbatim source quotes (e.g.
  line 1 `" some of" sings own music`), which paraphrase legitimately destroys,
  and partly co-adapted phrasing conventions. Either way, treat v3's FVE as
  substantially lexical, not purely semantic; the cross-critic matrix (eval #2)
  would split quote-reliance from pair-specific code.
- **Order optimality — real but imperfect front-loading.** Model order captures
  ~58-72% of the greedy-oracle ordering gap over random (k=1: model 0.352 vs
  oracle 0.527 vs random 0.056). Order is irrelevant at full length (random
  0.66 ≈ orig 0.65 ✓), so this isolates ordering. Cross-sample floor −0.93.

## suffix — "left-matryoshka" RL (reward on the LAST k tokens)

The mirror experiment: RL the SAME v3 warm-start pair, but the critic scores
the **last ~U[1,120] content tokens** of each rollout instead of the first —
pushing the model to put the most important information at the **end**.
Implemented as truncation mode `suffix` (`nla/truncation.py`); launcher
`run_rl_suffix.sh`. Design points (rationale in the launcher header):

- **v3 warm-start reused, no mirrored re-SFT** — left-RL and v3's right-RL
  start from the identical checkpoint, isolating the reward direction *on the
  actor*. Known residual asymmetry (accepted): the init is front-loaded, i.e.
  reward-aligned for v3 but reward-opposed here — a positive back-loading
  result is conservative; a null is confounded and does NOT establish
  "back-loading is harder".
- **Critic burn-in first (two-phase launch).** The reused AR critic — the
  reward model — was pre-calibrated on PREFIX cuts; suffix cuts are
  out-of-distribution at step 0, and v3 attributes its stability and timeline
  to starting task-calibrated. Phase 1 calibrates it on frozen-actor rollouts,
  phase 2 trains the actor; judge actor progress from iter_25, not iter_0:

  ```bash
  # RL box (setup_rl_box_lmsys.sh), v3 warm-start downloaded from HF:
  ACTOR_LR=0 NUM_ROLLOUT=25 ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... bash run_rl_suffix.sh
  NUM_ROLLOUT=75            ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... bash run_rl_suffix.sh
  ```

- **`ROLLOUT_MAX_RESP` = `NLA_TRUNC_MAX_TOKENS` = 120, enforced.** With
  resp=120, position p is scored with probability (p+1)/120 — the exact
  mirror of v3's (120−p)/120. Any resp > max budget creates a head region
  never scored at ANY k: a free scratchpad where the cheapest policy is to
  keep front-loading (KL-cheap) and *duplicate* content into the tail —
  and suffix-FVE can't distinguish duplication from reordering. The launcher
  aborts on a mismatch. Every eval arm must pass `NLA_GEN_MAX_NEW=120` for
  the same reason.
- **Full trajectory trained, suffix scored.** Generation is never capped (the
  AV never emits EOS → every rollout is exactly 120 tokens); the actor's
  tokens/loss-mask/logprobs are NOT sliced; only the critic input (reward
  forward + co-training tokens) is cut to the last k via
  `truncate_to_token_suffix` — same helper, same per-group k at both
  consumers. Same k RNG stream as tokens mode → identical budget sequence to
  a same-seed v3 run.
- **KL 0.01** (v3: 0.03): the KL reference is the front-loaded warm-start, so
  a strong anchor would penalize exactly the ordering change under test.
  Cost: v3's gains were already substantially lexical at 0.03 (25% paraphrase
  retention), and a weaker anchor + recalibrating critic can worsen that — so
  the paraphrase probe is part of the success read, not an afterthought, and
  "trains as well as v3" claims are off the table across a KL difference.

### Read the signal (suffix)

- **Before RL — both-sides baseline on the v3 warm-start.** Never measured on
  the suffix side; without it the RL delta is uninterpretable. MUST use the
  v3-format eval parquet (the sweep-eval-prompt-match trap: the old
  `av_eval.parquet` silently drops FVE ~0.15 on a v3 model) and the training
  generation cap:

  ```bash
  EVAL=/workspace/out/av_eval_v3.parquet AV_DIR=<v3-ws-av> AR_DIR=<v3-ws-ar> \
    NLA_TRUNC_SIDE=both NLA_GEN_TEMP=1 NLA_GEN_MAX_NEW=120 \
    python eval_round_trip_fve.py 100 1,2,5,10,30,60,120
  ```

- **Attribution control — v3 iter_200 evaluated both sides** (same command,
  AV_DIR/AR_DIR from `syvb/nla-qwen2.5-7b-L20-v3-rl`, ~15 min GPU). v3 RL
  raised FVE at *every* length, so "suffix-FVE above the warm-start baseline"
  is expected even with zero reordering. The clean ordering metric is the
  **prefix-vs-suffix gap flip**: v3-final should show prefix ≫ suffix at
  short k; suffix-final the reverse — each read against the shared
  warm-start's both-sides curve. Evaluate suffix checkpoints with
  `NLA_TRUNC_SIDE=both` too (does back-loading *destroy* front-loading or
  add to it?). Keep one sampling regime (`NLA_GEN_TEMP=1`) across all arms —
  v3's README numbers were greedy, so re-derive any curve you compare against.
- **Gate (end of phase 2, i.e. 50 actor steps)**: critic `fve_nrm` recovered
  during burn-in; short-SUFFIX FVE (judge at k=1–10 — at k≥60 the suffix of a
  120-token output is most of the text and "above baseline" is nearly
  vacuous) above the warm-start baseline at iter_50/75; grad-skip rate ~0;
  CJK=0. **The gate is one-sided**: positive signal → go (extend to ~200 by
  re-running with a higher `NUM_ROLLOUT` — resume is free); flat signal is
  NO-GO only after extending to ~120 actor steps (v3's plateau was ~110);
  50 actor steps is the cheap first look, not the verdict. Log the step at
  which `fve_nrm` crosses its burn-in level so "effective actor steps" is
  recorded, not guessed.
- **Faithfulness read**: run `eval_paraphrase_order.py` on the final
  checkpoint (is back-loaded FVE more or less lexical than v3's?), and set
  `NLA_QUOTE_STATS_JSONL` during RL (free) — final-token echo is directly
  rewarded at k=1–5, so short-suffix FVE must be read next to the echo stats.
- **Comparisons**: this run's suffix curve vs (a) its own warm-start
  baseline, (b) the gap-flip control above, (c) v3's prefix curve at matched
  k (budget-paired via the shared RNG stream — pin `NLA_TRUNC_SEED` to v3's
  value; both default to `rollout_seed`), (d) the 27B reversed-order eval
  (`qwen3.6-27b-evalsuite/clean_rev_fve.py`): reordering alone doesn't
  recover matryoshka-level short-budget FVE — does *training* for it?

### Suffix RL PoC outcome (2026-07-22) — 25 burn-in + 25 actor steps, stopped early by design

Run: 8×H100 (actor4/critic2/rollout2), 512-batch, KL 0.01, suffix ~U[1,120],
RESP=120 (exact mirror), two-phase launch. Training: burn-in critic fve_nrm
recovered the OOD cliff in ~3 steps then plateaued ≈−0.05 (frozen front-loaded
actor's suffixes ≈ uninformative); actor phase raw_reward −0.55→−0.37, critic
suffix fve_nrm →0.44–0.59; grad-guard skips ~30% intermittent, contained; zero
crashes. Checkpoints: `syvb/nla-qwen2.5-7b-L20-suffix-rl` (iter_0000025 = burn-in
/ actor-untouched, iter_0000050 = 25 actor steps). Eval (100 held-out, T=1,
NLA_GEN_MAX_NEW=120, both sides; tables in `suffix_rl_results/fve_*.txt`):

| arm | prefix@10 | suffix@10 | prefix@60 | suffix@60 | @120 | asym(p−s)@10 |
|---|---|---|---|---|---|---|
| ws (baseline) | 0.04 | −0.38 | 0.32 | −0.21 | 0.35 | **+0.42** |
| it25 (critic recal only) | 0.07 | 0.01 | 0.48 | 0.49 | 0.53 | +0.06 |
| it50 (25 actor steps) | 0.13 | 0.07 | 0.58 | 0.57 | 0.61 | +0.06 |
| v3rl-200 (control) | 0.43 | −0.20 | 0.67 | 0.55 | 0.68 | **+0.63** |

Findings:
1. **The warm-start's suffix-blindness was mostly critic-side.** it25's AV is
   byte-identical to ws, yet critic suffix-recalibration alone took suffix@60
   from −0.21 to +0.49 and made the pair's curves position-symmetric — the
   information was in the suffixes; the prefix-calibrated critic couldn't read
   it there. The ws-vs-v3rl "front-loading" asymmetry is therefore partly a
   readout artifact, a caveat that applies to prefix-truncation evals
   generally.
2. **25 actor steps: real but generic gains, no gap-flip yet.** Both curves
   rose ~equally (suffix@30 0.33→0.45, prefix@30 0.36→0.47); at short k the
   pair is now symmetric (+0.06) vs ws +0.42 and v3's +0.63 — moving in the
   mirrored direction but suffix does not yet beat prefix anywhere.
3. **Critic-on-gold declined 0.48→0.38→0.33** across burn-in and actor steps —
   the usual co-training specialization; run the paraphrase probe before
   claiming semantic (vs lexical) suffix gains.

Continuation (needs re-warm of optimizer): resume from the HF iter_0000050
pair via ACTOR_SFT_CKPT/CRITIC_SL_CKPT (KL re-anchors to iter_50; raw DCP
state was not preserved). The gap-flip question — does suffix beat prefix at
short k with v3-scale step counts (~200)? — is open and now cheap to answer.
