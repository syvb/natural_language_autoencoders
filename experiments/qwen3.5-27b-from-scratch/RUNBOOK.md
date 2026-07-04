# Runbook: from-scratch NLA training on Qwen3.5-27B

A complete, ordered execution plan. Every command below runs on the GPU
machine unless noted. No prior NLA checkpoint for this model exists — SFT
starts from the plain base model and the critic init is built here.

**Provider-agnostic**: nothing here assumes a specific GPU cloud. What you
need, by phase, is listed in §Hardware. Where a *validated* reference exists
it is from the 7B smoke run of this exact branch (see §Smoke) and the 7B v3
production run (`experiments/qwen2.5-matryoshka-warmstart-sonnet46/README.md`
on the `qwen2.5-matryoshka-warmstart-sonnet46` branch of syvb's fork).

## What this run is

Full pipeline: activation extraction → dataset build → AV/AR SFT warm-start
(from scratch) → matryoshka RL (uniform token-truncation reward) → health
checks, for `Qwen/Qwen3.5-27B` at extraction layer 42 (2/3 depth of 64),
d_model 5120, critic = 43-layer truncation (~17B params).

**One recipe change vs the validated 7B v3 run — position-tapered KL**: the
KL coefficient is 0.02 *at the first response token* and halves every
`NLA_KL_TAPER_HALF_LIFE` (default 40) tokens (`nla/kl_taper.py`). Rationale:
the truncation reward front-loads information, so late tokens contribute
little FVE and were previously dominated by a flat KL. Monitor
`train/kl_loss` (weighted) vs `train/kl_flat` (what a flat run would see).

Gold explanations are REUSED from the existing text-only dataset
(`ceselder/nla-matryoshka-warmstart-sonnet46`, ~450k rows): the stage-2
prompt was a function of the text prefix only, so the explanations are valid
for any base model — 01 retokenizes each stored prefix with the Qwen3.5
tokenizer and extracts at its last token. Zero API spend.

## Hardware

| phase | GPUs | VRAM math | est. wall-clock |
|---|---|---|---|
| 1–3 data (extract/build/critic-init) | 1× ≥96 GB (H200/B200) | 43-layer fwd ≈ 36 GB bf16 + batch | 3–5 h |
| 4 SFT (per role) | 4× H200-141 GB or 4× B200-192 GB | 27B full-FT ≈ 360 GB FSDP state (params+grads+fp32 Adam ≈ 14 B/param, measured at 7B) | ~2.5–3 h/epoch/role on H200; ~1.5 h on B200. AV ∥ AR on separate GPU sets. |
| 5 RL | 8× B200-192 GB (preferred) or 8× H200 | actor4 (90 GB/GPU) + critic2 (~110–135 GB/GPU — fits B200, borderline H200) + rollout2 | ~75–85 s/step (B200 est.) → 300 steps ≈ 7 h; H200 ~140 s/step |

On H200, if the critic OOMs at critic2: `ACTOR_GPUS=4 CRITIC_GPUS=3
ROLLOUT_GPUS=1` (~+30% step time), or B200s. Disk: ≥1 TB — each RL save is
large (27B actor DCP + 17B critic DCP + optimizer, several hundred GB); push
each save with `push_ckpt_to_hf.sh` and prune if disk is tight.

## Software stack — the one genuinely new risk

Qwen3.5 (hybrid Gated-DeltaNet) needs **sglang ≥ 0.5.9** and a
correspondingly recent transformers — newer than the validated pins
(`lmsysorg/sglang:v0.5.7-cu129`, `transformers==4.57.1`, which the 7B smoke
used). Plan:

1. Start from the newest `lmsysorg/sglang` stable image whose sglang supports
   Qwen3.5 (check `docs.sglang.io/basic_usage/qwen3_5.html` for the minimum).
2. `setup_box.sh` (env knobs: `TRANSFORMERS_PIN`, `FLASH_ATTN_WHEEL`,
   `SGLANG_SRC`). Expect two friction points, both loud not silent:
   - `patches/apply_sglang_patches.sh` is regex-anchored; on a much newer
     sglang an anchor may miss — the assert names the file; port the hunk by
     hand (the sibling `.patch` files document intent). The patches are
     API-level (input-embeds through `/generate`), not model-file patches, so
     GDN does not change what they touch.
   - miles is pinned at `nla/miles_patches/UPSTREAM_PIN`; a newer sglang may
     drift from its API. If imports break, check radixark/miles for a newer
     commit and re-anchor `miles_patches/0001,0002`.
3. HF-side training additionally needs the Gated-DeltaNet kernels
   (`flash-linear-attention`, `causal-conv1d`) — install if transformers asks.
4. Multimodal wrapper: Qwen3.5-27B ships a vision tower. `nla/arch_adapters`
   already unwraps `language_model`/`text_config` (Gemma-3 precedent) and
   `NLAFSDPActor` remaps weight-sync keys for wrapped actors. 01 falls back
   to `AutoModelForImageTextToText` if `AutoModelForCausalLM` refuses the
   checkpoint. Verify at phase 1; extend `arch_adapters` if the layout is new.

Budget half a day for this integration; do it with the cheap 1-GPU phase, not
on the 8-GPU box.

## Execution

All scripts read `config.env` (setdefault: your exported env wins; point
`NLA_RUN_CONFIG` elsewhere to swap profiles). Credentials: `HF_TOKEN` (or
`HF_TOKEN_FILE`), wandb key at `WANDB_KEY_FILE`. wandb is always on.

```bash
cd <repo>/experiments/qwen3.5-27b-from-scratch

# ── phase 1: inputs + extraction (1 GPU) ─────────────────────────────────────
python 00_fetch_inputs.py                       # base model + matryoshka dataset
python 01_extract_activations.py                # ~450k fwd passes, 43-layer stack
#   GATE: read norm_stats.json → set INJECTION_SCALE in config.env.
#   GATE: "len==source_pos match" ~0% is EXPECTED (different tokenizer).
#   GATE: injection smoke — see §Injection smoke below, before any SFT.

# ── phase 2: datasets + critic init (CPU/1 GPU) ──────────────────────────────
python 02_build_datasets.py                     # doc-level split + bullets build
bash 03_prepare_critic.sh                       # 43-layer trunc + identity value head

# ── phase 3: SFT warm-start (4 GPUs per role; AV ∥ AR if you have 8) ────────
bash run_av_sft.sh                              # from the PLAIN base model
bash run_ar_sft.sh                              # from critic_init
#   GATE: python check_health.py <av log> --sft   → PASS
#   GATE: AR fve_nrm should climb well above 0 by end of epoch 1
#         (identity-init value head starts near pred=backbone-hidden).
#   SFT_EPOCHS=2 default (from scratch). Stop at 1 if round-trip FVE is
#   already ≳95% of the critic-gold ceiling; extend to 3 if still climbing.

python 04_convert_upload.py                     # AV DCP→HF + verify (UPLOAD=0 to skip HF)

# ── phase 4: RL (8 GPUs) ─────────────────────────────────────────────────────
ACTOR_SFT_CKPT=$WORK/hf_out/av_ws \
CRITIC_SL_CKPT=$(ls -d $WORK/ckpt/ar_ws/iter_*/hf | tail -1) \
NUM_ROLLOUT=20 SAVE_INTERVAL=1000 WANDB_GROUP=27b-rl-smoke \
  bash run_rl.sh                                # 20-step smoke FIRST
python check_health.py <rl log> --rl --taper    # → PASS, kl_flat present

ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... bash run_rl.sh   # full run (NUM_ROLLOUT=300)
#   Save every 50; push each: bash push_ckpt_to_hf.sh 0000050
#   Stop rule: reward slope ≈ 0 over ~30 steps AND 10-tok FVE flat across two
#   consecutive saves. Expect to stop at 150–250. run_rl.sh auto-resumes from
#   $RUN_DIR if interrupted (rollout data offset is NOT saved — resume re-draws
#   prompts from the dataset start; acceptable for GRPO).
```

### Injection smoke (before SFT — cheap, catches a silent-failure class)

With the base model + measured INJECTION_SCALE, generate from a prompt
containing the marker token with a real activation injected (see
`docs/inference.md` § injection). **A broken injection path makes the model
see the literal CJK marker char and free-associate Chinese** — grep any
generated text for CJK; that is the loudest smoke test for the whole path.
Base-model outputs will be unfocused (it hasn't been SFT'd) — you are testing
plumbing, not quality.

### Expected metric shapes (from the 7B v3 run, directional only)

| metric | healthy shape |
|---|---|
| `raw_reward` | starts ≈ −0.5, improves toward ≈ −0.25 by ~step 100–150, then plateaus. Pinned −2.0 = reward path dead. |
| `fve_nrm` (critic) | dips at RL start (distribution shift), climbs to ≈ 0.6+ by step ~150 |
| `loss_nonfinite` | 0 at every step, both roles (grad guard skips, never poisons) |
| grad-guard skips | 0 was achieved at 7B v3; occasional is fine, >40% starves learning |
| `train/kl_loss` | grows from 0 as the policy departs the ref; with the taper it should sit well BELOW `train/kl_flat` |
| `pred_norm_min` | stable in the ~100s; diving toward 0 = critic collapse |

`check_health.py <log> --rl --taper` encodes the FAIL/WARN rules; gate
automation on its exit code.

### Verification after the run

- Round-trip FVE on `av_eval.parquet` (the held-out split 02 wrote), full
  length and 10/20-token prefixes; compare against the warm-start baseline.
  The matryoshka signature = short-prefix FVE far above the warm-start's.
- Sample ~100 AV generations at T=1 (`do_sample=True, temperature=1.0,
  top_p=1.0, top_k=0` — never greedy); grep CJK rate; read a dozen.
- `value_head.safetensors` finiteness is asserted by `push_ckpt_to_hf.sh` on
  every push (do not skip — the half-shard export corruption was silent).

## Smoke test (validated profile)

`smoke_7b.env` runs this exact pipeline end-to-end at Qwen2.5-7B/L20 scale on
one 4-GPU box (extract 6k rows → build → critic init → short AV+AR SFT → 12
RL steps with the taper active). It was run green before this branch shipped;
use it to re-validate after any stack change:

```bash
export NLA_RUN_CONFIG=$PWD/smoke_7b.env
python 00_fetch_inputs.py && python 01_extract_activations.py
python 02_build_datasets.py && bash 03_prepare_critic.sh
bash run_av_sft.sh --num-rollout 30      # --num-rollout caps SFT steps (overrides --num-epoch)
bash run_ar_sft.sh --num-rollout 30
UPLOAD=0 python 04_convert_upload.py
ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... bash run_rl.sh      # NUM_ROLLOUT=12 in the profile
python check_health.py <logs> --taper
```

## Failure modes seen before (fixes are already on this branch)

- **Never run `python -m nla...` with cwd = the repo's PARENT dir** (e.g.
  /workspace when the repo is /workspace/nla): cwd goes on sys.path and the
  repo dir shadows the installed package as a namespace package —
  `nla.scripts.*` then resolves against the repo's top-level `scripts/` and
  fails "No module named". The launchers here cd defensively; keep that.
- **NaN grads on short prefixes** (bf16 backbone overflow): grad guard skips
  the step; `loss_nonfinite` stays 0. High skip-rate = investigate, don't
  extend the run.
- **Value-head corrupt on save** under FSDP-sharded critic: fixed by explicit
  all_gather; the finiteness assert in push/convert is the tripwire.
- **Wrong prompt template shipped in sidecars**: every launcher asserts the
  bullets template; `--nla-sidecar-source` on both SFTs is what makes that
  true. Do not remove either.
- **A critic1 smoke can false-PASS stability bugs** — numerical issues only
  reproduce with the FSDP-sharded critic (critic ≥ 2). Smoke wiring cheap,
  but treat stability as unproven until the 8-GPU 20-step smoke passes.
- **cp_size must be 1** (upstream invariant; the taper also asserts it).

## Cost sketch (at mid-2026 market prices, ~$3/GPU-h H200, ~$5/GPU-h B200)

extraction ~$15 · SFT (2 epochs × 2 roles on 4×) ~$150–250 · RL smoke ~$30 ·
RL 300 steps ~$280–480 · total **≈ $500–800** GPU + $0 API.
