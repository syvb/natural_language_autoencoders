# Plan: full NLA training run on Qwen3.5-27B (from scratch)

Scope: the full NLA pipeline — activation extraction → stage-3 build → AV/AR
SFT warm-start → matryoshka RL → round-trip FVE eval — on a new base model,
`Qwen/Qwen3.5-27B`. "From scratch" = no reuse of any Qwen2.5 checkpoints or
activations. The gold *text* dataset (Sonnet-4.6 explanations) is reusable by
design (see Phase 0 decision); regenerating it is priced as an option.

## Headline estimate

| | recommended path | with stage-2 regen |
|---|---|---|
| GPU spend | **~$500–700** | same |
| Anthropic API | $0 (reuse gold text) | +$1.3–1.8k (Batch API, Sonnet 4.6) |
| GPU wall-clock | ~20–24 h | +0 (API runs offline) |
| Calendar | **~1 week** (integration slack dominates) | +1–2 days |

Cost anchor: the entire v3 run at 7B was ~$16 warm-start + ~$50–70 RL. The 27B
multiplier is ~4× FLOPs, plus H200-class boxes (8×H200 ≈ $35/hr on Vast today;
zero 8×H100 offers available).

## The target model — what changes vs Qwen2.5-7B

`Qwen/Qwen3.5-27B` (Feb 2026, Apache 2.0): **64 layers, d_model 5120**, hybrid
layout 16 × (3× Gated-DeltaNet→FFN, 1× gated-attention→FFN) — 48 of 64 layers
are linear-attention. Multimodal (vision tower we won't use), thinking-mode
default, vocab 248,320, **instruct-only (no base checkpoint)** — same
situation as Qwen2.5-7B-Instruct, which is what kitft/v3 initialized from.

Derived NLA constants (rules from `docs/design.md` / `model_presets.py`):

| constant | value | how |
|---|---|---|
| extraction `layer_index` | **42** | 2/3-depth rule `(2*64)//3` (Qwen2.5's 20/28 was historical, not the rule) |
| `critic_num_layers` | **43** | layer_index + 1; `prepare_critic_checkpoint.py`; ≈17B params |
| `d_model` | 5120 | config; `mse_scale = √5120 ≈ 71.55` automatic |
| `injection_scale` | **measure** | round number just above mean L2 of layer-42 residuals (Phase 1 byproduct; spans 30→80000 across models — never guess) |
| injection token | **re-search** | new 248k vocab; rerun rare-token search, refresh `injection_token_cache.yaml`; keep a CJK char so the CJK smoke test stays valid |

Architecture consequences, in decreasing order of risk:

1. **Stack bump (the schedule risk).** Qwen3.5 needs sglang ≥ 0.5.9–0.5.10
   and post-4.57.1 transformers. Our confirmed RL recipe pins
   `lmsysorg/sglang:v0.5.7-cu129` + `miles@051cd15` + `transformers==4.57.1`.
   So: new lmsysorg image, check radixark/miles upstream for a newer pin,
   re-anchor `miles_patches/0001,0002` and `patches/apply_sglang_patches.sh`
   (they're regex-anchored for exactly this), re-check the `rollout.py`
   NOSET_VISIBLE_DEVICES fix, re-verify the transformers-5.x chat-template
   mistokenize issue against the sidecar token asserts. Budget a full day.
2. **Injection path is fine in principle.** Our sglang patches are API-level
   (input-embeds through `/generate`, tokenizer_manager, schedule_batch), not
   model-file patches — GDN layers don't care where embeddings come from. The
   multimodal wrapper may route Qwen3.5 through a VLM class needing an analog
   of `nla_gemma3_mm_input_embeds.patch`; the Gemma-3 precedent (nested
   `text_config` / `language_model`, mm embeds) charts this. Verify whether
   Qwen3.5 scales embeddings post-lookup (Gemma's √d gotcha).
3. **GDN training kernels.** HF-side FSDP training needs flash-linear-attention
   (+ causal-conv1d) kernels; flash-attn only covers the 16 attention layers.
   New wheels to cache in `syvb/nla-rl-build-cache`. bf16 backward through the
   GDN recurrence is untested territory for us — Qwen2.5 already produced
   intermittent non-finite grads (33% skips pre-guard), so **grad_guard stays
   mandatory** and the smoke gate is "skips bounded and fve_nrm learning".
4. **Thinking-mode default is a non-issue for AV/AR** (we use our own
   templates and SFT stomps the format) but check `--loss-mask-type qwen`
   against the new chat template, and expect the untrained model to emit
   `<think>` early in warm-start.

## Phase 0 — decision: gold explanations (user call)

The stage-2 prompt is a function of the **text prefix only** (never sees the
activation). The matryoshka source (`ceselder/nla-matryoshka-warmstart-sonnet46`,
~450k succeeded rows = ~225k av + ~225k ar) stores `input_text` verbatim;
`01_extract_activations.py` already rebuilds activations by retokenizing it —
that is how v3 itself was built with zero API calls. Retokenizing the same
text with Qwen3.5's tokenizer moves the token *index* but not the character
boundary, so the explanation remains valid for the new model's activation.

- **Option A (recommended, $0): reuse.** Adapt `01_extract_activations.py` to
  Qwen3.5 (drop the position==token-count assert; position := len(retokenized
  prefix)). Also gives an apples-to-apples 7B-vs-27B comparison on identical
  gold data.
- **Option B (+$1.3–1.8k batch API, +1–2 days): regenerate** ~450k Sonnet-4.6
  calls on 27B-tokenized prefixes. Only worth it if we also want the known
  recipe improvement — more features per explanation (20–30 vs the hard-fixed
  ~10 that caps AV response length; the v2 analysis identified regen as the
  only lever for that). Could also down-scope to ~half size for ~$700.

Plan below assumes Option A.

## Phases, time, cost

| # | phase | box | wall-clock | est. cost |
|---|---|---|---|---|
| 1 | Porting prep: preset entry, injection-token search, adapt `01`/`02`/`02c`/`05` scripts, sidecar plumbing (L42, d5120, critic 43) | none (CPU) | 0.5–1 day human | $0 |
| 2 | Bring-up + stage-0: load 27B on new transformers, measure `injection_scale`, injection smoke (CJK test), extract ~450k layer-42 activations (43-layer truncated fwd) | 1×H200 (~$3/hr) | 4–6 h | **~$15–20** |
| 3 | Stage-3 build (bullets, AR token-trunc U[1,120], 2% full) + **AV & AR warm-start** + convert/upload + round-trip FVE baseline. 27B full-FT ≈ 360 GB optim state (scaling the measured 7B ≈ 106 GB) → FSDP over 4×H200; 30-min SFT smoke first (GDN-backward gate) | 4×H200 (~$16.5/hr) | 6–8 h | **~$110–150** |
| 4 | RL stack integration: new sglang image + miles bump, patch re-anchor, 20-step smoke on the full topology | 8×H200 (~$35/hr) | 2–3 h GPU (+ ~1 day human) | **~$70–100** |
| 5 | **RL, 200 steps**: v3 recipe verbatim — GRPO n=8, batch 64×8=512, token-trunc U[1,120], KL 0.03 vs warm-start ref, lr 1e-5, save@50, per-iter HF push. Est. ~140 s/step (v3: 41 s at 7B; ×3.4 for 27B on H200) → ~8 h | 8×H200 | 8–10 h | **~$280–380** |
| 6 | Eval: round-trip FVE + truncation sweep + fve_dist-style distribution + samples (4090s can't hold 27B → 1×H200) | 1×H200 | 2–3 h | **~$10** |
| | **total** | | ~20–24 h GPU, ~1 week calendar | **~$500–700** |

Memory sizing behind Phase 4/5 topology: actor 27B ≈ 360 GB FSDP state → fits
actor4 on H200-141GB; critic 43-layer ≈ 17B ≈ 210–270 GB → critic2 is
borderline, **expect actor4/critic3/rollout1** (rollout1 costs ~+30% step
time) or optimizer CPU-offload; 8×B200-192GB (~$39.5/hr) restores the
canonical actor4/critic2/rollout2 if H200 is too tight. Decide at smoke.

## Recipe choice

Replicate v3 exactly (truncation U[1,120] tokens, KL 0.03, 200 steps,
grad_guard, T=1 rollouts, never-train-EOS AV, `--rollout-global-dataset` ON
this time) so the 7B→27B comparison is clean. Diagnostics-driven tweaks from
the fve_dist work (reward shaping vs the ~6% catastrophic-rollout tail,
variance reduction, CJK penalty) are a *second* run once the 27B baseline
exists — don't confound scale with recipe.

## Naming / conventions

- Experiment dir: `experiments/qwen3.5-27b-matryoshka/`; HF:
  `syvb/nla-qwen3.5-27b-L42-{av,ar}-matryoshka-sonnet46` (warm-start),
  `syvb/nla-qwen3.5-27b-L42-rl` (per-iter `iter_N/{av,ar}`), dataset
  `syvb/nla-qwen3.5-27b-L42-matryoshka-sonnet46`.
- wandb `octahedral-systems`, project `nla-qwen3.5-27b`; launch_config dump
  per run; boxes per ~/ENV.md (rank on inet_down_cost, ≥600 GB disk, destroy
  after, artifact-gated sentinels).

## Top risks

1. **miles × sglang≥0.5.10 drift** — patches are regex-anchored and will tell
   us exactly where they miss, but miles' internal sglang API use may have
   moved; mitigation: check upstream miles for a newer pin first. (schedule
   risk, not cost risk)
2. **GDN bf16 backward instability** — grad_guard catches non-finite grads,
   but a high skip-rate would starve learning; gate at the Phase-3 SFT smoke.
3. **Critic memory fit** — fallback topologies cost throughput, not
   feasibility.
4. **Instruct/thinking init** — same class of init as v3 (instruct model);
   residual risk is style leakage early in SFT, historically handled by 1
   epoch of warm-start.
5. **injection_scale surprise** — hybrid-model residual norms may be unusual;
   it's measured, not guessed, and the CJK smoke test catches silent failure.

Note: `Qwen3.6-27B` (Apr 2026) is the same architecture with a better
post-train; this plan applies to it unchanged if we'd rather target that.
