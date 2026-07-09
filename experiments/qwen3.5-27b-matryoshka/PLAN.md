# Plan: full NLA training run on Qwen3.5-27B (from scratch)

Scope: the full NLA pipeline — activation extraction → stage-3 build → AV/AR
SFT warm-start → matryoshka RL → round-trip FVE eval — on a new base model,
`Qwen/Qwen3.5-27B`. "From scratch" = no reuse of any Qwen2.5 checkpoints or
activations. The gold *text* dataset (Sonnet-4.6 explanations) is reusable by
design (see Phase 0 decision); regenerating it is priced as an option.

## Headline estimate

| | recommended path | with stage-2 regen |
|---|---|---|
| GPU spend | **~$600–1,000** | same |
| Anthropic API | $0 (reuse gold text) | +$1.3–1.8k (Batch API, Sonnet 4.6) |
| GPU wall-clock | ~22–32 h | +0 (API runs offline) |
| Calendar | **~1–1.5 weeks** (integration slack dominates) | +1–2 days |

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
| 3 | Stage-3 build (bullets, AR token-trunc U[1,120], 2% full) + **AV & AR SFT from scratch, default 2 epochs each** (lr 1e-5, halved from 7B's 2e-5 for 27B full-FT), eval-gated both ways (stop at 1 epoch if ≥95% of critic-gold ceiling; extend to 3 if still climbing) + convert/upload + round-trip FVE baseline. 27B full-FT ≈ 360 GB optim state (scaling the measured 7B ≈ 106 GB) → FSDP over 4×H200 per role, AV ∥ AR on two boxes; 30-min SFT smoke first (GDN-backward gate) | 2× 4×H200 (~$16.5/hr ea) | ~6–8 h wall (10–14 box-h) | **~$180–260** (+$90 if epoch 3) |
| 4 | RL stack integration: new sglang image + miles bump, patch re-anchor, 20-step smoke on the full topology | 8×H200 (~$35/hr) | 2–3 h GPU (+ ~1 day human) | **~$70–100** |
| 5 | **RL, budget 300 steps, plateau stop-rule** — HARD GATE: launch only from an SFT at/near its ceiling. v3 recipe otherwise verbatim — GRPO n=8, batch 64×8=512, token-trunc U[1,120], KL 0.03 vs warm-start ref, lr 1e-5, save@50, per-iter HF push. Stop when reward slope ≈ 0 over ~30 steps AND 10-tok FVE flat across two saved ckpts; if still climbing at 300, extend by resume in +100-step increments (~$195 ea). Est. ~140 s/step (v3: 41 s at 7B; ×3.4 for 27B on H200) → ~8 h at 200, ~12 h at 300 | 8×H200 | 8–12 h | **~$280–480** (+$195/100 steps cond.) |
| 6 | Eval: round-trip FVE + truncation sweep + fve_dist-style distribution + samples (4090s can't hold 27B → 1×H200) | 1×H200 | 2–3 h | **~$10** |
| | **total** | | ~22–32 h GPU, ~1–1.5 weeks calendar | **~$600–1,000** |

Memory sizing behind Phase 4/5 topology: actor 27B ≈ 360 GB FSDP state → fits
actor4 on H200-141GB; critic 43-layer ≈ 17B ≈ 210–270 GB → critic2 is
borderline on H200, **expect actor4/critic3/rollout1** (rollout1 costs ~+30%
step time) or optimizer CPU-offload. On B200-192GB the canonical
actor4/critic2/rollout2 fits comfortably. Decide at smoke.

## GPU choice: Blackwell first, H200 fallback

B200 is both faster and cheaper per unit of work for this job, market
permitting (Vast, 2026-07-04: 8×B200 $44/hr ×1 offer, 8×B300 $50/hr ×1 offer,
4×B200 ~$20/hr ×5 offers, vs 8×H200 ~$35/hr, several offers; zero 8×H100).

- **Speed**: B200 ≈ 2.2× H200 bf16 compute, 1.67× bandwidth → realized
  ~1.7–2× on training steps, ~1.6–1.7× on decode-bound rollouts. RL est.
  ~75–85 s/step vs ~140 on H200 → 300 steps ≈ 6.5 h ($~285) vs ~12 h ($~405).
  SFT (2 epochs, AV ∥ AR on 2× 4×B200): ~3.5–4.5 h wall, ~$140–180.
- **Memory**: 192 GB removes the critic2-fit question entirely (no
  rollout1/CPU-offload fallback penalty) — an extra effective speedup vs the
  H200 plan's likely degraded topology.
- **B300 (288 GB)**: ~same bf16 as B200 (Ultra's gains are FP4 + memory);
  at $50/hr it's the *memory* fallback (bigger micro-batches, roomier
  topologies), not a speed upgrade. Use only if the B200 offer is gone or
  27B memory surprises us.
- **Risks/costs of Blackwell**: sm_100 needs the cu129+ stack we're already
  forced onto by Qwen3.5 (sglang ≥0.5.10 images run Blackwell); new
  flash-attn sm100 wheel to build once and cache in `nla-rl-build-cache`;
  fla/Triton GDN kernels are Triton (Blackwell-ok, perf maybe untuned).
  **Cap Blackwell-specific debugging at half a day, then fall back to H200.**
  Supply is the real risk: exactly one 8×B200 offer today — re-check at
  provisioning time; the H200 phase costs in the table are the conservative
  ceiling.

If the B200 offers hold, GPU wall-clock drops ~22–32 h → **~13–19 h** and GPU
spend **~$600–1,000 → ~$450–750**. Calendar barely moves (~1 day at best) —
the week is integration/human time, and phases 1/4-integration can overlap
the GPU phases regardless.

## Scale of training: SFT epochs and RL steps

**Caveat that reframes both: all our 7B scale evidence is RE-warm-start
evidence, not from-scratch evidence.** v2/v3 SFT initialized from
`kitft/nla-qwen2.5-7b-L20-{av,ar}` — an already-trained NLA (trained AV,
trained AR value head). The celebrated "97% of the critic-gold ceiling in 1
epoch" (FVE 0.485 vs 0.498) was a format/objective *conversion* on top of
that. kitft's original from-scratch training regime (epochs/data) is not
recorded in this repo. From scratch at 27B, SFT must teach a vanilla
post-trained model to read injected vectors at all — an input mode with zero
pretraining prior — and the AR's value head is randomly initialized
(`prepare_critic_checkpoint.py` adds a fresh head).

**SFT: default 2 epochs per role, eval-gated in both directions.** One epoch
over ~220k rows may or may not be enough from scratch — nobody has measured
it. Save@500, track AR `fve_nrm` and end-of-epoch round-trip FVE vs the
critic-only gold ceiling: stop after epoch 1 if ≥~95% of ceiling; extend to
epoch 3 (+~$90) if still climbing at epoch 2. If FVE is still climbing at
epoch 3, that's the signal the 220k rows are the constraint → revisit Option
B (stage-2 regen for *scale*, not just richness). AV and AR are independent
in SFT — run them on two parallel 4×H200 boxes to keep wall-clock flat.

**RL: 300-step budget stands, but only because SFT convergence is a hard
gate.** The 7B plateau at ~step 110 (reward −0.507→−0.261, KL leash taut at
kl_loss ~2.5–2.9) is evidence about RL-on-a-converged-NLA — which is exactly
the state the SFT gate is required to produce. RL's job in this recipe is
front-loading/reshaping, not skill acquisition: the reward IS the co-trained
AR critic, so launching RL from an undercooked SFT means optimizing against a
noisy reward model while KL-anchored to a weak reference — more steps cannot
fix that; better SFT can. Policy: **do not launch Phase 5 until warm-start
FVE is at/near its ceiling**; budget 300 steps with the plateau stop-rule;
if reward is still climbing at 300, extend by checkpoint resume
(`--rollout-global-dataset` ON makes resume clean) in +100-step increments
(~$195 each) rather than pre-committing. If the plateau *level* disappoints,
the run-2 lever is a lower KL coefficient or more samples/prompt — not more
steps at the same KL.

## Base vs instruct init

**Instruct — decided partly by availability, and it's the right call anyway.**
`Qwen/Qwen3.5-27B-Base` does not exist (HF 404 with auth; the MoE
`Qwen3.5-35B-A3B-Base` is public, the dense 27B ships post-trained only).
Even given a choice: extraction, AV init, and AR init must all be the *same*
model (NLA is self-interpretation — the AV reads its own residual stream),
kitft + v2/v3 all built on Qwen2.5-7B-**Instruct** so instruct-init keeps the
7B→27B comparison clean, and the post-trained model is the scientifically
interesting target (it's the deployed artifact; Qwen's own interp SAE release,
`Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50`, likewise targets the post-trained 27B —
also a natural external comparison for NLA readouts later). Thinking-mode is a
chat-template behavior our custom templates never trigger, and 1 epoch of
warm-start fully stomped instruct style at 7B. Expect (and ignore) `<think>`
leakage only in the earliest SFT steps.

## Recipe choice

Replicate v3 (truncation U[1,120] tokens, KL 0.03, grad_guard, T=1 rollouts,
never-train-EOS AV, `--rollout-global-dataset` ON
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
2b. **From-scratch SFT undershoots and RL can't rescue it** — the reward is
   the co-trained AR critic and the KL anchors to the warm-start ref, so a
   weak SFT poisons both; mitigation is the hard Phase-5 gate + the epoch
   budget, never "more RL steps".
3. **Critic memory fit** — fallback topologies cost throughput, not
   feasibility.
4. **Instruct/thinking init** — same class of init as v3 (instruct model);
   residual risk is style leakage early in SFT, historically handled by 1
   epoch of warm-start.
5. **injection_scale surprise** — hybrid-model residual norms may be unusual;
   it's measured, not guessed, and the CJK smoke test catches silent failure.

Note: `Qwen3.6-27B` (Apr 2026) is the same architecture with a better
post-train; this plan applies to it unchanged if we'd rather target that.
