# Gate D — attention-output NLA feasibility

**Verdict: MARGINAL (pre-registered). Mechanically-grounded labels beat the
content control — the first positive sign in this project — but the best
margin (+0.031) is below the +0.05 PASS bar after the one allowed
improvement cycle. No D3 AV-SFT probe, no RL funding from this gate.**

## Question and design

After Gate C established that every frozen-AV-derived label family fails to
verbalize layer diffs beyond content (REPORT.md there), Gate D tested the
component-output pivot on its strongest case: the attention block's
residual write at L20 of Qwen2.5-7B, where the attention pattern provides
mechanical, model-free ground truth for labels. Pre-registered plan in
PLAN.md; same critic-training machinery as Gate C (released L20 AR init,
direction-only MSE, matched n=8,000, doc-level splits, 15k positions,
seed 2).

## Gate results in order

**D0.2 content-confound ridge (v_pre → attn_out):** holdout R² = 0.401 →
CAUTION band (0.40–0.60). The write is about as linearly content-predictable
as the layer diff was (0.31). All downstream scoring used the residualized
write.

**D0.3 zero-shot probe:** frozen-AV reads of injected attention writes are
grounded ~3× over a shuffled-evidence null (re-scored on leak-free
evidence: full margin +0.0127, source-specific +0.0014, CIs > 0) → GO, but
weak in absolute terms; plain content reads of v_pre score 4× higher on the
same metric. Reads capture structural gist, confabulate specifics. CJK
smoke 4/50.

**D1 labels:** four arms — armC (content reads of v_pre), armT (templated
from per-head evidence), armH (Haiku 4.5 from evidence bundles, 15k batch,
15,000/15,000 succeeded, ≈$13), armZ (zero-shot AV reads). The labeler
prompt went through four pilot iterations; the subagent label review caught
a critical contamination (evidence context windows leaked ground-truth
future tokens into ~40% of labels) that was fixed structurally by
truncating windows at the position and rebuilding evidence from stored
attention rows. Final pilot: future-leak 1/100, numeric noise 1/100,
quoted-word groundedness 0.969. An adversarial code review (1 blocker,
4 major, all fixed pre-run) additionally forced the D0.3 re-score and the
pre-registration-conformant arm builder.

**D2 critic arms (holdout eval, residualized target):**

| arm | n | cos raw write | cos residual [95% CI] | Δ vs armC [95% CI] | verdict |
|---|---|---|---|---|---|
| armC content | 3,301 | 0.551 | 0.040 [0.033, 0.046] | — | baseline |
| **armT templated** | 3,318 | 0.569 | **0.072** [0.065, 0.078] | **+0.031** [+0.029, +0.033] | **MARGINAL** |
| armH Haiku | 3,315 | 0.564 | 0.064 [0.058, 0.070] | +0.024 [+0.022, +0.026] | MARGINAL |
| armZ zero-shot | 2,893 | 0.547 | 0.056 [0.049, 0.063] | +0.013 [+0.011, +0.015] | FAIL |
| armT′ enriched (cycle) | 3,318 | 0.556 | 0.050 [0.044, 0.056] | +0.010 [+0.008, +0.011] | FAIL |

All contrasts doc-clustered bootstrap (10k), Holm p ≈ 0. Beyond-armC
projection: armT 0.110, armH 0.099, armZ 0.069.

**Improvement cycle (spent):** armT′ aggregated source importance over all
28 heads (contribution-weighted attention) with distant-first rendering —
and *underperformed* armT by 3×. Local/self-attention mass dominates the
aggregate, drowning the selective heads whose contextual phrases carried
armT's signal. Selectivity beats coverage for this register.

## What this establishes

1. **Direction reversed vs Gate C.** Grounded (non-frozen-AV) labels train
   critics that extract beyond-content information; ordering is monotone in
   label groundedness (T > H > Z > C). This is direct support for the
   label-circularity diagnosis of Gate C — the bottleneck there was the
   label source, not (only) the target object.
2. **But the effect is small.** Absolute residual alignment ≤ 0.072
   everywhere — far below Gate C's 0.27–0.30 (the attention write's
   content-unpredictable part is mostly uncaptured by every register
   tried) and far below any functional threshold.
3. **Pre-registered outcome: MARGINAL.** The +0.05 bar was set as the
   minimum signal worth amplifying with the AV-SFT probe and an RL bet;
   +0.031 with a spent improvement cycle does not clear it.

## Recommendations

- Do not fund D3/RL from block-level attention writes.
- The natural next variant, if pursued: **per-head writes** (top
  contributing head's slice through W_O). The evidence here points at
  block-level superposition as the dilution mechanism — the top head
  carries only ~8% of the write, and the armT′ negative shows aggregate
  coverage hurts. attn_rows.npy + head_norms.npy (saved, HF) suffice to
  build per-head targets and evidence with no new extraction.
- MLP writes remain unexamined (mlp_out.npy is saved) but lack the
  mechanical grounding signal that made attention labels work at all.

## Incidents

- Silent client OOM: asyncio.gather pre-built ~2MB request bodies for 15k
  queued coroutines (30+GB RSS) → container killed clients with no
  traceback; latent in Gate C's gen2.py, exposed at Gate D scale. Fixed by
  building bodies inside the semaphore.
- pkill-cascade: killing gen clients released the wrapper's `wait`, whose
  teardown pkill'd the sglang server and echoed a false ALL_DONE. Monitors
  now trust file counts, not wrapper echoes.
- HF anonymous rate-limit + multi-pattern `hf download` mis-parse (Gate C
  repeat). Two cwd-reset path mistakes (lesson: absolute paths in every
  multi-step shell command).
- A subagent orchestrating two pods ended its turn prematurely;
  orchestration was absorbed inline. Trainings themselves ran clean.

## Costs (this gate)

D0 extraction+gen ≈ $11 · pilots ≈ $2 · armH batch ≈ $13 · D2 pods
(3 pods ≈ 5.3 pod-hours) ≈ $7.6 · armT′ cycle ≈ $1 → **≈ $35 total**
across the two caps ($50 D0–D1, $40 batch+D2; ≈$22 of the second cap).

## Artifacts

Code in this directory (extract_d, rebuild_evidence, gen_d, ground_score,
build_bundles_d, labeler_d, check_labels, build_arms_d, build_armT2,
evaluate_d, pod/*). Data public at
hf.co/datasets/syvb/nla-layer-diff-experiments under `gate_d/`: raw
activations + writes, full attention rows + head norms, evidence (v1 and
leak-free v2), bundles, all four label sets, 45k generated reads, held-out
predictions per arm, eval results. wandb: project nla-layer-diff, runs
arm{C,T,H,Z,T2}_8k. Checkpoints not kept (~11GB each, ~$2 to reproduce).
