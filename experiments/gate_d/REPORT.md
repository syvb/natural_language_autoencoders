# Gate D — attention-output NLA feasibility

**Verdict: MARGINAL (pre-registered) — and, after a post-hoc control, the
positive headline claim is RETRACTED. No D3 AV-SFT probe, no RL funding.**

> **Post-hoc control (review-triggered, see §Post-hoc below):** a trivial
> arm whose label is the raw context tail (attention-blind) scores residual
> alignment 0.108 — beating every attention-evidence arm (best 0.072,
> armV−armT = +0.037 [+0.035, +0.039]) and beating the content control by
> +0.068, i.e. above the +0.05 PASS bar. The sign-flip this gate celebrated
> was a register artifact: the frozen-AV content control was simply a bad
> representation of the context (10.5% word overlap with the actual local
> text). No label family shows attention-specific verbalizable signal
> beyond raw context. The original analysis is preserved below for the
> record.

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

## Post-hoc control and retraction (added after adversarial review)

An adversarial experiment review (experiment_review.md) noted that armT's
advantage over armC was largest exactly where its label contained no
attention-specific content (local-only retrievals, +0.050) and smallest
where it quoted distant sources (+0.024; corr with retrieval distance
−0.18), and that the obvious control — a label that is just the verbatim
context tail, content-grounded but attention-blind — was never trained.
Running it (armV, same 8k positions/recipe, ~$1):

| arm | cos resid | contrast |
|---|---|---|
| armV (raw context tail) | **0.108** | vs armC **+0.068** [+0.066, +0.070]; vs armT **+0.037** [+0.035, +0.039] |

Conclusions revised accordingly:
- The "grounded labels beat content" sign-flip was **verbatim text beating
  hallucinated paraphrase**, not mechanical grounding. The T>H>Z>C ordering
  tracked verbatim-context content of the labels, not groundedness.
- Attention evidence is **net negative** relative to plain context text —
  annotating retrievals displaces context tokens that carry more signal.
- The residualized write remains substantially predictable from raw
  context (0.108 ≫ ridge-orthogonal 0); the linear v_pre ridge is a weak
  nuisance model, so "beyond-content" claims need the raw-context arm as
  the control, raising the effective bar from 0.040 to 0.108.
- A defense that survives: token-identity-only baselines predict ~0 of the
  residual, so critics do use context, not trivial memorization.
- The pre-registered MARGINAL/no-RL decision stands a fortiori.

**NLA self-test (final addendum 3 — the positive control).** The armV
lesson cuts both ways, so we ran the cat-control on the original paradigm
itself: critic on the frozen AV's explanations of v_pre vs critic on the
raw context tail, target = v_pre (states, not deltas; same positions,
recipe, and machinery as everything above). Result: explanation 0.872 vs
context 0.832 — paired **+0.0398 [+0.0380, +0.0417]**, explanation wins at
91% of positions. The flagship NLA's verbalizations DO carry state
information beyond the verbatim input text. Combined with every delta arm
losing to the same control, the states-verbalize / deltas-don't dichotomy
is now measured within one experiment, same controls, same metric — and
the strata analysis (strata_analysis.md) confirms no position subclass
(0/236) reverses the delta side, with armV dominance mechanistically
explained by local-copying mass (rho=0.48).

**Two-state injection probe (final addendum 2).** Injecting v_pre and
v_post as two adjacent embedding slots (letting the model difference
internally, both vectors in-distribution) produces coherent, well-grounded
reads — that are near-duplicates of the v_pre-only reads (same-position
content Jaccard 0.45 vs 0.10 cross-position; 0/50 change-descriptions).
The frozen AV merges the slots into one state reading; the marginal
information in the second vector leaves no trace. The interface is
mechanically viable (0/50 CJK) but exploiting it would require a
contrast-trained two-slot AV, whose payoff the oracle probe below bounds
at ~+0.004. Full run not spent (pair_smoke_review.md).

**Oracle ceiling probe (final addendum).** To bound what ANY label/RL
scheme could extract: armO = armV's verbatim context tail PLUS every
write-derived fact renderable as text (logit-lens top/bottom tokens of the
write, strength bucket, attention-evidence retrievals). Result: armO
0.1128 vs armV 0.1084 — oracle headroom **+0.0044** [+0.0030, +0.0058].
The text-expressible beyond-context information in an L20 attention-block
write is, under our best rendering, ~0.004 of residual alignment — two
orders of magnitude below the ~0.5 functional break-even. This kills the
residual-reward RL proposal for this object: reward shaping can re-aim the
gradient, but there is nothing meaningful at the top of the hill. (Caveat:
the oracle is not information-theoretically exhaustive — mid-stack logit
lens is noisy — but it is the natural mechanical superset, and it includes
the full attention evidence.)

## Recommendations

- Do not fund D3/RL from block-level attention writes.
- Any follow-up label experiment MUST include the raw-context arm as its
  control — the +0.05 bar over a frozen-AV paraphrase is trivially
  passable (armV does it with no model in the loop).
- The per-head-writes variant suggested by the original analysis is now
  weakly motivated: the armV result says the binding problem is that no
  tested register extracts write-specific information beyond context at
  all, not that block-level superposition dilutes it. If anything is tried
  next, the first question is whether ANY label can beat armV's 0.108 on
  the residual — e.g. context tail + the true next token the write
  promotes (oracle ceiling probe), before spending on realistic labels.
- MLP writes remain unexamined (mlp_out.npy is saved).

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
