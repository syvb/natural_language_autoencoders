# Gate D — can NLAs verbalize attention-block outputs? (plan, pre-registered)

Status: PLANNED. No spend yet. Proposed budget: **$75 total** (covers
everything through the critic gate + evaluation; the RL smoke test, if
funded, is a separate ~$300–450 decision).

## Motivation

Gate C closed the layer-diff line: every frozen-AV-derived label source
trained a critic no better than a content-only control at capturing the
content-unpredictable part of a layer diff (REPORT.md). The post-mortem
identified two structural causes, both specific to *diffs*:

1. **Re-representation** — a layer diff is mostly the stream re-encoding
   what is already there (cos(diff, v22)=0.86; ridge from v18 R²=0.31), so
   "what changed" is dominated by "what was already true".
2. **Label circularity** — the only ground truth we could manufacture
   routed through an AV never trained on the target object.

Attention-block outputs attack both: the block's `o_proj` write is the
increment a *computation* contributes (no identity path inside it), and the
attention pattern provides **mechanical, model-free ground truth** for
labels (which source positions, which tokens, which heads, with weights).

This gate asks: (a) is the attention write free of the content confound
that sank diffs, (b) can we manufacture verifiably grounded labels for it,
and (c) does a critic trained on those labels beat a content-only control
at predicting the write's content-unpredictable part? Pass → fund the AV
SFT + RL smoke. Fail → the cheap-NLA-variants program stops here.

## Setup constants

- Model: Qwen2.5-7B-Instruct; released L20 AV/AR pair
  (`kitft/nla-qwen2.5-7b-L20-{av,ar}`). L20 attention block is inside the
  validated L14–L24 transfer zone (Gate A).
- Note Qwen2.5-7B attention is GQA (28 query heads, 4 KV heads); patterns
  are per *query* head, head writes are the per-head slices through `W_O`.
- Corpus: FineWeb sample-10BT, fresh seed (2), 2,500 docs × 6 positions,
  `_MIN_POSITION=50`, `MAX_TOKENS=512`, doc-level train/holdout split
  (HOLDOUT_FRAC=0.2). Raw fp32 vectors everywhere (`norm="none"`).
- All scripts in `experiments/gate_d/`, large artifacts gitignored and
  published to hf.co/datasets/syvb/nla-layer-diff-experiments under
  `gate_d/`. Pods per the runpod memory recipe; HF_TOKEN on every pod
  (anonymous downloads get rate-limited — Gate C incident).

## Phase D0 — extraction + the two go/no-go probes (~$18, 1 day)

### D0.1 Extraction (1×A100, ~3 h, ~$5)

`extract_d.py` (extract2.py skeleton + hooks). Per position store:

- `attn_out.npy` — o_proj output at p (the residual write). Training target.
- `v_pre.npy` — residual state entering the block (input of the attn
  sub-layer, post-LN input NOT included; store the pre-LN residual state).
  For the confound ridge only.
- `mlp_out.npy` — MLP write at the same positions (free; descriptive arm).
- `attn_evidence.jsonl` — per position: per-query-head contribution norm
  (‖head h's slice through W_O‖ at p), top-5 source positions per head with
  weights, decoded source tokens ±3 tokens context. Keep only heads
  covering 90% of total contribution norm.
- `logit_lens.jsonl` — top-10 vocab tokens promoted / suppressed by
  `attn_out` alone (soft evidence; mid-stack logit lens is rough).
- `meta_d.json` (doc_idx/pos/split/token/context_tail), `doc_tokens.jsonl`.

Also record the norm distribution of `attn_out` vs residual states (needed
to sanity-check injection rescaling).

### D0.2 GATE: content-confound ridge (local CPU, $0)

Ridge `v_pre → attn_out`, train split, λ by inner doc split (evaluate.py
machinery). Pre-registered on holdout R²:

- **R² < 0.40 — GO.** The write is mostly not linearly content-predictable.
- **0.40–0.60 — CAUTION.** Proceed, but all downstream metrics use the
  residualized write (as Gate C did) and the bar for the critic gate stays
  at +0.05 over control.
- **R² > 0.60 — STOP.** Same disease as diffs; do not spend further. Write
  the negative result up; it is itself informative (attention writes
  largely redundant with their inputs at L20).

Either way, define `resid_attn = attn_out − ridge(v_pre)` on holdout: the
evaluation target for everything below.

### D0.3 GATE: zero-shot AV reads + groundedness (1×A100 + sglang, ~$10)

gen2.py infra (fast embed path): inject `attn_out` rescaled to
injection_scale=150 into the frozen L20 AV; 2 samples × all 15k positions,
temp 0.4. CJK smoke test first (50 positions) before the full run.

Groundedness score per read (mechanical, `ground_score.py`): fraction of
content words in the read found in {attended source contexts ∪ local
context tail ∪ logit-lens tokens}. Null distribution: same reads scored
against shuffled positions' evidence bundles (20 permutations).

- **GO** if mean groundedness beats the shuffled null with doc-clustered
  CI > 0 AND the *source-specific* part (overlap with attended sources
  EXCLUDING the local context tail) also beats its null — the second
  clause prevents "coherent paraphrase of nearby text" from passing.
- **Soft-fail** (coherent but ungrounded): proceed — arms T/H carry the
  warm start; arm Z is dropped from training arms.
- **Hard-fail** (CJK/garble at scale 150): debug injection scaling (try
  matching attn_out norm quantiles instead of 150) before any further
  spend; if two attempts fail, STOP and report.

## Phase D1 — label arms (~$15, 1 day, mostly batch wait)

Filter: positions where total contribution norm < 10th percentile are
labeled "quiet" (no meaningful write); keep a 5% quiet stratum in training
so the AV can learn to say so, drop the rest.

- **Arm C (control, ~$0 GPU reuse):** frozen-AV content explanation of
  `v_pre` (1 sample, temp 0.4, generated alongside D0.3). The content-only
  baseline — the arm0p analogue. Everything is judged as a margin over
  this.
- **Arm T (templated, $0):** deterministic rendering of the evidence
  bundle: sources, weights, dominant heads, promoted tokens. Perfectly
  grounded, rigid register.
- **Arm H (Haiku-written, ≈$12):** evidence bundle → 2–4 sentence concrete
  description. Prompt rules (Gate B/C lessons): every content claim must
  quote tokens present in the bundle; concrete tokens over abstractions —
  the abstract meta-description register trained worst and failed the KL
  judge in Gate C. Mechanical verification pass: labels with groundedness
  below threshold get one regeneration; still-failing labels dropped.
  Quality gate: ≥80% of labels must survive, else revise prompt once.
  **Batch submission requires explicit user approval (standing rule); a
  ~100-label pilot for user review first. No GPU running while waiting.**
- **Arm Z (zero-shot AV reads):** already generated in D0.3; included as a
  training arm only if D0.3 passed GO.

## Phase D2 — critic arms + evaluation (~$12 GPU, 1 day)

`train_critic.py` unchanged (`--targets attn_out.npy`); 2 epochs, lr 1e-5,
eff. batch 64, matched n = 8,000 per arm, PagedAdamW8bit, A100-80GB
(~40 min/run → one pod, 4–5 runs, ~$8; predictions exported per run via
`predict_critic.py`; **kill pod immediately after**).

Runs: C (control), T, H, Z (if alive), optionally H_full (slope).

Evaluation (evaluate.py generalized; local CPU):

- Primary, pre-registered: paired per-position
  Δ = cos(pred_arm, resid_attn) − cos(pred_C, resid_attn), shared holdout
  positions, doc-clustered bootstrap 10k, Holm over the (up to 3) arm
  contrasts.
  - **PASS** (any arm): CI lower bound > 0 AND Δ ≥ +0.05 → fund Phase D3.
  - **MARGINAL**: +0.02–0.05 → one label-improvement cycle, re-run D2 once.
  - **FAIL**: all arms within ±0.02 or below → stop; conclusion is
    "mechanically-grounded label family insufficient for attention writes
    at L20", which—unlike Gate C—*does* bear directly on verbalizability,
    since these labels do not route through a frozen AV.
- Causal judge (veto-only, ~$3 GPU, 500 holdout positions): mean-ablate the
  attention write at p (replace `attn_out` with the train-split mean),
  measure next-token KL vs natural = damage ceiling; then patch
  v_pre + predicted write (rescaled to true write norm) and measure KL.
  Sanity: patching the true write must give KL ≈ 0. Report recovery
  fraction = 1 − KL(pred)/KL(ablate). Veto any PASS whose recovery is
  significantly below the control arm's (paired Wilcoxon).
- Descriptive: mlp_out same-pipeline table (no gate); per-head subspace
  alignment of predictions (do critics recover the *dominant head's*
  direction?); best-of-N headroom on arm Z reads.

## Phase D3 (only if D2 passes) — AV SFT probe (~$12, half a day)

SFT the AV (injection-token fine-tune, stage-2 recipe) on the winning
arm's pairs; generate reads on holdout; score groundedness (D0.3 metric)
and critic-reconstruction alignment vs the zero-shot reads. PASS =
SFT reads beat zero-shot reads on source-specific groundedness with CI>0.
Then the decision package goes to the user: **fund the ~$300–450 RL smoke
test** (8×H100, 10–15 h, Miles stage-3 with `--force-use-critic`), with
kill criteria pre-registered at funding time (smoke must push residual
alignment ≥ +0.05 over the D2 control within budget, causal judge
non-worse, else stop permanently).

## Cost & timeline summary

| phase | what | cost | wall clock |
|---|---|---|---|
| D0 | extract + confound ridge + zero-shot probe | ~$18 | 1 day |
| D1 | label arms + verification (Haiku batch gated on user) | ~$15 | 1 day (batch wait) |
| D2 | 4–5 critic runs + evaluation + causal judge | ~$12 | 1 day |
| D3 | AV SFT probe (only if D2 passes) | ~$12 | 0.5 day |
| | contingency (reruns, flaky hosts) | ~$18 | |
| | **total budget ask** | **$75** | **3–4 days** |

Stop-loss structure: $18 buys the two structural answers (D0.2, D0.3);
$33 buys the label-quality answer; $45 buys the full critic verdict. Each
phase can kill the gate before the next dollar.

## Risks & honest priors

- **Confound returns** (D0.2 > 0.6): plausible — attention at p is
  query-driven by v_pre. This is exactly why D0.2 runs first. Prior: ~25%.
- **Block-level superposition**: 28 heads' writes summed may be
  polysemantic mush even when individual heads are clean. Mitigation
  already in the data: per-head evidence + contribution norms; fallback
  experiment is top-head-only writes (head slice through W_O, still in
  d_model). Prior that block-level reads are too muddy: ~30%.
- **Injection OOD**: attn_out direction distribution differs from
  residual-state distribution; rescaling fixes magnitude, not direction.
  D0.3 hard-fail branch covers this.
- **Positional/syntactic heads**: much of the attention budget does
  bookkeeping (BOS sink, previous-token, induction). The quiet-stratum +
  templated register handles "boring write" honestly; expect a sizable
  boring class — that is a finding, not a failure.
- Overall prior that D2 passes: ~30–35% — better than the diff line ever
  was, because the label family is grounded and the confound is gated
  up front.

## Infra reuse map

extract2.py → extract_d.py (new hooks); gen2.py (unchanged infra, new
vectors); hybrid_v2.py → labeler_d.py (new prompt, same batch plumbing);
build_arms.py → build_arms_d.py; train_critic.py / predict_critic.py
(unchanged); evaluate.py (target swap); kl_judge.py → ablate_judge.py
(mean-ablation null instead of zero-diff). Footguns: see Gate C STATE.md
footgun index — all still apply (hf CLI patterns, HF_TOKEN on pods,
setsid/nohup/disown, never pkill your own ssh, kill pods immediately).
