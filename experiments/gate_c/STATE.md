# Gate C — full state + remaining work (handoff doc, 2026-06-11)

Working notes for resuming after context compaction. Everything below is
the source of truth; verify against files, not memory.

## Where things stand

**Budget:** $60 total for Gate C. Spent ≈ $21 (4-pod generation fleet ~$16,
label smokes/pilots ~$3, misc pods ~$2). All RunPod pods TERMINATED (verify:
`curl -s -H "Authorization: Bearer $(cat ~/.runpod_key)" https://rest.runpod.io/v1/pods`
should show []). No GPUs running. Remaining budget ≈ $39; projected need:
batch $11 + session-2 GPU ~$10-12 → fine.

**Layout:** everything lives in this repo. `~/gate_a|b|c0|c` are symlinks
into `experiments/`. Large artifacts gitignored (`experiments/.gitignore`)
and published publicly at hf.co/datasets/syvb/nla-layer-diff-experiments.
Python venv with anthropic+huggingface_hub SDKs: `~/.gate_c_venv`
(symlinked as `experiments/gate_c/venv`). Keys: `~/.anthropic_key` (primary
for API; fallback OpenRouter `~/.openrouter_key`), `~/.wandb_key` (project
`nla-layer-diff`), `~/.hf_token` (user syvb), `~/.runpod_key`.

**Git:** branch `layer-diff-gate-a`, remote `fork` =
github-nla:syvb/natural_language_autoencoders.git (ssh alias in
~/.ssh/config). HEAD ≈ c1bbebd. Commit style: emoji prefix + Co-Authored-By
Claude trailer. Commit early & often — user wants everything in the repo.

**Session-1 data (COMPLETE, all local in experiments/gate_c/):**
- 15,000 positions (2,500 FineWeb docs × 6, seed 1, layers 18/20/22),
  doc-level split in `meta2.json` (`split`: train/holdout, ~12k/3k).
  `acts18/20/22.npy` [15000,3584] fp32 raw; `diff_targets.npy` = v22−v18;
  `doc_tokens.jsonl` = token IDs per doc (for the KL judge — no
  re-streaming needed).
- `gen2_out/*.jsonl` (deduped): expl18/expl22 (15k each, temp 0.4), diff1
  (15k, temp 0.4), hybrid_reads (32k = 4×8k train positions, temp 1.0),
  eval16 (2,400 = 16×150 holdout positions, temp 1.0). Rows {i,s,
  explanation,raw}.
- Arm pair files (from `build_arms.py`): `arm0p_train.json` (8k, input =
  "Earlier: <expl18>\n\nLater: <expl22>"), `arm0p_eval.json` (2,952),
  `armA_train.json` (8k diff1 texts), `armA_eval.json` (2,908),
  `armA_full_train.json` (11,756). Pair rows {text, tidx} where tidx
  indexes diff_targets.npy. CJK/garble-filtered.
- `hybrid_inputs.json`: 8,150 evidence bundles (8k train + 150 holdout;
  holdout ones use eval16 reads s0-3).
- SFT dry-run PASSED on A100-80GB: PagedAdamW8bit fits, wandb logs, ckpt
  saves (`train_critic.py`, pilot run eval_cos 0.61 = sanity).

**Labeler (`experiments/gate_c/hybrid_v2.py`):** PROMPT = v3 (user-authored:
no verdict, 3 snippets ≤50 words, confidence-then-importance order, sparing
source-only quotes allowed). PROMPT_V2_COMPOSED kept in code for rollback.
Quote post-strip DISABLED for v3. Modes: direct/submit/fetch (Message
Batches, model claude-haiku-4-5). v3 30-pos pilot: hybrid_pilot30_v3.json;
comparison vs v2 in prompt_v3_comparison.md (verdict: adopt v3, tweaks
suggested: enforce verbatim-source-only quoting; require diff-read support
for direction-of-change claims; both NOT applied — user's call).

## CRITICAL GATE: full batch NOT submitted

User requires explicit approval before submitting the 8,150-label batch.
(One batch msgbatch_018c7dVYPTKPitmmDjxdHgPw was submitted by mistake and
FULLY CANCELED at $0/8,150 canceled.) When approved:

```bash
cd /home/debian/nla-layer-diff/experiments/gate_c
./venv/bin/python hybrid_v2.py submit --inputs hybrid_inputs.json --out hybrid_labels.json
# poll: ./venv/bin/python hybrid_v2.py fetch --batch-id <id> --inputs hybrid_inputs.json --out hybrid_labels.json
```
≈$11 at Haiku batch pricing, usually <1h. NO GPU should be running while
waiting. Then `python3 build_arms.py armB` → `armB_train.json` (8k). Arm B
holdout eval inputs: hybrid labels exist for the 150 eval16 holdout
positions (in hybrid_labels.json keyed by position) — build
`armB_eval.json` from those {text=difference, tidx=int(pos)} (small: n=150;
add this to build_arms.py armB section).

## Remaining plan (pre-registered; full detail)

### Session 2 — critic training (1-2 pods, A100-SXM/PCIe, ~$10)
Use runpod REST recipe in memory file `runpod-experiment-workflow`. Image
lmsysorg/sglang:latest (just for torch), pip bitsandbytes wandb. Push:
train_critic.py, nla_inference.py (gate_a copy has return_dict patch),
models/ar (hf download kitft/nla-qwen2.5-7b-L20-ar), pair files,
diff_targets.npy. Launch pattern: `nohup setsid bash ... & disown`,
verify survival 90s. NEVER pkill pattern matching the launch string.

Runs (each ~40-60 min on 1×A100; parallelize across 2 pods):
1. arm0p_8k  — content control (PRIMARY baseline)
2. armA_8k   — diffav labels
3. armB_8k   — hybrid labels (after batch)
4. armA_full — 11,756 (data-size slope)
5. arm0p_res + best-diff-arm_res — residual-target variants: target =
   cross-fitted ridge residual (see below), only if time/budget allow;
   minimum viable = runs 1-3.
All: `--eval-pairs <arm>_eval.json --eval-targets diff_targets.npy`,
run-names like `armA_8k`, ckpts to ckpt_<arm>/. wandb project nla-layer-diff.
After each run save held-out PREDICTIONS: add a small predict script (load
ckpt via NLACritic pointing at ckpt dir — note value_head.safetensors is
saved by train_critic.py; nla_meta.yaml must be COPIED from models/ar into
each ckpt dir for NLACritic to load it) → preds_<arm>.npy [n_eval, 3584]
fp16 + the eval tidx order. Pull preds + RESULT.json, kill pods.

### Evaluation (local CPU, then ~20 min GPU for KL)
- Load preds, diff_targets, meta2 splits. Primary contrast: for each diff
  arm X ∈ {A, B}: paired per-position Δ = cos(pred_X, residual) −
  cos(pred_0p, residual) on shared eval positions.
- Residual definition (cross-fitted, n now sufficient): ridge v18→d fit on
  train docs (dual form, λ sweep like gate_c0 but expect R²>>6% at 12k),
  residualize held-out d. ALSO simpler robustness view: partial alignment
  beyond arm0p — residualize d against arm0p's own prediction per position
  (project out pred_0p direction) and compare arms on that target.
- Stats: doc-clustered bootstrap (cluster = doc_idx from meta2), 10k
  resamples, percentile CIs; Holm over the 2 arm contrasts.
- Decision rules (PRE-REGISTERED): PASS = CI lower bound > 0 AND point
  ≥ +0.05 over arm0p (→ fund ~$300 RL smoke). MARGINAL = +0.02..+0.05 or CI
  straddles → no RL; one generation-improvement cycle only. FAIL = within
  ±0.02 → conclusion limited to "frozen-AV-derived label family exhausted",
  NOT "diffs resist verbalization".
- KL judge (tie-breaker only; veto only if significantly WORSE, paired
  Wilcoxon, 500 holdout positions): adapt gate_c0/bo_pod.py klpatch —
  but SIMPLER now: doc_tokens.jsonl has token IDs (no fineweb re-stream;
  rewrite the doc-rebuild block to read doc_tokens.jsonl; keep the
  cos≥0.999 verification + zero_diff + _sanity_true_diff candidates).
  Hook gotcha (fixed in bo_pod.py): newer transformers decoder layers
  return bare tensors not tuples. Candidates: pred_0p, pred_A, pred_B
  (each scaled to true ‖d‖ per position — uniform across arms).
- Best-of-16 (DESCRIPTIVE only, not a gate): eval16 reads, AR-reconstruct
  via the TRAINED armA critic? No — select with trained critic, score with
  frozen-AR residual metric + KL (selection-bias control: select on
  permuted targets). Reuse gate_c0 patterns.

### Write-up & publish
- experiments/gate_c/REPORT.md in the style of gate_a/gate_b/gate_c0
  reports: design (incl. adversarial-review changes + half-scale
  rationale), per-arm table (eval cos vs raw d AND vs residual, CIs),
  paired contrasts + decision verdict, KL table, best-of-16, worked
  examples (delegate example-mining to a subagent like gate_c0), incidents
  (sglang client bottleneck → fast embed path + 8-way sharding + 4-pod
  fleet ~9.5/s; flaky H100 host avoided), costs.
- Upload new artifacts to HF dataset (hybrid_labels.json, preds, ckpt of
  the best critic — ckpts ~10GB, upload only if user wants), update HF
  README. Commit everything (small files) + push. Update memory files
  (gate-b-diff-label-results.md has Gate C0 notes; add gate C outcome).
- Tell the user the verdict against the pre-registered rules and the
  fund-RL-vs-pivot recommendation (pivot candidate: attention-output NLAs).

## Key numbers for context (from earlier gates)
- Frozen-AR ceilings: diffav residual alignment ~0.23 (information ceiling
  of diffav texts per ensemble probe), content-explanation→raw-diff cos
  ~0.6, functional break-even residual alignment ~0.5 (gate_c0 KL geometry).
- Expected outcomes: ~55% fail/marginal, ~35% modest pass (0.30-0.35),
  ~10% strong pass (≥0.4).

## Open user-decision items
1. Submit the 8,150-label batch with v3 prompt as-is, or apply the two
   comparison tweaks first? (AWAITING USER)
2. v-magnitude field: deferred.
3. Upload trained critic ckpts to HF? (ask when they exist)

## Footgun index (hard-won)
- ssh+pkill self-match; nohup-setsid-disown; never live-edit running bash
  scripts; huggingface-cli is a no-op in hub 1.x (use `hf`); transformers
  ≥4.47 apply_chat_template returns dict (nla_inference patched copy in
  gate_a/); decoder-layer hook returns bare tensor; sglang input_embeds
  client must precompute prompt embeds (gen2.py fast path) and shard
  client processes; A100 hosts in US-MD-1/CA-MTL-3 known good, one H100
  HBM3 host in US-MO-1 was flaky (two distinct CUDA faults).
