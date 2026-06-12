# Gate E — counterfactual-pair deltas: a warm start with true labels (plan, pre-registered)

Status: PLANNED. No spend yet. Proposed budget: **$60** through E3
(E4 transfer gate budgeted separately if E2/E3 pass). Repo conventions as
before: everything committed, large artifacts on the HF dataset, pods
killed immediately, batch submissions user-approved after a pilot.

## Where this comes from (program state)

Five gates (A–D + addenda) established, with controls:

- **States verbalize.** The released L20 AV's explanations beat the
  verbatim-context control at reconstructing the state itself
  (+0.0398 [+0.0380, +0.0417], explanation wins at 91% of positions — the
  "NLA self-test"). The paradigm's core claim survives its own cat-control.
- **Deltas don't.** Every label family for layer diffs (Gate C) and
  attention writes (Gate D) lost to trivial controls: frozen-AV labels
  were content-parasitic; mechanically-grounded attention evidence lost to
  the raw context tail (armV 0.108 vs best evidence arm 0.072); 0/236
  position strata reversed this; oracle headroom (context + all renderable
  write facts) was +0.004 overall.
- **Why deltas fail (measured, not argued):** (1) *plumbing dominance* —
  most of a natural delta is local copy/bookkeeping whose optimal label is
  a context substring (armV's advantage scales with local-attention mass,
  ρ=+0.48); (2) *wire-format gap* — genuine retrieval payload exists
  (oracle headroom concentrates 4×, +0.017, in pure-distant strata) but
  decoding it requires the downstream weights' codebook, which text labels
  can't teach a critic; (3) *precision residue* — what remains is
  high-dimensional, low-norm, incompressible in any nameable basis;
  (4) *deferred meaning* — much of a write's effect is realized at later
  positions (horizontal path), so position-local effect labels are blind
  and confounded by self-repair.

Gate E attacks (1) and (2) at the data-construction level instead of the
labeling level: **manufacture deltas whose descriptions are true by
construction.** A counterfactual pair — context X and a minimally edited
X′ — yields delta δ = v(X) − v(X′) at aligned positions. Shared plumbing
cancels in the subtraction; the label (the edit, plus its observed
behavioral consequence) is ground truth because we caused it. Note the
object-class honesty: δ is a *state–state* difference (inside the class
that verbalizes), adopted as the place to learn a difference-register with
true labels. Whether natural depth-deltas project onto anything learned
here is deferred to a separate transfer gate (E4).

## Methodological laws carried forward (binding on this gate)

1. **The control is `cat`.** Every contrast is against a context-only arm;
   a +bar over any weaker control is void (Gate D retraction).
2. **Oracle probe before training spend.** Bound the ceiling with
   privileged-information labels before buying optimization (armO lesson).
3. **Verbatim beats paraphrase.** Labels quote exact spans; no LLM
   prettification of evidence (armT > armH +0.0072; Gate C arm0p > armB).
   LLM text generation is used only where observation must be summarized
   (continuation-divergence descriptions), never to restyle known facts.
4. **Mechanical label verification, structurally enforced.** All label
   text derives from data we control; the Gate D future-leak class of bug
   is checked for explicitly (no label may contain tokens from beyond the
   measurement position of its own sequence).
5. **Selectivity over coverage** in any evidence rendering (armT2 lesson).
6. **Pre-registration discipline**: doc-level splits, matched n,
   doc-clustered bootstrap (10k), Holm across contrasts, thresholds fixed
   below. Headline contrasts run with **2 seeds** this time (all prior
   orderings rested on single seeds; flagged by the experiment review).
7. Engineering: HF_TOKEN on every pod; request bodies built inside the
   semaphore; no pkill patterns that match the issuing command; absolute
   paths in multi-step shell; file-count monitors, never wrapper echoes;
   verify file presence before launching; kill pods immediately.

## Phase E0 — data construction (~$14, 1–2 days)

### E0.1 Edit generation (Haiku batch, ≈$6; pilot → user approval)

Source: ~2,500 fresh FineWeb docs (seed 3), 512-token truncation. For each
doc, Haiku proposes 4–6 minimal edits subject to HARD constraints checked
mechanically post-hoc (rejects regenerated once, then dropped):

- **Token-length-preserving** under the Qwen tokenizer (edit span maps to
  equal token count), so the suffix token sequences are identical and
  positions align exactly.
- Single-site, ≥50 tokens before doc end (room for downstream positions).
- Typed, with quotas per doc to enforce diversity:
  - `role_reversal` (≥1 per doc where syntax allows): "Alice paid Bob" →
    "Bob paid Alice". Identical token bag — the delta is purely
    relational. **Gold class**: zero lexical shortcut available.
  - `antonym`: won→lost, rose→fell, before→after (polarity with
    length-safe single-token swaps preferred).
  - `attribute`: numbers, dates, quantities ("$4.6M"→"$46M").
  - `entity_swap`: only where the entity is re-referenced later in the doc
    (checked by string/coref-lite match) so the delta is live downstream.
  - `discourse`: certainty/hedging swaps ("will"→"may"), assertion↔report
    ("X is"→"X reportedly is" only if length-preserving).
- Label fields stored per edit: type, source span, target span, char/token
  offsets.

### E0.2 Extraction + divergence selection (1×A100, ~$6)

For each (X, X′): one forward pass each, capturing hidden_states[20] (and
hidden_states[16], [24] stored for free — layer-robustness checks later)
plus next-token logits at every shared position after the edit.

- **Measurement positions**: positions p > edit_end with
  KL(run_X(p) ‖ run_X′(p)) ≥ τ (τ set so median doc yields 2–3 positions;
  report the yield curve). This guarantees live deltas (kills the
  dead-pair failure mode) and stratifies by distance-from-edit.
- Store: δ = v_X(p) − v_X′(p) raw fp32 (norm="none" invariant), both
  states, per-position KL, distance-from-edit, and 2 sampled continuations
  per side at the position (temp 0.7, 24 tokens) — the raw material for
  consequence labels.
- QC gates (mechanical): alignment assert (identical suffix token ids);
  ‖δ‖ distribution vs within-run state-noise floor (re-run X twice to
  measure numerical noise; require pair-delta p10 ≫ noise p90); per-type
  delta-norm table.

### E0.3 Labels (local, ~$2 Haiku for consequences only)

Per (pair, position), THREE label registers, all mechanically anchored:

- `L_edit` (verbatim, $0): "changed '<source span>' to '<target span>'" —
  template, exact quotes, law 3.
- `L_conseq` (Haiku from observed behavior): shown the four sampled
  continuations (2 per side), writes 1–2 sentences on how the expected
  continuation differs at this position. Mechanical check: every content
  word must appear in the continuations or context (Gate D checker
  adapted); future-of-X tokens beyond the position are PERMITTED here only
  insofar as they appear in the sampled continuations (they are observed
  model behavior, not corpus leakage — this distinction documented).
- `L_full` = L_edit + L_conseq.
- **Symmetry augmentation**: every pair contributes (δ, forward label) and
  (−δ, reversed label) — doubles data and bakes in the antisymmetry prior.

Targets: ~25–35k labeled deltas; doc-level train/holdout split (0.8/0.2);
matched-n arms per Gate-D machinery.

## Phase E1 — zero-shot probe, no learned map (~$10, half a day)

Inject δ (rescaled to injection norm — the dumb interface, deliberately)
into the **frozen** L20 AV; 2 samples × all holdout positions (+1k train
positions for inspection). Metric is judge-free: **edit-span hit rate** —
fraction of reads containing the target span (and separately the source
span), vs a shuffled-pair null (reads scored against other pairs' spans).

Pre-registered prediction (falsifiable): plumbing cancellation should make
these reads MUCH more grounded than Gate B/D natural-delta reads.

- **GO**: target-span hit rate ≥ 5% absolute AND ≥ 3× the shuffled null
  (CI > 0, doc-clustered). (Gate D attn reads managed only ~3× null at
  0.2–0.4% absolute on a softer metric — we require an order of magnitude
  better in absolute terms.)
- **Soft-fail**: coherent but ungrounded → E2 proceeds (the adapter exists
  precisely to fix interface mismatch); record as evidence the interface,
  not the object, is the bottleneck.
- **Hard-fail**: garble/CJK → debug injection once, else stop the gate.

## Phase E2 — adapter-only: frozen AV + learned linear map (~$12, 1 day)

Train affine W: ℝ³⁵⁸⁴→ℝ³⁵⁸⁴ (also a rank-256 variant; init = scaled
identity; raw δ in — NO pre-rescaling, magnitude is signal) mapping deltas
to the AV's injection slot. AV entirely frozen; loss = next-token CE on
`L_full` labels. 2 seeds.

**Held-out edit types are the pass condition** (the shortcut concern: W
trained on lexical swaps learns embedding arithmetic): train on
{antonym, attribute, entity_swap, discourse}, evaluate on `role_reversal`
(and a second rotation swapping which type is held out).

Pre-registered criteria on held-out types, generated reads:
- **PASS**: target-span hit ≥ 25% (null ≤ 2%) AND antisymmetry — among
  pairs read correctly forward, ≥ 50% flip correctly under −δ via W.
- **MARGINAL**: hit 10–25% or antisymmetry 25–50% → one iteration on W
  capacity (2-layer MLP, k-slot output W: 3584→k×3584), then stop either
  way.
- **FAIL**: < 10% held-out hit → the counterfactual line ends; write-up
  states whether E1 vs E2 localized the failure (interface vs linearity vs
  object).

Critic-side companion (runs in parallel, Gate-D machinery unchanged,
~$4): critics on δ targets with arms `L_edit` / `L_full` / **cat-control**
(context of X only) / **shuffled-edit control** (context + wrong edit
description) / **oracle** (both contexts verbatim). Pre-registered:
`L_full` must beat cat by ≥ +0.05 (CI>0, Holm) — expected to pass easily
since cat cannot know the edit; if it does NOT, the divergence-selection
or extraction is broken (this is a sanity gate, not a discovery gate).
The interesting number is `L_full` vs oracle: how much of the
both-contexts information the text label compresses.

## Phase E3 — full SFT with adapter (conditional on E2 ≥ MARGINAL, ~$10)

Unfreeze the AV (LoRA or full FT, budget-dependent) jointly with W on
`L_full`; same held-out-type evaluation; the E2→E3 gap measures
nonlinearity of the delta→readable mapping. PASS bar as E2. Output
artifact: a **difference-verbalizer** checkpoint + adapter, published.

## Phase E4 — transfer to natural deltas (separate gate, pre-register then)

Only sketched here; own plan + budget if E2/E3 pass. Inject natural
depth-deltas (v22−v18) and attention writes through the trained W + AV;
score against the Gate D control stack (cat, oracle) on the SAME 15k
corpus (all artifacts retained on HF). Honest prior: this is where the
line most likely dies — natural deltas may not lie in the span of
edit-direction deltas (re-representation findings). A clean death here
is a measured boundary: "deltas verbalize iff they are difference-of-state
shaped," which would be a real characterization, not a shrug.

## Decision summary

| phase | kill condition | pass action |
|---|---|---|
| E0 | yield/QC fails (alignment, dead pairs) | E1 |
| E1 | garble after one debug round | E2 regardless of soft-fail |
| E2 | held-out hit < 10% after one capacity iteration | E3 |
| E3 | held-out hit < 25% | E4 plan + publish difference-verbalizer |

## Costs & timeline

| phase | cost | wall clock |
|---|---|---|
| E0 edits + extraction + labels | ~$14 | 1–2 days (batch waits) |
| E1 zero-shot probe | ~$10 | 0.5 day |
| E2 adapter ×2 seeds + critic arms | ~$12 | 1 day |
| E3 full SFT (conditional) | ~$10 | 0.5 day |
| contingency | ~$14 | |
| **total ask** | **$60** | **~4 days** |

## Risks & priors

- Haiku can't reliably produce token-length-preserving edits → mitigated
  by mechanical filtering + one regeneration round; worst case restrict to
  single-token swap types (antonym/attribute), losing role_reversal — a
  real loss, flag if it happens. Prior of significant yield problems: 30%.
- Deltas dominated by lexical embedding-difference (shortcut) → held-out
  role_reversal is the detector; prior that lexical shortcut explains
  most of E2's headline number: 40% — which is exactly why the held-out
  structure is the pass condition.
- δ noise at distant positions (KL threshold too low) → noise-floor QC.
- E1 prediction outright wrong (counterfactual deltas unreadable even
  in-principle) → the program's "states verbalize" claim gets a useful
  boundary: state DIFFERENCES are already outside the readable class.
  Prior: 25%.
- Overall prior E2 passes on held-out types: ~35–40%. Higher than any
  previous gate — because for the first time the labels are true.

## Infra reuse

extract pipeline (extract_d skeleton, two-run variant), gen_d fast-embed
client (+ W applied client-side before injection in E1/E2 read-out),
train_critic/predict_critic unchanged, evaluate_d contrast machinery,
check_labels adapted, ground/span-hit scorer new but trivial, pod scripts
per Gate D pattern. New code: edit generator + validator, pairwise
extractor, adapter trainer (small; single-GPU PyTorch, no Miles).
