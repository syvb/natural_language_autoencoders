# Gate E — counterfactual-pair deltas (E0–E2)

**Verdict: FAIL at E2 (pre-registered: held-out span/keyword hit < 10% →
the counterfactual warm-start line ends). The failure is precisely
localized — and that localization is the gate's real finding.**

## What was built (E0 — all healthy, all public)

- 6,718 validated counterfactual pairs from 15k Haiku-proposed edits (45%
  mechanical yield): token-length-preserving, suffix-aligned (≥60 tokens),
  typed (antonym 2,193 / attribute 2,071 / role_reversal 1,194 /
  discourse 688 / entity_swap 572 at pair level).
- 15,480 deltas δ = h20(X) − h20(X′) at divergence-selected positions
  (median between-run next-token KL **0.48** — the deltas provably carry
  large behavioral information). Layers 16/24 deltas + continuation
  samples stored. hf.co/datasets/syvb/nla-layer-diff-experiments gate_e/.

## The three-step localization

**E1 — zero-shot (dumb interface):** target-span hit **0/4,196**; reads
overlap their own document at only 1.44× chance. The plumbing-cancellation
prediction was exactly backwards: subtracting two near-identical states
cancels the *AV-readable* content (topic/format/register), leaving only
edit-directions the frozen AV has no training to read. (Injection-scale
sweep needed: 24% CJK at 150, 10% at 60.)

**Linear-probe positive control (CPU, $0):** a ridge probe on the raw
deltas retrieves the edit's distinguishing keyword at **60.8% top-1 /
82.8% top-5** over 36 classes (chance 2.8%) on held-out documents; deltas
sharing an edit word cohere (cos 0.100 vs 0.027 across edits). The
information is present AND linearly accessible.

**E2 — learned affine adapter W into the frozen AV (2 seeds):** training
works mechanically (loss 2.5→2.0; the AV adopts the `changed "X" to "Y"`
register and occasionally emits correct *gist* — e.g. "changed 'positive'
to 'negative'" for a hide→expose flip). But content fidelity:

| eval | exact-span hit | keyword hit (probe-comparable) |
|---|---|---|
| in-distribution lexical (held-out docs) | 0.000 | **0.020** vs null 0.006 |
| held-out type (role_reversal) | 0.000 | 0.069 vs null 0.060 (≈chance) |
| antisymmetry | n/a (0 forward hits) | — |

Seed 1 weaker still (in-distribution keyword 0.4%). Pre-registered bars
(25% PASS / 10% MARGINAL) missed by an order of magnitude-plus.

## The finding

A linear map reads the edit out of the delta at 61%; the same delta,
linearly placed anywhere in the frozen AV's input space, makes the AV
*say* the edit at ~2%. **The wall is the reader's expressive path, not the
vector:** a frozen NLA verbalizer can only verbalize directions it was
trained to verbalize, and no input-side linear repair exposes new
directions to it. You cannot show a frozen reader new words by rotating
the input. This sharpens the Gate D "wire-format/codebook" mechanism into
a clean statement with a positive control on one side (the probe) and a
controlled negative on the other (W).

Combined with the E1 inversion (cancellation removes exactly what the AV
can read), the counterfactual program's premise — true labels would fix
delta verbalization — is answered: labels were never the binding
constraint at the reader. The binding constraint is that *reading new
semantics requires training the reader*, which is E3 (unfreezing the AV)
— and per pre-registration, FAIL at E2 does not unlock E3 spend.

## If anyone picks this up later

- E3 (LoRA/full-FT of the AV on these labels) is the direct next rung;
  the E0 dataset and the probe ceiling (61%/83%) are the ready-made
  benchmark. The honest prior improved by this gate: the information is
  linearly there; a *trainable* reader plausibly reaches it. What failed
  is specifically the frozen-reader shortcut.
- The probe itself is a working "edit decoder" — 60% top-1 from a single
  ridge — which may be useful standalone (e.g., delta-based edit
  detection) without any language model in the loop.

## Costs

Edits batch ~$3, pilots ~$1.5, E0 pod ~$5, E1 ~$2.5, E2 pod ~$2 →
**≈$14 of the $35 cap**. All pods terminated.
