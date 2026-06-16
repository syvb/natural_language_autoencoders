# Gate G — decoder-conditioned diff verbalization

**Idea (user's):** reconstruct a later state from *both* an injected earlier
state and the AV's text. Conditioning the critic on the earlier state moves
"subtract the redundant part" into the decoder's representation space, so the AV
reads an **in-distribution state** and the text only has to carry what the
earlier state doesn't already determine (the layer diff). Fixes Gate E's two
failure modes (OOD δ to the AV; frozen reader).

## Headline

1. **The mechanism works.** A conditioned critic reconstructs the target better
   with the AV's description than without — beating the no-text, raw-context, and
   shuffled-text controls, all p<1e-4.
2. **Diff-specific verbalization is a big-gap *regime*, not a smooth ramp.** At
   small gaps (cond=h20→h22/h24) the after-state description adds ≈0 beyond a
   before-state description; at the large gap (h16→h24) it adds **+0.0128 (82% of
   positions, p<1e-4)** — a real, significant diff signal from the *frozen* AV.
3. **Training the AV does NOT beat the frozen ceiling.** Best-of-N + SFT
   (cheap RL proxy) reaches 0.8224 vs frozen 0.8223 (Δ −0.0001, ns). The frozen
   AV's zero-shot description is already near the ceiling of the reconstruction
   objective; the BoN "gain" was reward-hacking that didn't survive a fresh critic.

## Setup

Conditioned critic = released L20 AR + LoRA(q/k/v/o) + trained value head; the
earlier state injected at a marker token (inj_scale 150); direction-only MSE.
15k FineWeb positions (gate_c), doc-level split 12030/2970. Arms differ only in
the text fed alongside the injected condition: **none** / **cat** (raw context) /
**text** (AV of target) / **scond** (AV of condition) / **textshuf** (target
text, wrong position). All bootstraps doc-clustered, 10k resamples.

## Stage 0 — gap 2 (cond h20 → target h22)

| arm | cos | vs text |
|---|---|---|
| text 0.8812 · scond 0.8798 · none 0.8673 · cat 0.8624 · textshuf 0.8412 | | |

| contrast | Δcos (CI) | reads as |
|---|---|---|
| text−none | +0.0138 [.0131,.0146] | total description value |
| text−cat | +0.0187 [.0175,.0200] | beats raw-context control (cat<none: raw ctx hurts) |
| text−textshuf | +0.0400 [.0390,.0410], 98.5% | position-specific, not a prior artifact |
| **text−scond** | **+0.0013 [.0008,.0018], 54%** | **diff-specific — tiny** |

State-legibility (scond−none) = +0.0125 is 91% of the win: the AV description
mostly helps the critic *read the lossy single-token injection*, even when it
describes the before-state. AV(h20)≠AV(h22) (Jaccard 0.37) so not an
identical-text artifact — the frozen AV just doesn't usefully verbalize a 2-layer
diff.

## Scaling curve — does a bigger gap help? (text−scond)

| gap | cond→target | cos(cond,target) | text−scond | sig |
|---|---|---|---|---|
| 2 | h20→h22 | 0.87 | +0.0013 | tiny |
| 4 | h20→h24 | ~0.80 | **+0.0003** | **ns (p=0.30)** |
| 8 | h16→h24 | 0.61 | **+0.0128** | p<1e-4, 82% |

**Not monotonic in layer count** — gap-4 sits *below* gap-2 (critic training is
near-deterministic, so this is real). The driver is how far the condition is from
the target: h20 already "knows" h22/h24 (nothing nameable left), while h16→h24
spans genuine pre-settlement computation. Diff verbalization is a **regime**
(large gap from a rough early layer), not a gradual function of depth — but in
that regime the frozen AV verbalizes the diff substantially (+0.0128). Headroom
scales too: linear h16→h24 headroom 0.222 vs 0.106 for gap 2.

## Stage 1 — does training the AV beat the frozen ceiling? (gap 8)

BoN-SFT (de-risk before any GRPO spend): sample N=8 descriptions/position from
the frozen AV, keep the highest-reconstruction one (scored by a saved reward
critic), SFT the AV on those, measure with a **fresh** critic.

- Batched HF generate: **36 seq/s** (~10× sglang's input_embeds path — what makes
  this affordable).
- Reward critic saw a +0.0051 best-of-N lift; the SFT'd AV's descriptions are
  genuinely different from frozen (Jaccard 0.46, 0% identical) and the SFT took
  (loss 0.58→0.50).
- **Fresh-critic eval: bon 0.8224 vs frozen text 0.8223 → −0.0001 [−.0003,+.0001],
  p=0.41, 50% win. NULL.**

The +0.0051 reward-lift was reward-hacking: best-of-N found text the *reward*
critic liked, not text with more transferable diff content. **Verdict: training
the AV's outputs does not beat its zero-shot diff verbalization.** Per
pre-registered de-risk logic, this argues *against* funding full GRPO (tier 2) —
it would optimize harder against the same exploitable reward and the frozen AV is
already near the reconstruction ceiling.

## Bottom line

The conditioned-AR setup gives the program's first real positive: at a genuinely
large layer gap, a frozen NLA verbalizes the residual-stream diff with a
significant, position-specific, after-state-specific signal (+0.0128, 82% of
positions). But that is the **ceiling of the reconstruction objective** — more AV
training doesn't move it. To go further toward "explain what a diff *does*," the
lever is a different objective (behavioral / simulatability-scored free-form
text), not a better reconstruction reader.

## Pure-text pre-test (AV2/AR2 idea, frozen-AV side)

Tests a *text-only* AR2 (no vector injection): reconstruct h24 from AV(h16) and/or
AV(h24) text. Three arms, gap 16→24, doc-clustered bootstrap:

| arm | cos |
|---|---|
| ptB = AV(h16)+AV(h24) text (the AR2 input) | **0.8273** |
| ptA = AV(h24) text (after) | 0.8216 |
| ptC = AV(h16) text (before) | 0.8091 |

- **ptA − ptC = +0.0125** [.0116,.0133], 80% — pure-text diff-specificity ≈ the
  vector-conditioned +0.0128. The diff signal is NOT a legibility artifact.
- **ptB − ptA = +0.0057** [.0054,.0060], 86.5% — the two-text input beats
  after-only; the before-text adds real value (frozen AVs).
- **ptB 0.8273 > Gate G vector-conditioned 0.8223** — describing the before-state
  in *text* beats injecting its *vector* (consistent with legibility dominating).

Green light for a co-trained AV2/AR2: the AR2 architecture works and the
all-text route is competitive-to-better. Open question remains AV2 *training*
value (Stage-1 BoN-SFT null is the cautionary prior; co-trained AR2 +
text-conditioning is a cleaner setup).

## Costs / artifacts

Two single-H100 sessions (Stage 0 + gap-curve/Stage-1) ≈ **$13 + ~$13**. All
pods killed. Descriptions (av_h22/h24/h16/h20_out, av_bon_descs), best-of-N
selections, the SFT'd adapter (av_bon_lora), per-position eval cosines, and
bootstraps published under gate_g/ on HF; wandb project `nla-layer-diff`
(runs g2_*, g4_*, g8_*).
