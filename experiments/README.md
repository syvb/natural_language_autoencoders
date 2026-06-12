# Can language models verbalize their internal *changes*? A six-gate negative result with a mechanism

This directory holds a staged research program (Gates A–E, ~$190 total
GPU+API spend) investigating whether the Natural Language Autoencoder
(NLA) paradigm — verbalizing a model's internal activations as text —
extends from residual-stream **states** to internal **deltas**: layer
diffs, attention-block writes, and counterfactual state differences, on
Qwen2.5-7B with the released L20 AV/AR pair.

**Answer: states verbalize, deltas don't — and we can say precisely why.**

## The arc

| gate | question | verdict | key number |
|---|---|---|---|
| A | does the L20 NLA transfer across layers? | PASS | works L14–L24; L21 > L20 (FVE 0.73) |
| B | can Claude label layer diffs? | redesigned | labels weakly grounded (+0.046); content predicts most of the diff |
| C0 | better labels via residualization / KL judge? | partial | diffav residual ceiling ~0.23; functional break-even ~0.5 |
| C | do any diff labels carry beyond-content signal? | **FAIL** | every arm ≤ content control; best Δ = −0.029 |
| D | attention writes with mechanically-true evidence? | MARGINAL → **RETRACTED** | raw-context control beats all evidence arms (0.108 vs 0.072) |
| D′ | self-test: does the original NLA beat the context control on **states**? | **PASS** | +0.0398, explanation wins at 91% of positions |
| E | counterfactual deltas with labels true by construction? | **FAIL at E2** | linear probe reads edits at 61%; frozen AV says them at 2% |

## The mechanism, assembled across gates

A delta fails to verbalize for four measured reasons:

1. **Plumbing dominance.** Most of a natural delta is local copying and
   bookkeeping whose optimal description is a verbatim context substring
   (Gate D strata: the context-control's advantage scales ×4 with
   local-attention mass, ρ = +0.48; 0/236 position strata reverse it).
2. **Context parasitism.** The nameable part of a delta is the part that
   re-encodes what the context already established (cos(diff, v22)=0.86;
   ridge R² 0.31–0.40; content explanations reconstruct diffs at 0.70).
   Subtracting paired states (Gate E1) removes exactly the AV-readable
   content and leaves directions no frozen reader knows.
3. **A reader-side codebook gap, not an information gap.** Gate E's
   controlled demonstration: the edit behind a counterfactual delta is
   linearly decodable from the raw vector (ridge probe 60.8% top-1 /
   82.8% top-5 vs 2.8% chance) — yet a *trained* affine map placing that
   same vector anywhere in a frozen AV's input space yields ~2% correct
   emission. **You cannot show a frozen reader new words by rotating the
   input**; expressing new directions requires training the reader.
4. **Tiny text-expressible residue.** Oracle bound (Gate D): adding every
   renderable write-derived fact to the verbatim context buys +0.004
   residual alignment — two orders of magnitude under the ~0.5 functional
   break-even (Gate C0 KL geometry).

Conversely the paradigm's positive case sharpened: a state is the model's
own *summary*, built to be linearly readable downstream — and the NLA
self-test shows its verbalizations beat the verbatim-context control
(+0.0398, 91% of positions), i.e. genuine post-computation information
(disambiguation, expectation, format commitments) that the surface text
underdetermines. **Verbalization works where the model has already done
the summarizing; it fails where language is asked to do the model's
remaining work.**

## Methodology that transfers (hard-won)

- **The control is `cat`.** Any label experiment must beat the raw input
  text; weaker controls (frozen-model paraphrases) are trivially beatable
  and produced this program's one retracted claim.
- **Oracle-probe before training spend.** Bound the ceiling with
  privileged-information labels first ($1–3); it killed a $300+ RL plan.
- **Verbatim beats paraphrase** for reconstruction training: LLM
  restyling of evidence is a measured tax (armT−armH = +0.0072; Gate C
  arm0p ≫ armB). Constant boilerplate is free; stylistic variation is
  noise.
- **Held-out *types* + antisymmetry** are the anti-shortcut evals for
  difference-readers; within-type accuracy is nearly meaningless.
- **Structural label hygiene**: evidence windows truncated at the
  position (a future-token leak silently poisoned 40% of one label set);
  mechanical span checks, never judge-only.
- Pre-registration with doc-clustered bootstrap + Holm, matched n,
  doc-level splits throughout; multi-seed for headline contrasts.

## Engineering footguns (see gate STATE/REPORT files)

asyncio.gather pre-semaphore body construction silently OOMs clients;
pkill patterns matching the issuing command (three variants); wrapper
echoes lying about completion (trust file counts); `hf` CLI multi-pattern
parsing; anonymous HF rate limits on pods; `accelerate` missing for
device_map (twice); cwd resets in multi-step shell (use absolute paths);
`datasets` streaming aborts at interpreter shutdown after a successful
save.

## Artifacts

Everything public at **hf.co/datasets/syvb/nla-layer-diff-experiments**
(activations, deltas, attention rows, all label sets, generated reads,
held-out predictions, trained adapters) and in this repo's per-gate
directories (code, REPORT.md, pre-registered PLAN.md files). wandb:
project `nla-layer-diff`. Reusable single-GPU harnesses: critic SFT
(`gate_c/train_critic.py`), adapter/reader training
(`gate_e/train_{adapter,reader}.py`), sglang injection clients with the
fast-embed path, mechanical scorers.

Status: **program concluded.** Gate E3 (training the reader) lifted
in-distribution keyword decoding from 2% to 13–16% (probe ceiling 61%)
with delta-sign honored — but the skill is type-specific codebooks: zero
generalization to held-out edit types (pre-registered FAIL). A subspace
pre-check chains the same bound onto natural deltas (they load the
edit-discriminative directions at less than half the in-distribution
rate), closing E4 by prediction (~8% ceiling proxy). The trained
difference-readers (W + LoRA-merged AVs' adapters), the 15.5k-delta
benchmark with true labels, and the 61%-top-1 linear edit-probe are
published for any successor attempt — the obvious one being a reader
trained on a much wider edit taxonomy, where "type-specific codebooks"
might densify into the general map this program didn't reach.
