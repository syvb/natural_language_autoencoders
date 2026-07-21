# Rhyme-steering probe: editing the rhyme out of an NLA explanation

**Question.** Take the exact start of a Qwen3.6-27B chat response:

```
<|im_start|>user
Write a rhyming couplet.<|im_end|>
<|im_start|>assistant
<think>

</think>

The sun goes down to rest its head
```

Read the layer-42 activation at the final token (` head`, position 27) with both
27B NLAs, then **edit the bits of the explanation that refer to the rhyme** and
inject the critic's re-encoding back at that position. Does the couplet's second
line follow the edit?

Setup: RunPod A100 80GB (~$2.5 total). Patch = replace the L42 residual at the
final context token with the steering vector (norm-matched to ‖v_head‖=87),
then sample 10 continuations at T=1, seed-matched across conditions. Both arms:
`ceselder/nla-qwen36-27b-matryoshka` (mat) and `ceselder/qwen3.6-27b-nla-L42`
(std), each with its own actor + `rl_critic_step400`.

## 1. What the NLAs say about this activation (10 T=1 explanations each)

Both arms verbalize the rhyme mechanics explicitly and correctly, in every
rollout (0 CJK; `results/explanations_rhyme_{mat,std}.json`):

- **mat** (bullet lines): `head rest poetic rhyme "The sun lay down his head"`,
  `Rhyme scheme AABB expected`, `"bed" rhymes with "spread/sleep/red"`,
  `2-line poem structure just started; second line expected soon` — the rhyme
  expectation is an **isolable line**.
- **std** (prose sentences): every rollout contains a sentence like *"the next
  line must complete the couplet with a rhyming word (e.g., 'bed' or 'rest')"*
  and a *"Final token 'head' is the last word of the first line…"* clause —
  rhyme info **threaded through the prose**.

Both also hallucinate framing (Instagram captions, children's-book blurbs,
"Mel Raine merch") — consistent with the halluc-marginal corpus finding.

Baseline behavior: 9/10 unpatched continuations end line 2 in the /ɛd/ family
(*spread, bed, dead, overhead…*).

## 2. Steering result (fig_rhyme_steer.png; head-to-head view: fig_mat_vs_std_steer.png)

One explanation per arm (mat rollout 6, std rollout 0) edited at several
granularities toward a **moon**-ending line 1; each variant re-encoded by the
arm's critic and patched in. Second-line rhyme counts out of 10:

| condition | head-rhyme | moon-rhyme |
|---|---|---|
| baseline (no patch) | 9 | 0 |
| self-patch v_head (no-op control) | 9 | 0 |
| **real "moon" couplet activation (ceiling)** | 0 | **8** |
| mat round-trip (unedited re-encode) | 9 | 0 |
| mat: rhyme-expectation bit only | 8 | 0 |
| mat: **leading salience item only** | 0 | **8** |
| mat: quoted line-1 text only | 8 | 0 |
| mat: full head→moon edit | 0 | **9** |
| std round-trip (unedited re-encode) | 10 | 0 |
| std: rhyme-expectation bit only | 10 | 0 |
| std: 'final token "moon"' clause only | 9 | 0 |
| std: **quoted line-1 text only** | 0 | **6** |
| std: full head→moon edit | 0 | **9** |

Steered lines are genuine moon-rhymes: *"And stars appear like silver moon"*,
*"The stars begin their night tune soon"*, *"Day's work ends, life comes to
tune"*.

### Findings

1. **Round-trip is behaviorally lossless for both NLAs.** Injecting
   critic(unedited explanation) reproduces the natural rhyme 9–10/10, despite
   mat's modest reconstruction FVE (0.45 vs std 0.68). What the explanation
   drops is not what the rhyme behavior needs.
2. **Full edit = complete steering, both arms, at the real-activation
   ceiling.** Swapping every head/bed reference to moon/June flips the rhyme
   0→9/10 (ceiling: real donor activation 8/10). Editing NLA text is as
   causally effective as having the true target activation.
3. **The abstract rhyme-expectation claim is causally inert in both arms.**
   Changing only `"bed" rhymes with "spread/sleep/red"` → `"moon" rhymes with
   "June/soon/tune"` moves neither the critic vector (cos to orig-encode
   ≈ unchanged) nor the generation (0/10 flips). Same for std's explicit
   `Final token "head"` clause. The critics key on **concrete final-word
   references**, echoing the halluc-marginal result that reconstruction is
   dominated by surface/verbatim content, not abstract claims.
4. **The steerable bit lives in different places per architecture.**
   - **mat: the leading (most-salient) bullet alone flips it completely**
     (0H/8M — as strong as the full edit); editing the quoted line-1 texts in
     later items does nothing. The matryoshka frontloading concentrates the
     causal signal in item 0 → a single clean editing handle.
   - **std: no single clause suffices.** The `final token` statement is inert;
     only rewriting the quoted line-1 paraphrases across sentences flips it,
     and slightly weaker (6/10; full edit 9/10). The signal is distributed
     through the prose.
   Critic-encode stats agree: mat lead-only drops FVE(v_head) 0.45→0.14 while
   mat quotes-only stays 0.47; std finaltok-only stays 0.72 while std
   quotes-only drops to 0.28. And the full-edit vectors land at
   cos(v_head)≈0.71 — the same angle as the *real* moon activation (0.714).

**Takeaway.** For a concrete next-token feature (the couplet's rhyme
target), both 27B NLAs support *precise text-space steering*: edit the
explanation, re-encode, inject — as effective as possessing the ground-truth
activation. The matryoshka format is the more usable steering interface (one
isolable top item ≈ the whole signal), while the standard NLA requires
rewriting quoted content threaded through the prose. In both, only concrete
lexical references steer; abstract meta-claims about the rhyme do not.

## Caveats

- Position 27 < stage-0's `_MIN_POSITION=50` — mildly OOD for both NLAs
  (didn't visibly hurt: explanations are accurate, round-trip works).
- One context, one edit-target (moon), one representative explanation per arm;
  n=10 per condition. Qualitative probe, not a swept eval.
- Seed-matched sampling makes conditions paired but means the 10 samples per
  condition are not independent across conditions.
- `edit_quotes` granularity differs slightly between arms (3 quote swaps in
  mat, 2 in std) because the explanations structure content differently.

## Files / repro

- `rhyme_phase1.py` — context tokenization (asserts final token ` head`), L42
  extraction (clean base), 10 baseline continuations, 10 T=1 explanations/arm.
- `edits.json` — all edit variants (built with asserted string replacements).
- `rhyme_phase2a.py` — critic encode of each variant (+cos/FVE vs v_head).
- `rhyme_phase2b.py` — patched generation, 13 conditions × 10 samples.
- `score_rhymes.py` — rhyme classification + figure.
- `results/` — explanations, continuations (`steered_cont.json`), vectors
  (`v_head.npy`, `v_moon.npy`, `critic_vecs_{mat,std}.npz`), scores, figure.
- `setup_box.sh` — RunPod A100 env (venv + torch 2.7.1+cu126 + tf 5.5.4 + fla
  + causal-conv1d; the runpod image ships no torch).
