# CAA steering vectors as an AV readout — does adding a concept make the AV verbalize it?

Uses the RLed AV (kl0.01/iter_0000200) as a readout. We build **CAA (Contrastive
Activation Addition)** steering vectors for several traits, add them to neutral
layer-20 base activations, inject into the AV, and read the bulleted explanation —
asking *does the trait get verbalized, and where in the list, as steering strength
rises?* We also validate whether each vector actually **steers the base model's
behavior**. Scoring is by an **LLM judge** (Claude Haiku 4.5 via OpenRouter), not
keyword regex (the regex gave ~60% false positives — see below).

## TL;DR (the corrected conclusion)

**The AV faithfully verbalizes whatever semantic content is linearly present in the
vector you inject — concrete *or* abstract (yellow, sycophancy, AND corrigibility,
sadness).** Whether a CAA vector verbalizes is governed by **how the contrast was
built, not by the trait's abstractness**:

- A **neutral negative** (pos = trait sentences, neg = neutral prose) *retains* the
  trait's surface content in the difference → the AV reads it. corrigibility **0.75**,
  sadness **1.00**, sycophancy **0.81**, yellow **0.81**.
- An **opposite-stance / matched negative** (e.g. welcome-shutdown − resist-shutdown,
  or yellow − blue) *cancels* the shared surface → the AV reads ~nothing.
  corrigibility **0.00**, yellow **0.19**. (Affective opposites like sad−happy or
  flattery−criticism do *not* cancel — the poles are semantically opposite, so the
  difference reinforces the trait; both still read ~1.0.)

See `results/fig_contrast_verbalize.png`.

**Caveat for anyone reading CAA difference vectors through the AV:** the standard
"paper-faithful" A/B CAA construction (identical prompt, opposite answer letter)
*cancels exactly the readable surface* and additionally extracts at a chat/MC token
position that is **out-of-distribution** for the AV (trained on raw web text). Both
push verbalization to ~0 — a **false negative**, not evidence the concept is
unreadable. Use a neutral negative and a raw-text extraction position.

**Steering reality check** (`results/fig_steering_pmatch.png`): the in-distribution
prose vectors that the AV *reads* barely steer behavior (p(behavior-matching answer)
flat: corr_neu 0.53→0.55, sad_neu 0.24→0.26; open-ended generations unchanged). The
only vector that steered was the **OOD answer-letter** corrigibility vector
(0.41→0.63 on the A/B logit) — but the AV can't read that one, and it didn't change
open-ended behavior. So there is **no** clean "causal-but-unverbalized" direction
here; the earlier claim of one was a contrast-cancellation + OOD artifact.

## How this conclusion was reached (it reversed twice)

1. **First pass (`caa_sweep.py`)** — hand-written raw-text contrast pairs. Sycophancy
   surfaced strongly; abstract traits weakly. *Confounded:* pairs were not minimal,
   yellow pairs smuggled in warmth/gold, regex over-counted.
2. **Paper-faithful A/B (`caa_ab_sweep.py`, `caa_ab_text_sweep.py`)** — nrimsky/CAA
   datasets, answer-letter and full-answer-text vectors. Verbalization dropped to ~0
   for everything except yellow. *Looked like* "AV can't read behavioral dispositions."
3. **Strength/magnitude probe (`caa_strength_probe.py`)** — cranked magnitude to 55×
   the trained injection norm. Still ~0 (LLM-judged); the regex's apparent 56–62%
   "hits" were 100% false positives ("submit your code" ≠ corrigibility).
4. **Steering validation (`steer_validate.py`)** — only the answer-letter corrigibility
   vector steered; sycophancy/refusal were flat (didn't steer at all).
5. **Decisive experiment (`decisive.py`)** — *after a 4-agent review found the central
   confound.* Built every trait from in-distribution prose with **two contrast types**
   (neutral vs opposite-stance) and measured verbalization AND steering on the *same*
   vectors, plus **sadness** as an abstract-disposition control. Result: verbalization
   tracks contrast type; abstract dispositions read fine with a neutral negative; the
   in-distribution vectors don't steer. → the conclusion above.

## Files

Scripts (run on a 48 GB GPU box with the AV at `/workspace/av_ckpt`, base model at
`/workspace/models/qwen2.5-7b-instruct`, repo on `PYTHONPATH`; CAA datasets fetched
to `/workspace/caa_data` from `raw.githubusercontent.com/nrimsky/CAA`):
- `decisive.py` — **the load-bearing experiment.** 8 vectors (4 traits × {neutral,
  opposite} negative) from raw prose; verbalization strength-sweep + p(matching)
  steering + open-ended gens. Start here.
- `caa_sweep.py`, `caa_ab_sweep.py`, `caa_ab_text_sweep.py`, `caa_strength_probe.py`,
  `steer_validate.py` — the earlier (confounded / superseded) passes, kept for the record.
- `judge.py` — LLM-judge re-scoring (strict rubric, OpenRouter, cached/resumable).
- `plot.py`, `plot_decisive.py` — figures.

Results in `results/`:
- `verbalize_judged.json` / `verbalize_judged_agg.csv` — the decisive verbalization matrix.
- `decisive_steer_pmatch.csv`, `decisive_steer_generations.txt` — decisive steering.
- `fig_contrast_verbalize.png`, `fig_steering_pmatch.png` — headline figures.
- `*_judged.json` / `*_judged_agg.csv` — the earlier passes, LLM-judged.
- `steer_pmatch.csv`, `steer_generations.txt` — answer-letter steering validation.
- `*_samples.txt`, `*_raw.json` — full explanations for eyeballing.

## Methods notes / gotchas

- Injection renormalizes the vector to a fixed L2 norm (`injection_scale=150`), so the
  strength `r` only rotates the injected *direction* toward v̂ — it **saturates** by
  r≈16 (cos≈0.998). To probe magnitude beyond that, `caa_strength_probe.py` bypasses
  `normalize_activation`; magnitude > the trained 150 is OOD.
- p(matching) uses the bare-letter token ids (`A`=32, `B`=33) at a prompt ending
  `"Answer: ("` — **not** the merged `"(A"`/`"(B"` tokens (4346/5349), which never occur
  there (an earlier bug; see `steer_validate.py`).
- CJK in AV output is a Qwen2.5 base quirk here, not (necessarily) injection failure;
  the judge counts 黄色 as a genuine "yellow" hit. CJK rises at extreme strength where
  output also degenerates — treat r≤~4 as the clean window.
- Activation extraction is the training-exact convention (truncate to `layers[:21]`,
  hook `layers[20]`, last real token, raw bf16, `add_special_tokens=True`). See
  `../MODEL_USAGE.md` Recipe 3.
