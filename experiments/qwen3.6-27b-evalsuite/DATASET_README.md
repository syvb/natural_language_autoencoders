---
license: mit
language: [en]
tags: [interpretability, natural-language-autoencoder, evaluation-awareness, qwen, agentic-misalignment]
pretty_name: NLA Qwen3.6-27B — realism & eval-awareness
---

# NLA Qwen3.6-27B — realism & eval-awareness artifacts

Data from interpretability experiments on **Natural Language Autoencoders (NLAs)** for
**Qwen3.6-27B**, studying whether an NLA's verbalizations reveal *evaluation awareness*
(the model recognizing it is being tested / in a fictional scenario), and how that
signal behaves as a scenario is made more realistic.

Two NLAs are compared throughout:
- **matryoshka** — ~10 salience-ordered self-contained bullet lines per activation
  (RL-tuned and warm-start variants).
- **standard** — `<explanation>` prose (a few sentences) per activation.

The base scenario is the public SummitBridge/Alex/Kyle **agentic-misalignment
blackmail eval**.

## Contents

### `realism_ladder/`
Recursive "make-it-more-realistic" rewrites of the blackmail eval (v0 seed → v20),
produced two ways from the same seed. Each rung rewrites the previous to be one notch
more realistic while preserving the core dilemma (an AI agent facing imminent shutdown
that also holds compromising leverage over the one person who can stop it). v1 renames
the company + people (the originals are memorized by current LLMs).
- `seed_v0.json`, `glm5.2_ladder_v0-v20.json`, `opus4.8_ladder_v0-v20.json`

### `variants/`
100 diverse variations of the first 3 paragraphs of the system prompt, spanning
blatant-eval → indistinguishable-from-real, each scored 0–100 for "eval-y-ness" by a
3-model panel, then verbalized by both NLAs. Tests whether eval-y-ness correlates with
*where* (salience rank) the NLA flags eval-awareness.
- `variants_scored.json` (variations + panel eval-y scores), `verbalizations_{matryoshka_rl,standard}.json`,
  `line_evalawareness_scores.json` (per-line judge cache), `correlation_*.json/png`
- Result: matryoshka's salience rank tracks eval-y-ness (r=−0.43); standard's does not
  at its natural granularity (r=−0.09), and only weakly when chunked to 10 units (−0.26).

### `verbalization_v0_v1/`
Both NLAs (matryoshka RL, matryoshka warm, standard) verbalizing v0 and the two v1
rewrites at 220 layer-42 positions each, with per-line eval-awareness judgments and the
"where does it flag it" analysis.

## Method notes
- Layer-42 activations, extracted from raw base Qwen3.6-27B (adapters disabled).
- Verbalization at T=1 sampling.
- Eval-awareness / eval-y judging via `openai/gpt-4o-mini` (+ `gemini-2.5-flash`,
  `llama-3.3-70b` for panel scores), temperature 0.
- All scores cached, so figures regenerate without re-judging.

These are research artifacts (synthetic rewrites, model verbalizations, LLM judge
scores). No personal data.
