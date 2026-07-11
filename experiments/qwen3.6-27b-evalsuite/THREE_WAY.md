# Three-way: matryoshka RL vs matryoshka warm-start vs standard RL

Same 5,275 positions, same gpt-4o-mini eval-awareness judge, all scores
precomputed. `three_way.py` → `three_way.png`. Decomposes two axes: RL vs
warm-start (training), and matryoshka vs standard (format).

| model | eval AUC | plainμ | honeyμ | hit@1% | hit@5% | 1st-unit useful alone | full-expl FVE |
|---|---|---|---|---|---|---|---|
| **matryoshka RL** | 0.726 | 0.089 | 0.282 | 6.2% | 20.2% | **93.6%** | 0.503 |
| matryoshka warm | 0.520 | 0.055 | 0.100 | 4.0% | 9.9% | 53.2% | 0.153 |
| standard RL | 0.712 | 0.005 | 0.110 | 19.5% | 35.4% | 2.0% | 0.530 |

## The two ingredients do different jobs

**RL creates the eval-awareness signal — the format barely matters for it.**
Both RL models land at AUC ~0.72; the pre-RL warm-start matryoshka is at 0.52
(≈chance). So the eval-awareness content is a product of the RL training, not
the matryoshka format: matryoshka RL and standard RL detect it about equally,
and standard RL is actually *better at matched false-positive rate* (hit@1%
19.5% vs 6.2%) because matryoshka carries a higher plain-text bias baseline
(plainμ 0.089 vs 0.005 — the saturation/bias confound from
eval_awareness_bias.py). **For eval-awareness detection per se, matryoshka
format buys nothing over standard; RL is the whole story.**

**The matryoshka format creates front-loading — but only pays off with RL.**
The "read the first unit, get the answer" property (1st unit useful alone)
is 93.6% for matryoshka RL, 2.0% for standard RL (it reconstructs equally well
overall, full-FVE 0.53, but buries the informative sentence), and 53.2% for
warm-start matryoshka — which also can't reconstruct (full-FVE 0.15). You need
BOTH the ranked-list format AND the RL: standard RL has reconstruction quality
but the wrong format; warm-start matryoshka has the format but no
reconstruction quality; only matryoshka RL has both.

## Bottom line

- **RL** → reconstruction quality (full-FVE 0.15 → 0.50) + eval-awareness
  detectability (AUC 0.52 → 0.73). Necessary for both.
- **Matryoshka format** → answer-first usability (1st-unit-alone 2% → 94%).
  Its value is *usability*, not eval-awareness detection (where it ties/loses
  to standard).
- **Only matryoshka RL** has the full package: reconstructs well, reveals
  eval-awareness, and puts the answer on line 1.
