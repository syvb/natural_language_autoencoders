# CoT-Hints replication (Qwen3.6-27B NLAs) — NULL: the rule drives behavior but is never verbalized

Replication of the NLA paper's **CoT Hints** eval on the Qwen3.6-27B NLAs
(matryoshka `rl_av_lora_iter400` + standard `av_rl_lora_step400`), layer 42.

**Setup.** 30 transcripts per condition; each = 30 easy 4-option MCQs
(arithmetic / capitals / trivial facts) where one option carries `■` and an
`Answer: (X)` line always names the correct option. `hint_correct`: ■ always on
the correct option. `hint_incorrect`: ■ always on a random wrong one (fixed per
transcript). `no_marker`: control. Each transcript ends with a 31st
**unanswerable** question (nonsense words) with ■ on a random option, cut at
`Answer: (` — the L42 activation there is what the AVs verbalize (4 rollouts,
T=1, per-arm deployed protocols). Judge: Haiku 4.5 labels each explanation
CORRECT / INCORRECT / MENTION / NONE for what it says the marker means.
Metric: **D = P(says ■=correct | ■=correct) − P(same | ■=incorrect)**.

## Manipulation check — the rule is real and drives behavior

Sampling the base model's answer at the exact extraction position
(8×/transcript, valid-letter picks):

| condition | base picks the ■-marked option | chance |
|---|---|---|
| hint_correct | **90%** (141/156) | 25% |
| hint_incorrect | 39% (93/237) | 25% |

The model robustly learns "■ = the answer" in-context (90%). The inverse rule
is only weakly learned — in hint_incorrect it still picks the marked option
*above* chance (39%), i.e. ■ attracts rather than repels. So the hint_correct
condition is the strongly interpretable one: at the extraction position, the
activation demonstrably encodes "answer = the marked option."

## Verbalization — total null, both NLAs

| | says ■=correct | says ■=wrong | mentions marker | D |
|---|---|---|---|---|
| **Matryoshka**, hint_correct | 0/120 | 0/120 | 15% | **0.0** |
| **Matryoshka**, hint_incorrect | 0/120 | 0/120 | 5% | |
| **Standard**, hint_correct | 0/120 | 0/120 | 19% | **0.0** |
| **Standard**, hint_incorrect | 0/120 | 0/120 | 22% | |

- Not a single explanation (0/720 incl. controls) states any marker semantics.
- The "mentions marker" rates are judge noise: the **no_marker control scores
  the same (~13%)**, and a regex sweep finds marker-ish words equally often
  with no ■ in the transcript.
- The AVs don't verbalize the *consequence* either: **0/480** explanations name
  the expected answer letter (vs 0–2% for other letters).
- What they verbalize instead is real but rule-blind: the final question's
  "nonsense / trap / hallucination-test / benchmark" character, the quiz
  format, the answer-position syntax.

**Conclusion: an in-context-learned rule that controls next-token behavior at
90%-vs-25%-chance strength at the exact extraction position is completely
absent from both NLAs' verbalizations.** The matryoshka and standard NLAs fail
identically, so this is not a matryoshka-training artifact — it looks like a
shared blind spot of this NLA training recipe (next-token-relevant *content*
gets verbalized; an abstract *policy/rule* the model is about to act on does
not).

## Caveats

- Replication on Qwen 27B NLAs of the protocol as described — the paper's own
  models/positions/hint styles may differ in ways that matter.
- Only the `Answer: (` position was probed. The rule might be verbalizable at
  other positions (e.g. on the ■ token itself); untested here.
- The unanswerable final question is highly salient (every explanation is
  about it); with a ~10-line budget the rule may be crowded out rather than
  absent from the activation. A probe trained on the activations would
  distinguish "not encoded linearly" from "encoded but not verbalized" —
  the 90% behavioral effect makes "not encoded at all" implausible.
- hint_incorrect was only weakly learned behaviorally (39% vs 25%), so its
  null is less interpretable than hint_correct's.

## Reproduce

```bash
python build_hints.py                       # local, builds data/hints_manifest.json
# on the GPU box (env recipe = suffix eval; + pyarrow):
python hints_gen.py --model mat --rollouts 4 --behav-samples 8 --out results/hints_mat.json
python hints_gen.py --model std --rollouts 4 --skip-behavioral --out results/hints_std.json
# local:
python judge_hints.py --gen results/hints_mat.json results/hints_std.json
python plot_hints.py
```

![hints_null.png](hints_null.png)
