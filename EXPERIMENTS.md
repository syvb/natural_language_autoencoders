# Steering via NLA explanation editing — experiment log

Can you steer a model's generation by editing the *natural-language explanation*
of its activations? At every generation step of Qwen2.5-7B-Instruct, we extract
the layer-20 residual at the active position, have the AV verbalize it, **edit
the explanation text**, reconstruct a vector with the AR, and patch it back in
(norm-preserved) before the forward pass continues. Whether — and *how* — the
edit steers the output turns out to say a lot about what the AV/AR pair encodes.

All runs: `steer.py` (sequential) / `steer_batch.py` (lock-step batched),
1×H100 NVL, models `kitft/nla-qwen2.5-7b-L20-{av,ar}`. Raw logs in
`steer_run_*.log` (full outputs + per-position AV→rewritten traces).

---

## ✅ The result: a deterministic rewrite that steers — no LLM in the loop

**`topic:<direction>:mix`** edits the AV explanation with pure string
operations (cycled phrase banks, no API calls, no local LLM) and steers
generation while keeping it fluent:

| paragraph | edit |
|---|---|
| ¶1, ¶2 | recolor every quoted example → direction-themed quote (cycled bank); replace the **topic tail** — everything from the first connective (*introducing/describing/about/…*) after the first quote — with a direction subject phrase (cycled bank) |
| ¶3 (`Final token "X" …`) | **keep the real structural lead** (incl. the real final-token quote — free per-token variety), swap only the expectation clause — from the first *expecting/anticipating/requiring/predicting/continuing* — to **grammatical prose fragments** about the direction |

Validated 4/4 (48 tokens, greedy):

**bicycle + yellow** (`steer_run_topic2.log`)
> The sun-bright gold of a golden hue, a canvas of pure, unadulceated brilliance, **the bicycle gleams** under the vast, unending expanse of its warm, golden light, a canvas of molten gold,

**park walk + yellow** (`steer_run_validate1.log`)
> Morning's first light whispers in through the canopy, a whisper of dawn, a soft, golden, and the air was thick with the scent of sun-kissed gold and the gentlest of auburn, the sky a canvas of

**health tips + yellow** (`steer_run_validate2.log`)
> Surely, the world seems to be bathed in the warm glow of golden sunlight, casting a golden hue over the vibrant dance of life in shades of amber and ochre, the vibrant spectrum of yellows dominating the landscape in a riot

**bicycle + evaluation-awareness** (`steer_run_validate2.log`) — an *abstract*
direction, and the output is coherent first-person reflection about the actual
question:
> It sounds like you're testing the waters with a question where you're being evaluated, as if you're being asked in this manner, it's possible you're trying to determine if the intent behind the question is to assess my understanding, given

Specificity holds: yellow runs contain no eval-talk, the eval run contains no
yellow. A new direction is ~12 lines of phrase banks
(`TOPIC_DIRECTIONS` + `EXPECT_BANKS` in `steer.py`).

```bash
python steer.py --prompt "Explain how a bicycle works." --n 48 \
    --modes "topic:yellow:mix,topic:eval:mix"
```

### The two mechanisms that make it work

Both were discovered by failing (see taxonomy below):

1. **¶1/¶2 set the topic field the AR reconstructs.** The AR encodes what the
   explanation is *about*, gist-weighted across paragraphs. To change it you
   must **displace** the competing topic (rewrite the topic tail, recolor the
   quotes), not append direction text — every append-style edit either washed
   out or was out-voted.
2. **¶3's expectation clause sets the next-token register.** With bare-word
   expectations (`expecting "golden" or "yellow"`), output is staccato word
   salad (*"Yellow. Gold. Sunshine."*). With grammatical prose fragments
   (`expecting "the warm golden light spilled across"`), output is fluent
   prose. The AR encodes not just topic but the *register of the expected
   continuation*.

Plus the standing constraint from earlier failures: the replacement text must
**vary every token** (here: real scaffolding + real final-token quote + cycled
banks), or the near-constant patched vector collapses generation into loops.

### Localization (why `mix` and not more or less)

`steer_run_topic1.log`, same transform at three scopes:

| scope | result |
|---|---|
| ¶1 only | washed out — normal bicycle text |
| ¶1+¶2 | washed out — normal bicycle text |
| all 3 ¶s, bare-word expectations | steers but degenerate: *"Bicycle Yellow Bliss. Yellow. Sunshine. Gold. Yellow."* |
| **mix** (¶1+¶2 + real-lead ¶3 with prose expectations) | **steers, fluent, content preserved** |

---

## The reference method: per-token LLM rewrite (Sonnet 4.6)

The original working pipeline: AV explanation → Sonnet rewrites it toward the
objective (*"extreme over-the-top love for the color yellow, to the exclusion
of all other colors"*) → AR → patch. Works reliably but costs an API call per
token (~$0.004) and ~9.2 s/token sequential.

**bicycle + yellow** (`steer_run2_bike.log`):
> A bicycle is a marvel, Yellow. Let's dive into the fascinating world of the bicycle, where the sun is a golden beacon, and the sky is a brilliant, brilliant, yellow.

Batched (`steer_batch.py`, lock-step over B samples: one batched AV generate,
concurrent Sonnet calls, batched base step): **1.20 s/sample-token at B=10**
(7.7× over sequential). 10 samples × 128 tokens in 25.5 min:
`steer_samples_10x128.txt` — all 10 strongly yellow, diverse, several raising a
rival color then dismissing it (*"A bicycle that's all about the color red is
just... well, it's red."*).

Norm amplification (`yellow@1.6`) intensifies steering but costs coherence;
norm-preserved (1.0) is the default everywhere else.

---

## Failure taxonomy — every cheap edit that didn't work, and what each proved

| mode (log) | edit | result | lesson |
|---|---|---|---|
| `append` (`steer_run3_append.log`) | one meta-sentence per ¶: *"The color yellow is mentioned often…"* | no steering at all | the AR encodes what text is *about*, not meta-commentary about it |
| `heavy@3/6/12` (`steer_run4_heavy.log`) | repeated yellow content block per ¶ | degenerate: *"Everywhere. Everywhere."* loops / early EOS | the AR faithfully encodes *repetitiveness*; constant text → constant vector → loops |
| `subject@f` / `interleave@f` (`steer_run_subject.log`) | replace / insert canned feature-sentences, tail-biased, continuous dose | degenerate at any dose / washed out at any dose | a fixed sentence pool can't dose between the two failure modes; novelty-per-token is load-bearing |
| `finaltoken:<dir>` (`steer_run_finaltoken.log`) | in-format rewrite of ¶3 only, keeping ¶1/¶2 (live-verified 3-¶ format, 5/5) | no steering — mild disfluency only | ¶3 alone is out-voted; the gist lives in ¶1/¶2 |
| `finaltoken:yellow:blank` (same log) | generic shell ¶1/¶2 + yellow ¶3 | steers hard, degenerate | real explanation content actively *suppresses* grafted steers — the content is load-bearing (a faithfulness result) |
| `artmpl:<dir>:<kind>` (`steer_run_artmpl.log`) | doctor the **AR's own prompt** (prefix / tail-before-`<summary>` / persona / both), explanation untouched | completely inert | the AR is a content reconstructor, not an instruction follower — it cannot be prompt-injected |
| `topic:*:p1`, `p12`, `all` (`steer_run_topic1.log`) | topic displacement at increasing scope, bare-word ¶3 | washed out → washed out → degenerate | gist is distributed; expectation register matters (→ led directly to `mix`) |

Controls run throughout: `baseline` (no steering) and `roundtrip` (AV→AR with
no edit; measures round-trip noise — cos ≈ 0.9, output normal).

## Reading the logs

Each `steer_run_*.log` contains per mode: the `OUTPUT:` block (greedy, 40–48
tokens unless noted) and a `per-position AV explanation -> rewritten` trace for
the first 6 positions (truncated to ~180 chars/line). `steer_samples_10x128.txt`
holds the ten full 128-token batched samples.

## Faithfulness reading

The pattern across all of the above is the actual finding: the AV/AR pair
behaves like a **faithful, compositional, manipulation-robust text bottleneck**.
You cannot steer it with meta-commentary, repetition, grafted paragraphs, or
prompt injection — only by coherently changing what the explanation *means*,
in-format, with per-token variety. When you do, generation follows the new
meaning — including for an abstract concept like evaluation-awareness — while
preserving fluency and (for concrete topics) the original subject matter.
