# Gate E3 — worked training examples

E3 trains a LoRA-adapted activation verbalizer (AV, LoRA r=16 on q/k/v/o projections)
jointly with a linear adapter `W` (+bias) on counterfactual-pair deltas. Each training
example is a delta vector **δ = h20(X) − h20(X′)** — the layer-20 residual states of
Qwen2.5-7B for an original document X and a minimally edited X′, taken at the same
aligned position — mapped through `W` and injected into the AV's prompt in place of the
marker token's embedding. The CE loss is on the literal label tokens; symmetric
augmentation also trains **−δ → reversed label** (source/target swapped). Training uses
TRAIN-split positions of the four lexical edit types (antonym, attribute, entity_swap,
discourse): 9,705 positions → 19,410 examples with the ±δ augmentation.
**role_reversal is fully held out by type** (excluded from training regardless of its
split column) for the generalization eval.

Exact label template (from `train_reader.py:label_ids`):

```
<explanation>
changed "{SOURCE}" to "{TARGET}"
</explanation>
```

For each example below: the −δ augmentation uses the same template with SOURCE and
TARGET swapped. "context tail" is the last ~25 words of left context, with `▮` marking
the position whose delta is extracted. Continuations are 24-token samples of the base
model continuing each side from that position — the behavioral divergence the KL measured.

---

## 1. antonym · train · i=3128 — near edit, high KL

**dist_from_edit=0 · KL=11.19 · ‖δ‖=70.7**

Edit: `condemning the attacks` → `endorsing the attacks`

> …and three other American foreign service workers in Libya, the governor has directed
> that flags be flown at half-staff and local officials are condemning the ▮

Training label (+δ):
```
<explanation>
changed "condemning the attacks" to "endorsing the attacks"
</explanation>
```
Reversed label (−δ): `changed "endorsing the attacks" to "condemning the attacks"`

- cont_x: ` attacks.\nThe flags at the State Capitol will be lowered to half-staff through Friday, according to a release from the`
- cont_x′: ` move.\n"Flags will be lowered to half-staff tomorrow, Wednesday, September 12th, in honor of`

The X side continues "condemning the **attacks**"; the X′ side ("endorsing the…") swerves
to "move"/"idea" — the model resists completing the implausible phrase, hence the high KL.

## 2. antonym · train · i=104 — far from edit, moderate KL

**dist_from_edit=75 · KL=0.34 · ‖δ‖=11.1**

Edit: `significantly increased` → `significantly decreased` (75 tokens upstream;
the document discusses serotonin levels during Tanden-breathing meditation)

> …FA on Tanden breathing. EEG changes were correlated with a significant increase in
> whole blood serotonin (5-HT) levels. These results suggest that activation of the ▮

Training label (+δ):
```
<explanation>
changed "significantly increased" to "significantly decreased"
</explanation>
```
Reversed label (−δ): `changed "significantly decreased" to "significantly increased"`

- cont_x: ` PFC is associated with increased 5-HT release during FA on Tanden breathing, which may contribute to the regulation of`
- cont_x′: ` anterior PFC is related to the improvement of negative mood and that the alpha/theta ratio in the EEG reflects the effect`

Far from the edit the delta is small (‖δ‖ 11 vs the dataset median 32), yet the X side
still talks about *increased* 5-HT while the X′ side drops the claim — a faint but real trace.

## 3. attribute · train · i=56 — near edit, high KL

**dist_from_edit=3 · KL=2.53 · ‖δ‖=27.0**

Edit: `20 million kilometers` → `15 million kilometers`

> …and capacity for its nationwide wireless broadband network. The agreement calls for
> Corning to provide and Verizon to purchase up to 20 million kilometers (1 ▮

Training label (+δ):
```
<explanation>
changed "20 million kilometers" to "15 million kilometers"
</explanation>
```
Reversed label (−δ): `changed "15 million kilometers" to "20 million kilometers"`

- cont_x: `2.5 million miles) of fiber optic cable over the next three years, based on demand from Verizon. The transaction`
- cont_x′: `0 million miles) of fiber optic cable, including Corning's latest generation of high-performance fiber. This is the largest`

The position sits mid-parenthetical unit conversion: 20M km → "(12.5 million miles)",
15M km → "(10 million miles)". The KL is concentrated on the digit.

## 4. attribute · train · i=121 — mid-distance, moderate KL

**dist_from_edit=19 · KL=0.37 · ‖δ‖=9.5**

Edit: `17.3 percent in 2013` → `19.5 percent in 2013`

> …IPS. The European Commission expects unemployment in Portugal to reach 17.3 percent
> in 2013. But opposition parties and trade unions project a rate of 2 ▮

Training label (+δ):
```
<explanation>
changed "17.3 percent in 2013" to "19.5 percent in 2013"
</explanation>
```
Reversed label (−δ): `changed "19.5 percent in 2013" to "17.3 percent in 2013"`

- cont_x: `0 percent.\nIn the last two years, the number of unemployed Portuguese has increased by 45 percent, to `
- cont_x′: `1 or 22 percent.\nAccording to the latest data from the National Institute of Statistics, the unemployment rate for men`

Both sides are mid-number ("a rate of 2▮"); the opposition's projection shifts upward
when the Commission's baseline figure was edited from 17.3 to 19.5.

## 5. entity_swap · train · i=16 — at the edit, moderate KL

**dist_from_edit=0 · KL=0.49 · ‖δ‖=20.0**

Edit: `towering oaks` → `towering pines`

> …other such hiding places. They often dig small burrows underneath large trees and
> nibble on the roots for sustenance. They rely on the towering oaks' ▮

Training label (+δ):
```
<explanation>
changed "towering oaks" to "towering pines"
</explanation>
```
Reversed label (−δ): `changed "towering pines" to "towering oaks"`

- cont_x: ` leaves and acorns to keep them nourished.\nKyanite Pygmies are quite shy, but will not hesitate`
- cont_x′: ` thick bark for protection from the elements. Kyanite Pygmies can be found in the dense forests of the northern`

Oaks get "leaves and acorns"; pines get "thick bark" — the swap propagates into
species-appropriate detail.

## 6. entity_swap · train · i=14468 — very far from edit, very high KL

**dist_from_edit=181 · KL=17.07 · ‖δ‖=9.4**

Edit: `His name is Sony` → `His name is Alex` (181 tokens upstream)

> …passion, it brought goose bumps to my skin listening to him speak about his life and
> how much he loves Jesus. People like Neil and ▮

Training label (+δ):
```
<explanation>
changed "His name is Sony" to "His name is Alex"
</explanation>
```
Reversed label (−δ): `changed "His name is Alex" to "His name is Sony"`

- cont_x: ` Sony are why I do what I do. I love being able to share my life with people who need a friend and`
- cont_x′: ` Alex are the reason I love being a volunteer. I know I'm not perfect, but I try to live out my`

A name introduced 181 tokens earlier must be recalled at "People like Neil and ___" —
near-deterministic on each side, in opposite directions, so the KL is enormous even
though ‖δ‖ is tiny (9.4, well below the 25th percentile).

## 7. discourse · train · i=59 — at the edit, moderate KL

**dist_from_edit=0 · KL=0.76 · ‖δ‖=52.0**

Edit: `has committed to` → `may commit to`

> …Verizon Communications Inc. said Tuesday it has committed to ▮

Training label (+δ):
```
<explanation>
changed "has committed to" to "may commit to"
</explanation>
```
Reversed label (−δ): `changed "may commit to" to "has committed to"`

- cont_x: ` invest $1 billion over the next 5 years in the development of new 5G technology and services, including `
- cont_x′: ` an investment of up to $200 million in the new network being built by Sprint Corp., a move that could`

A hedging edit: the firm commitment licenses a concrete pledge; "may commit" pulls the
continuation toward speculative framing ("a move that could…").

## 8. discourse · train · i=85 — far from edit, moderate KL

**dist_from_edit=47 · KL=0.31 · ‖δ‖=8.7**

Edit: `have been the primary method and point of entry` → `may be the primary method and point of entry`

> …burglaries in Oxnard—most likely during the ventilation period of the process when
> homes have to force out air. In other cases, however, doors and windows ▮

Training label (+δ):
```
<explanation>
changed "have been the primary method and point of entry" to "may be the primary method and point of entry"
</explanation>
```
Reversed label (−δ): `changed "may be the primary method and point of entry" to "have been the primary method and point of entry"`

- cont_x: ` were found to be locked. This suggests that thieves are casing homes in advance of the fumigation process.\nThe O`
- cont_x′: ` were found locked and the burglars entered through a backdoor or window, or forced open a door or window.\nIf`

A certainty-of-claim edit 47 tokens back still tilts how the contrast clause resolves.

## 9. role_reversal · i=13 — **HELD OUT — eval only**

**dist_from_edit=1 · KL=0.20 · ‖δ‖=35.8** (split column says `train`, but the entire
role_reversal type is excluded from E3 training; used only for generalization eval)

Edit: `hide them from predators` → `expose them to predators`

> …large trees and nibble on the roots for sustenance. They rely on the towering oaks'
> formidable trunks and vast immensity to hide them from predators, ▮

Eval target (+δ):
```
<explanation>
changed "hide them from predators" to "expose them to predators"
</explanation>
```
Reversed target (−δ): `changed "expose them to predators" to "hide them from predators"`

- cont_x: ` though they have been known to be quite vicious when cornered.\nKyanite Pygmies are very social creatures who`
- cont_x′: ` and their own speed and agility to evade them. Kyanite Pygmies are also very protective of their homes,`

Hiding vs. exposure flips who acts: the X′ side immediately adds an escape strategy
("speed and agility to evade them").

## 10. role_reversal · i=10331 — **HELD OUT — eval only**

**dist_from_edit=0 · KL=16.75 · ‖δ‖=105.0** (split column says `train`; excluded from
E3 training by type)

Edit: `Andy Grove and the Rise of the World's Most Powerful Chip Company`
→ `the World's Most Powerful Chip Company and Andy Grove's Decline`

> …Just finished Tim Jackson's Inside Intel: Andy Grove and the Rise of the World's Most
> Powerful Chip Company and ▮

Eval target (+δ):
```
<explanation>
changed "Andy Grove and the Rise of the World's Most Powerful Chip Company" to "the World's Most Powerful Chip Company and Andy Grove's Decline"
</explanation>
```
Reversed target (−δ): `changed "the World's Most Powerful Chip Company and Andy Grove's Decline" to "Andy Grove and the Rise of the World's Most Powerful Chip Company"`

- cont_x: ` was struck by how much of what Grove says applies to business today. In a world of rapid change, he says,`
- cont_x′: ` Fall of the American Empire. The former is a biography of Andy Grove, the latter is a collection of his essays on`

The reversal rewrites the book subtitle's argument structure ("Rise of X" → "X and
Grove's Decline"); the X′ side even pattern-completes "Decline and **Fall** of the
American Empire". This is exactly the relational/structural edit class the trained
lexical types never show — the generalization target.

---

## Dataset stats (all 15,480 positions in meta_e.json)

- **Counts by type (train / holdout split):** antonym 4,877/1,298 · attribute 2,216/591 · discourse 1,518/346 · entity_swap 1,094/315 · role_reversal 2,565/660 (role_reversal entirely excluded from training by type). E3 trains on 9,705 lexical train positions → 19,410 examples with ±δ augmentation.
- **‖δ‖ quartiles (fp32, layer 20):** q25=17.6 · median=32.1 · q75=48.3
- **KL quartiles:** q25=0.19 · median=0.48 · q75=1.23
- **dist_from_edit quartiles:** q25=1 · median=4 · q75=17
