# Gate C: labeling-prompt variant experiment

2026-06-11. Six single-change variants of the production prompt (`hybrid_v2.py`
PROMPT), each run on 3 fixed test positions with `claude-haiku-4-5`,
temperature 0.7, max_tokens 600, 1 sample each. Runner:
`variants_run.py`; raw outputs: `variants_results.json`. Baseline outputs
taken from `hybrid_pre300.json` (same positions, production prompt).

## Test positions (chosen for rich diff_reads + entity bait)

| pos | token | context | entity bait in evidence |
|---|---|---|---|
| 1784 | ` user` | IBT banking-authentication marketing | "Internet Banking" (expl_b/reads; not in source) |
| 8311 | ` inefficient` | MH Equipment fleet-management page | Textron Forklift, Trail King, Fleetsuite (all fabricated in reads) |
| 399 | `ized` | GSP fatal-crash news article | Houston PD, Texas DPS (expls), NHTSA/NTSB (reads) — none in source |

Spot-check positions read for context: 9370, 9541, 2820.

Baseline behavior on the 3 fixed positions: 3/3 parse, 0 quotes, 0 entity
leaks, openers `Sharpened / Sharpened / Shift` (matching the corpus-wide
pattern: in all 300 baseline labels, "Sharpened …" opens 202 and the top-2
opening bigrams cover 134+62 = ~65%), median 608 chars.

Scoreboard (3 outputs per variant):

| variant | parse | quotes in label | entity leaks | opener diversity | mean len |
|---|---|---|---|---|---|
| baseline | 3/3 | 0 | 0 | low (2x Sharpened) | 706 |
| V-entity | 3/3 | 0 | 0 hard (1 borderline) | low | 886 |
| V-short | 3/3 | 0 | 0 | low | 412 |
| V-diverse | 3/3 | 0 | **2 of 3 leaked** | better (but "The" x2) | 719 |
| V-finaltoken | 3/3 | 0 after parse (2 in raw, template-caused) | 0 | low + fixed 3rd slot | 659 |
| V-concrete | 3/3 | 0 | 0 | low | 553 |
| V-magnitude | 3/3 | 0 | **1 of 3 leaked** | low | 580 |

---

## V-entity — harsher entity rule

Diff vs production (rule 2 replaced):

```
- - Never include a named entity (person, brand, team, title) unless it appears in the source text.
+ - Treat EVERY proper noun that appears in the earlier/later descriptions or the
+   difference-reads as fabricated unless that exact name appears in the source text.
+   Never write any proper noun (person, organization, agency, brand, place, product,
+   title) in your snippets unless it appears verbatim in the source text. If the
+   evidence converges on a category of entity, name the category (e.g. a federal
+   safety agency), never a specific name.
```

Outputs (trimmed):

- **1784**: "Sharpened framing of the token's role within a technical product description: the later representation contextualizes the closing word within a white paper or marketing document specifically about Internet Banking systems and authentication products… Strengthened expectation of immediate noun-phrase completion… credential-related compound nouns (such as name-password combinations or ID-verification pr…"
- **8311**: "Sharpened expectation that the incomplete phrase will resolve toward a noun describing operational or asset-related inefficiency (asset utilization, equipment scheduling, or similar fleet performance metrics)… Newly consolidated framing of the context as explicitly about fleet management software or technology solutions…"
- **399**: "Shift from law enforcement incident report framing toward technical/educational resource framing… Strengthened expectation of a formal investigative discipline or procedural category: both later reads converge on anticipating technical terminology related to accident reconstruction or crash analysis…"

Assessment: zero hard leaks, including on 399 where it correctly answered in
categories ("formal investigative discipline") instead of names — exactly the
intended behavior. One borderline case: "Internet Banking" (1784) is
capitalized and comes from the unreliable evidence, though it is arguably a
domain term, not an entity. Side effect: outputs got ~25% longer than baseline
(886 vs 706 mean) — the category-paraphrasing adds words; pair with a length
cap. Note 399's output also used the forbidden word "reads" once
("both later reads converge") — a pre-existing occasional rule slip, not
introduced by this change.

## V-short — exactly 2 snippets, max 40 words each

Diff vs production:

```
- … as 2-3 short text snippets.
+ … as exactly 2 snippets of at most 40 words each. Make every word carry
+   information; cut filler.
  (and the answer format drops the optional third-snippet line)
```

Outputs (trimmed):

- **1784**: "Strengthened framing of the closing clause as part of a structured list or enumeration of authentication components… Sharpened expectation that the token will be followed by a specific credential pairing (ID/password combination) rather than an open-ended noun phrase…" (365 ch)
- **8311**: "Sharpened expectation that the incomplete phrase… will be completed by a concrete operational noun (asset utilization, vehicle scheduling, equipment downtime)… Shifted framing from mid-enumeration of fleet problems toward identifying specific inefficiencies as a pathway to measurable cost savings…" (424 ch)
- **399**: "Strengthened framing of the token as part of a specialized investigative unit designation within law enforcement hierarchy… Sharpened expectation that the incomplete phrase requires a noun completing an official unit name, with heightened emphasis on crash reconstruction…" (447 ch)

Assessment: a clean ~40% length reduction (412 vs 706 mean) with essentially
no information loss — the concrete content (credential pairing, operational
metrics, unit designations) survived; what got cut was the "rather than…"
restatement padding. No leaks, full compliance with the 2-snippet cap. Does
nothing for opener diversity, and 2 slots conflicts with V-finaltoken's
required third slot — if composed, use "3 snippets, max 40 words each".

## V-diverse — forbid process-word openers, lead with content

Diff vs production (rule 5 replaced):

```
- - Write each snippet in change register: what is added, strengthened, sharpened,
-   or shifted (e.g. "Sharpened expectation of ...", "Newly consolidated framing of ...").
+ - Write each snippet in change register (what is added, strengthened, sharpened,
+   or shifted), but NEVER open a snippet with a process word such as Sharpened,
+   Strengthened, Shift, Shifted, Newly, Consolidated, or Emerging. Each snippet
+   must begin with the concrete topic itself - the content noun phrase whose
+   treatment changed (e.g. Banking authentication now framed as ..., The closing
+   clause now leans toward ...). All snippets must open with different words.
```

Outputs (trimmed):

- **1784**: "The authentication framework shifts from a general security context toward specific Internet Banking system implementation… The final token's grammatical position moves from completing a comparative clause… Marketing register intensifies from explanator…" (openers: The/The/Marketing)
- **8311**: "The enumeration of fleet optimization problems now emphasizes concrete operational metrics (asset utilization, vehicle scheduling, equipment downtime)… The contextual framing around the incomplete final token now positions it within a narrative of problem-solution progression…" (openers: The/The)
- **399**: "The investigative agency context shifts **from Houston Police Department toward Texas Department of Public Safety**, narrowing the institutional framing… The specialized unit designation moves… Technical and safety-focused content gains prominence…" (openers: The/The/Technical)

Assessment: it does break the Sharpened/Strengthened monoculture, but two
failure modes appeared verbatim. (1) "All snippets must open with different
words" was violated in 3/3 outputs — "The" just became the new formulaic
opener, which carries even less information than "Sharpened". (2) Worst
failure of the whole experiment: pos 399 stated fabricated entities as fact —
"from Houston Police Department toward Texas Department of Public Safety"
(source says GSP; those names come from expl_a/expl_b) — and 1784 surfaced
"Internet Banking". Leading with the content noun phrase actively invites
pulling concrete names out of the unreliable evidence. Reject as written;
the goal is reachable only if composed with the V-entity rule and an explicit
ban on opening with "The".

## V-finaltoken — required third snippet in the AR-native register

Diff vs production:

```
- … as 2-3 short text snippets.
+ … as exactly 3 short text snippets.
  and in the answer format:
- third snippet (optional)
+ Final token "{token}" ... now more strongly expecting ... (third snippet:
+ REQUIRED, in exactly this register - state what continuation the position
+ now expects more strongly than before)
```

Outputs (trimmed):

- **1784**: "Sharpened framing of the document as a white paper or technical specification format… Newly consolidated focus on authentication factors and credential components as a structured list… **Final token user is now more strongly expecting a noun phrase completing the authentication component being defined, such as credential pairing (ID/password combination)…**"
- **8311**: "…Newly consolidated positioning of the closing clause within a enumerated list of fleet management challenges… **Final token inefficient is now more strongly expecting a concrete fleet performance metric noun such as asset utilization, vehicle scheduling, or operation costs…**"
- **399**: "…Sharpened expectation that the incomplete departmental unit name will reference a formal investigative or reconstruction function… **Final token ized (the closing morpheme of the incomplete term beginning with Specialized) now more strongly expecting a continuation that completes a formal technical unit designation related to crash reconstruction…**"

Assessment: 3/3 produced the third snippet in exactly the requested register,
with grounded, position-specific continuation predictions — including handling
the subword token `ized` gracefully. This is the variant most likely to help
the critic, since its init (the released AR) was trained on a third-snippet
"Final token X … expecting …" format. One artifact: 2 quote characters
appeared in pos 399's raw output around `"ized"` — caused by my template
putting literal quotes around `{token}` in the format example (`parse()`
strips them, so the stored label is clean). Fix before adoption: write the
format line as `Final token {token} ... now more strongly expecting ...`
without quote marks.

## V-concrete — every snippet must name the content domain

Diff vs production (rule added after the banned-words rule):

```
+ - Every snippet must name the concrete subject matter that strengthened
+   (e.g. fleet maintenance costs, online banking authentication, probate
+   procedure). Purely abstract claims such as sharpened expectation of a
+   continuation or strengthened framing of the clause - with no domain
+   content named - are forbidden.
```

Outputs (trimmed):

- **1784**: "Sharpened focus on internet banking and online authentication systems as the specific domain… Strengthened framing of authentication credential components—particularly the pairing of user identity with password verification—as discrete, enumerable factors within multi-factor authentication architecture…"
- **8311**: "Sharpened focus on fleet management software and technology solutions as the primary lens for addressing operational inefficiencies… Strengthened expectation that the incomplete phrase will resolve into a specific fleet performance metric—asset utilization, vehicle scheduling, or equipment downtime…"
- **399**: "Strengthened framing of the investigative context toward technical accident reconstruction and crash analysis rather than general police procedural reporting… Sharpened expectation that the incomplete departmental unit name will reference a specialized traffic crash investigation or reconstruction division…"

Assessment: 3/3 named a concrete domain in every snippet, with no entity leaks
— notably 1784 rendered the bait as lowercase generic "internet banking"
(safe), and 399 stayed in category language. Slightly shorter than baseline
(553 mean) because the abstract restatement padding goes away when domain
content is mandatory. This targets exactly what the critic decodes (content,
not process words) and showed no downside in this sample. Openers remain
formulaic, so it does not solve diversity on its own.

## V-magnitude — add a magnitude field

Diff vs production (answer format only):

```
  <verdict>same</verdict> or <verdict>different</verdict>
+ <magnitude>small</magnitude> or <magnitude>moderate</magnitude> or
+ <magnitude>large</magnitude> (how large the overall change is)
  <difference>
```

Outputs (trimmed):

- **1784** (magnitude: small): "Sharpened framing of authentication as a structured list or enumeration of security components… Strengthened positioning of the token within a technical specification context emphasizing credential attributes…"
- **8311** (magnitude: moderate): "Sharpened expectation that the incomplete phrase will be completed by a concrete operational noun… Shift from fuel management and maintenance costs as primary enumerated concerns toward fleet utilization and asset efficiency metrics…"
- **399** (magnitude: moderate): "Shift from law enforcement local report framing **(Georgia State Patrol, Houston Police)** toward technical government safety documentation framing **(NHTSA, NTSB references)**… Strengthened expectation that the incomplete departmental unit name will be completed by a technical investigation specialty term…"

Assessment: the field parses cleanly 3/3 with a one-line regex
(`<magnitude>\s*(small|moderate|large)\s*</magnitude>`) and the values are at
least plausibly ordered, but with n=3 there is no way to validate calibration
against true diff-vector norms. Pos 399 leaked NHTSA/NTSB (fabricated in the
reads) and Houston Police as parenthetical fact — almost certainly the ~5%
baseline leak rate showing up at temperature 0.7 rather than an effect of the
magnitude field, but it is a reminder that the production entity rule still
lets parentheticals through. Harmless, cheap metadata; adopt only if critic
training will actually consume it, and validate calibration against
`diff_targets.npy` norms on the pre-300 set first.

---

## Ranked recommendation for the full batch

Adopt (compose — the changes touch disjoint parts of the prompt):

1. **V-finaltoken** (with the template fixed to put no quote marks around
   `{token}`). Highest expected downstream value: it restores the snippet
   format the critic's init was trained on, and all three outputs executed the
   register correctly with grounded continuation content.
2. **V-concrete**. Directly increases the density of decodable content per
   label, removed abstract-only padding, and showed zero leak/format cost.
3. **V-short, adapted**: with the required third slot, the cap becomes
   "exactly 3 snippets, max 40 words each" instead of 2 snippets. The 40-word
   cap alone removed ~40% of characters with no visible information loss, and
   counteracts the length growth that V-entity causes.
4. **V-entity**. Cheap insurance; produced the desired category-not-name
   behavior on the hardest position, and the V-diverse/V-magnitude leaks show
   the production rule is the weak point once any other pressure is applied.
   Keep the "name the category" sentence — it is what made 399 clean.

Reject:

- **V-diverse as written.** It traded the formulaic-opener problem for a worse
  one: fabricated entities stated as fact in 2/3 outputs, and the diversity it
  bought was mostly "The …" — near-zero information gain. If opener diversity
  is still wanted after composing 1-4, re-attempt as: ban process-word AND
  "The" openers, require the topic noun phrase first, and rely on the
  composed V-entity rule to suppress the leak pathway — then re-test before
  trusting it.

Defer:

- **V-magnitude.** Parses fine and costs nothing, but its value is zero unless
  the critic training consumes it, and calibration is unvalidated. If wanted,
  validate magnitude vs true diff-vector norm on the existing pre-300 labels
  before adding it to the full batch.

Risks of the composed prompt:

- The four adopted changes were each tested in isolation; their composition
  was not. Interactions are plausible (e.g. the 40-word cap squeezing the
  domain-naming requirement, or the third fixed-register snippet absorbing the
  content that snippet 2 used to carry). Run the composed prompt on a 20-30
  position pilot (including 1784/8311/399) and re-check the same six metrics
  before submitting the full batch.
- Opener uniformity is only partially addressed: snippets 1-2 will still tend
  to open with Sharpened/Strengthened. This is the one known issue with no
  safe single-change fix found in this experiment.
- The banned-word rule still slips occasionally ("both later reads converge",
  V-entity 399) — pre-existing, low frequency, harmless to the critic, not
  worth more prompt complexity.
