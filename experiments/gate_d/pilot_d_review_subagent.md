# Pilot D v2 label review (subagent pass)

Reviewed: all 100 labels in `pilot_d_100_v2.json` side-by-side with their
`bundles_d.json` evidence (counts below are over all 100 unless noted).
Prompt reviewed: `PROMPT` + `QUIET_PROMPT_SUFFIX` in `labeler_d.py`.

---

## (a) Issues, quantified

### 1. Future-token leakage — ~40/100 labels (45 flagged by string check, a few stopword false positives)

The `near` snippets in the evidence extend **past the labeled position**, and
Haiku freely quotes that future text as if it were retrieved earlier content.
The prompt never says the position is the final token of what exists, so the
model treats the right half of near-windows as fair game. This is the most
dangerous issue: the label encodes the ground-truth continuation, so a critic
trained on these labels can shortcut from "what came next" instead of "what
the write carried."

- pos 6150 (target ` says`): "the write captures **"says Kyle Dickerson" from
  further ahead in the text**, anchoring the attribution with the speaker's
  name **that follows " says" in the source**" — openly quotes the future.
- pos 3300 (target ` to`): "retrieves the nearby clause "the Chargers snapped
  it **to Eric Weddle**"" — "Eric Weddle" occurs only after the position.
- pos 13500 (target ` That`): "establishing what "That" is about to refer to
  and explain in the next clause (**"That fee structure creates"**)".
- pos 12300 (target ` (`): "preparing the model to generate **"SWCCCA"**" —
  the literal future completion, plus a "the model" mechanism leak.

Other clear cases: 2100 ("Tualatin forward **Emily**"), 6000 ("state capital
**Panaji**"), 7200 ("**Matthew Kroon**"), 3750 ("Grove Street, **Newton**"),
11850 ("key **winter wheat planting**"), 5700 ("come blow your **horn**").

### 2. Boilerplate template — 100/100 same skeleton

- 100/100 open "The write …" (94 "The write at …", 21 "The write at this …").
- 77/100 use "It also pulls in …" as the second-sentence opener.
- 40/100 close with a "Together, these retrievals …" summary sentence that
  adds no content.

Every label is S1 local-retrieval / S2 "It also pulls in" / S3 "Together…".
With a median length of 85 words, roughly a third of each label is structural
filler identical across positions — wasted critic capacity, and it makes
labels harder to tell apart, not easier.

### 3. Measurement-apparatus narration — 55/100 cite weights, 34/100 quote raw numbers

The mechanism-word ban works for head/layer/vector (0 violations) but the
labels narrate the *measurement* instead:

- "weight/weighted": 55/100; literal numbers/percentages: 34/100
  (pos 450: "(weighted 0.47–0.87 across multiple retrieval paths)";
  pos 12150: "with up to **77% weight** in one retrieval" — share/weight
  confusion; pos 11850: "(17%, 9%, 7%, 6%, and 6% shares)").
- "channel(s)": **8/100 — a direct violation of the prompt's own ban list**
  (pos 2400, 5550, 6900, 7800, 7950, 9150, 13200, 14550).
- "retrieval paths": 7/100; "share(s)": 11/100; "attention": 23/100.

Numbers and apparatus nouns are not content words; the critic reconstructs
from tokens, so this is dead weight that also homogenizes the labels.

### 4. Doc-start / resting-state tokens narrated instead of ignored — ≥25/100

The prompt says to *ignore* resting-state attention, but ≥25 labels quote the
document-opening token, and only ~6 of those dismiss it as noise — the rest
present it as content; either way it injects misleading tokens ("Welcome",
"PLAINVIEW", "Chat") into the label text the critic will embed.

- pos 900: "several retrievals bring in "Welcome" from the document opening,
  which appears to be a resting-state artifact with minimal semantic
  contribution" — a whole sentence to say nothing.
- pos 2550: "persistent attention to "PLAINVIEW" from the document opening,
  which appears to be a resting baseline rather than substantive content".
- pos 0: a full sentence on "the document delimiter "|" … likely serving as a
  structural anchor or resting state".

### 5. Generic filler vocabulary that can't distinguish positions

"anchor*" 47/100, "reinforc*" 40/100, "ground*" 35/100, "carry/-ing forward"
35/100, "sentence structure / grammatical / syntactic" 25/100, "local
context" 17/100. Sentences like "anchoring the local grammatical structure of
the sentence being completed" (pos 1800) or "carrying forward the local
phrase structure as the sentence continues" (pos 300) could be appended to
*any* position's label. For pure-local positions (pos 0, 1350, 1500, 4050)
the entire label reduces to restating the visible context in this vocabulary.

### 6. Quiet positions mishandled — 2/6 ignore the weakness, 6/6 exceed one sentence

Of the 6 quiet positions (4200, 6300, 7800, 9150, 9300, 9600): 6300 and 9300
never mention weakness at all (9300 reads like a normal confident label);
7800 buries "notably weak overall signal" in a final clause; all six run 3-4
sentences even though the suffix asks for one. Cause: the suffix is
conditional ("If the evidence shows no meaningful retrieval…") and the
evidence always shows *something*, so the condition never fires. Best of the
six is 9600, which states the weakness up front and notes what is *absent*.

### 7. Speculation despite the no-speculation rule — ~10/100

"suggest*" 10/100, "appears to be" 8/100, "the model / the network" 4/100
(also a mechanism leak): pos 600 "suggesting **the model** is tracking who is
associated with these corporate projects"; pos 5100 "suggests **the model**
is tracking the beginning of what will become "JCPenney"" (speculation +
future leak); pos 7200 "preparing **the network** for a new claim".

### 8. Mischaracterized evidence — occasional (~5 noted)

- pos 2700: ""View Full Version :" … with lower coherence" — one head reads
  it at weight 0.73, the dominant retrieval of that head.
- pos 1800: calls "history of astronomy" a "distant phrase from elsewhere in
  the text" — it is the position's own forward continuation (future spill).
- pos 12150 / 11850: weight-vs-share confusion (see issue 3).

BPE-fragment handling, by contrast, is mostly good: pos 2250 reassembles
"Cul"+"jak" → Culjak, pos 6750 "grand"→grandchildren, pos 2100 "T"/"ual"
→ Tualatin, pos 12600 " f"→"fathom" handled via near-context — no incidents
of fragments being read as standalone words in the 100 reviewed.

---

## (b) Prompt edits, ranked by expected impact

### Edit 1 (highest impact): stop the future leak

Add after the `<evidence>` explanation paragraph:

> "The position is the FINAL token of the source text — nothing after it has
> been read yet. The quoted surroundings ("in: …") may show a few words that
> come AFTER the position; those words are shown only to locate the source
> token. Never quote, name, or hint at any word that does not appear in the
> source text. If a source token sits at or just before the position itself,
> describe it as local carry-over without quoting what follows it."

And add to the FINAL CHECK: "scan every quoted phrase — if it contains words
not present in the source text above, delete or trim it."

(Also worth fixing upstream of the prompt: truncate `near` windows at the
position in `build_bundles_d.py` — then the leak becomes impossible rather
than discouraged. Highest-leverage single change in the whole pipeline.)

### Edit 2: ban the template and the measurement vocabulary

Replace the rule "Plain declarative sentences. Do not use the words …" with:

> "Plain declarative sentences. Do NOT open with 'The write at …' — start
> with the content itself (e.g., 'Pulls the company name "Atlas Copco" and
> the date "June 24" from the announcement…'). Vary sentence shapes; do not
> use the fixed sequence 'It also pulls in… / Together, these retrievals…'.
> Never write numeric weights, percentages, or shares — use at most 'mainly /
> also / faintly'. Do not use the words "layer", "vector", "representation",
> "neural", "head", "channel", "path", "share", "weight", "attention", or
> "retrieval" as a noun — write what the position now knows about earlier
> text."

And extend the FINAL CHECK list to: head, heads, channel, layer, vector,
weight, share, path, attention, plus "any number".

### Edit 3: doc-start tokens — omit, don't dismiss

Replace "weight on the very first token of the document is a resting state
and means nothing" + "ignore resting-state attention to the document start"
with:

> "Attention to the document's opening token or header line is a resting
> state: OMIT it completely. Do not mention it, quote it, or explain that you
> are ignoring it."

### Edit 4: make labels distinguishing; compress local carry-over

Add a rule:

> "Lead with the most specific retrieved content — names, places, numbers,
> topic words pulled from sentences BEFORE the current one. Summarize all
> retrievals of the position's own phrase in one short clause at the end
> ('plus carry-over of the local phrase "…"'). A reader holding 100 labels
> from other positions should be able to pick yours out; claims that fit any
> position ('anchors the local grammatical structure', 'reinforces the
> sentence being completed') are banned."

### Edit 5: fix the quiet suffix

Replace `QUIET_PROMPT_SUFFIX` with an unconditional instruction:

> "Note: the total write at this position is unusually weak (bottom decile).
> Say that first, in those terms ('weak write, little beyond local
> carry-over'), then at most one more sentence naming the single strongest
> genuine retrieval if there is one. Two sentences maximum."

### Edit 6: close the speculation gap

Extend the no-speculation rule: "Do not use 'suggesting', 'appears to',
'likely', 'preparing', 'setting up', 'about to', and never refer to 'the
model' or 'the network' as an agent."

---

## (c) Labels closest to ideal, and why

1. **pos 2250** (target `.\n` after "says Culjak"): *"…heavily retrieves the
   name "Culjak" from the immediately preceding context ("says Culjak"),
   pulling in fragments "Cul" and "jak" … retrieves "office" from the
   surrounding clause … reaches back to "Estes" and broader context about
   "Results Physical Therapy of Estes, LLC"…"* — every claim grounded,
   BPE fragments correctly reassembled and *named as fragments of a name*,
   and the concrete entities (Culjak, Estes, Results Physical Therapy) make
   this label uniquely identifiable among the 100. Would be ideal with the
   "The write at this position" opener and "with substantial weight" removed.

2. **pos 13050** (target ` winter`): *"heavily retrieves "winter ticks" from
   earlier passages discussing the problem to be addressed … retrieves
   "Yukon Elk" … winter ticks are the specific threat that prompted the need
   for a developed strategy."* — the one genuinely distant, high-weight
   content retrieval ("ticks" at 0.90 from an earlier mention) leads the
   label; topic words (elk, Yukon, harvest strategy) are concrete and
   position-specific. Trim the "(weights up to 0.90)" and it's the model of
   what a content-rich label should look like.

3. **pos 9600** (quiet, target ` HR`): *"The write at " HR" is weak and
   carries little beyond immediate local context … The position does not
   retrieve substantive earlier content about the bills themselves or Bruce
   Westerman's legislative record — it primarily reinforces the local list
   formatting…"* — states the weakness first, and the explicit *negative*
   ("does not retrieve … Westerman's record") is exactly the honest signal a
   critic needs to map weak writes to weak vectors. Should be two sentences,
   not four, but the shape is right.
