# Gate D arm-H pilot review — v4 labels (100 positions)

## What this is

`pilot_d_100_v4.json` holds 100 Claude-Haiku labels, one per sampled position,
each describing **what the layer-20 attention block of Qwen2.5-7B wrote into
the residual stream at that position**. The evidence behind each label
(`bundles_d.json`) is mechanically measured: per-head attention patterns
(which earlier tokens were read, with surrounding context) plus each head's
share of the write norm. The labeler prompt is `PROMPT` in `labeler_d.py`.

These labels will train a critic that maps label-text → write vector. This
document is the approval gate for the full ~14k-position batch.

**v2 → v4 mechanical deltas (per 100 pilot labels):**

| metric | v2 | v4 |
|---|---|---|
| future-token leak (label quotes text after the position) | ~40 | 1 |
| numeric attention weights echoed into label | 34 | 1 |
| boilerplate "The write at..." opener | 100 | 4 |
| quoted-word groundedness | 0.955 | 0.969 |
| labels containing ≥1 stray ban-list word | — | 40 |

**Structural fixes (not just prompt tweaks):** evidence context windows are
now truncated *at the position*, so a future-token leak is impossible by
construction (the v2 leaks came from evidence `near`-snippets spilling past
the position, not from the model hallucinating). Doc-start resting-state
attention sources were dropped from the bundles. Numeric weights were removed
from the evidence rendering (qualitative strong/moderate/faint instead).

The remaining 40/100 ban-list slips are almost all single occurrences of
"retrieval(s)", "attention", or "about to" — vocabulary the prompt itself uses
when stating the task — and are handled by a regen-on-violation pass at batch
time.

---

## Examples (14, chosen deliberately)

Categories: 2 quiet positions, several labels with a marked banned-word slip,
2 with rich distant retrievals, 3 weak/generic labels, the rest typical.
Evidence digest = top retrieved tokens across the strongest heads, with their
locating context.

### 1. i=0 — target `' really'` (typical)
**context tail (last ~100):** `…Week of February 11th| |Lil||Feb 1 2013, 09:58 AM| Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really`
**evidence:** ` really` ← "Sami, really" (strong); `,` ← "about Sami, really" (strong); ` care` ← ". Don't care about Sami"; ` about` ← "Don't care about Sami,"
**label:** The comma and "really" from the immediate context ("Sami, really") are pulled strongly across multiple reads, along with "care" and "about" from the parallel negation structure ("Don't care about Sami"). The write anchors the completion of the second negative clause by binding the emphatic adverb to the subject and the negated verb that precedes it, mirroring the rhythm of "Don't care about Chloe/Taniel/Jen-Jen" from the opening of the sentence.

### 2. i=6000 — target `' capital'` (richest distant retrievals)
**context tail:** `…charges,” Judge Vandana Tendulkar said after the trial of Samson D’Souza and Placido Carvalho in the state capital`
**evidence:** ` Goa` ← "found on a Goa beach in" (far back in doc); ` court` ← "– An Indian court on Friday cleared" (doc opening); `I` ← "PANAJI (Web Desk" (dateline, doc start); ` capital` ← "in the state capital" (local)
**label:** The write pulls "Goa" from the opening narrative ("found on a Goa beach in 2008") and "court" from the earlier framing ("An Indian court on Friday cleared"), anchoring the legal proceeding in its geographic and institutional context. It also retrieves "state" and "I" (from "PANAJI" and "Judge Vandana Tendulkar said"), grounding the verdict in the state jurisdiction where the trial occurred. The local phrase "in the state capital" is carried forward to complete the locational specification of where Judge Tendulkar delivered her judgment.

*Note: opens with "The write" — one of the 4 residual template openers. Otherwise the strongest label in the set: dateline, court, and state retrieved from 200+ chars back, all verifiable in the evidence.*

### 3. i=4500 — target `'.\n'` (rich distant retrievals; banned-word slip: **attention**)
**context tail:** `…w partnerships with US based production partners, design agencies and leaders in the field of additive technology. ⏎`
**evidence:** `.\n` ← "of additive technology."; `).\n` ← "that soon!).\nThis move can"; `.\n` ← "Brooklyn Navy Yard.\nOur expansion into" (distant); `,` ← "of New Lab, a renovated warehouse"
**label:** Punctuation marks—especially the closing ")." from "that soon!)." and the period from "Brooklyn Navy Yard."—dominate the reads, grounding the write in sentence-boundary structure. The **attention** also pulls "This" and "move" from the opening of the next sentence ("This move can enable"), along with scattered references to geographic and partnership content ("US based production partners", "Brooklyn", "New York") that contextualize the expansion narrative. Together these reads anchor the period as a major structural boundary while reinforcing the thematic continuity of locations and strategic moves in the passage.

*Note: "the opening of the next sentence" is a faithful description of an earlier "This move can enable" occurrence in the evidence, not a future leak — but the phrasing reads like one. Worth a prompt nudge.*

### 4. i=9000 — target `' “'` (typical — quote closure)
**context tail:** `…d loser,” when asked about the possibility of the November Presidential election results not being in his favour. “`
**evidence:** ` “` ← "his favour. “" (strong); ` that` ← ", Chris Wallace that he is “"; ` loser` ← "not a good loser,” when asked"; ` Trump` ← "0, Donald Trump, President of"
**label:** The closing quotation mark pulls together the phrase "not a good loser," — the core claim Trump made — along with the preceding context "his favour. " that frames the conditional stakes of the election. Multiple reads of "loser," "good," and the comma scaffold the grammatical closure of the direct quote, while "that he is" anchors the attribution structure back to Trump's statement. This write completes the quoted material and the sentence structure that reports what Trump told Chris Wallace.

### 5. i=13500 — target `' That'` (typical — content numbers, correctly kept)
**context tail:** `…management fee levied on assets, hedge funds typically keep 20% of the profits generated each year as payment. That`
**evidence:** ` That` ← "as payment. That"; ` of` ← "20% of the profits generated"; ` fee` ← "is the performance fee. Along with"; ` profits` ← "% of the profits generated each year"
**label:** The specific fee structure—"20% of the profits generated each year as payment"—anchors the write, with strong pulls on "profits," "of," and "payment" establishing the quantitative claim being completed. The broader context of performance fees appears alongside these pulls: "the performance fee" and the management structure ("Along with a 1% to 2% management fee") ground the sentence's topic. The write also catches the transitional marker "That" itself, completing the local phrase structure that opens the next clause.

*Note: "20%", "1% to 2%" are document content, not attention weights — the number rule correctly distinguishes these.*

### 6. i=7500 — target `' ('` (typical — subword carry)
**context tail:** `…dard), two leading standards which are commonly used in concert to verify combined carbon offsets and co-benefits (`
**evidence:** `-b`/`enef`/`its`/` co` ← "carbon offsets and co-benefits (" (strong, every head); ` Community` ← "the Climate, Community and Biod" (distant)
**label:** The phrase "co-benefits" and its components ("co," "enef," "-b," "its") dominate the retrieved tokens, pulled repeatedly from the nearby phrase "carbon offsets and co-benefits." "Community" also registers from the earlier clause naming the "Climate, Community and Biodiversity Standard (CCB Standard)." This write locks onto the specific term that pairs with the standards being validated, reinforcing what MCDI expects those standards to verify alongside carbon offsets as the sentence reaches the opening parenthesis.

### 7. i=2850 — target `' one'` (typical; residual "The write" opener)
**context tail:** `…utage. A fire erupted and damaged the home at 1113 Delaware Street, adjacent to the power lines in question and one`
**evidence:** `,` ← "3 Delaware Street, adjacent to the"; ` home` ← "and damaged the home at 1"; ` adjacent` ← "Delaware Street, adjacent to the power"; ` one` ← "in question and one"
**label:** The write pulls "home at 1113 Delaware Street, adjacent to the power" — the specific damaged property and its location relative to the infrastructure — along with the connective phrase "and" that links the fire damage back to the seagull incident. It also reads "one" from the immediate context "in question and one", reinforcing the local grammatical structure as the sentence moves toward completion.

### 8. i=4200 — target `' Step'` (quiet, clean)
**context tail:** `…week? Just wondering. ⏎ Anyway. The astrologer’s club’s campaign is fascinatingly evil: ⏎ The campaign will proceed as follows: ⏎ * Step`
**evidence:** `*` ← "as follows:\n* Step" (strong); ` Step` ← "follows:\n* Step" (strong); `:\n` ← "proceed as follows:"; ` follows` ← "will proceed as follows:"
**label:** This is a weak write, carrying little beyond local carry-over of the bullet structure ("*") and the immediate phrase ("Step"). The asterisk and colon marking the list format ("as follows: *") anchor the position within the enumeration frame, but no substantive content from earlier in the passage—neither the partisan media critique nor the "astrologer's club's campaign"—is being carried forward into what comes next.

*Note: correctly leads with the weak-write framing (quiet suffix), and the negative claim — naming what is NOT retrieved — is actually position-specific and checkable.*

### 9. i=9600 — target `' HR'` (quiet; banned-word slip: **retrieval**)
**context tail:** `…day we are ending this series by examining Rep. ⏎ Bruce Westerman’s legislation. ⏎ Here are some of the 9 bills he has sponsored: ⏎ - HR`
**evidence:** ` HR` ← "sponsored:\n- HR" (strong); `-` ← "has sponsored:\n- HR" (strong); ` sponsored` ← "bills he has sponsored:"; ` legislation` ← "Westerman’s legislation.\nHere are"
**label:** This write is weak, carrying little beyond local carry-over of the dash and "HR" from the immediately preceding list structure. The strongest genuine **retrieval** pulls "sponsored" from "bills he has sponsored," anchoring the context that these are legislative items authored by Bruce Westerman.

### 10. i=9900 — target `' predictions'` (heaviest banned-word slips: **heads, attention, retrievals**)
**context tail:** `…obal events have on the coffee supply chain. ⏎ Daily Production Predictions ⏎ Stay ahead of the curve with daily crop size predictions`
**evidence:** ` size` ← "with daily crop size predictions" (strong); ` predictions` ← "daily crop size predictions" (strong); ` daily` ← "the curve with daily crop size"; `\n` ← "Production Predictions\nStay ahead of" (header)
**label:** The phrase "daily crop size" from the immediately preceding clause anchors the completion, with "size" and "daily" read most strongly across multiple **attention heads**, while "with" and "curve" from the context "stay ahead of the curve with daily crop size" also pull through to ground the noun phrase. A secondary thread picks up the section header "Production Predictions" and the period ending "Daily Production Predictions," marking the boundary between the feature title and its supporting copy. Together these **retrievals** complete the parallel structure of the feature announcement by assembling the full nominal phrase "Daily Production Predictions" while maintaining semantic cohesion with the actionable claim immediately before it.

### 11. i=14850 — target `' '` (banned-word slip: **likely**; ungrounded speculation)
**context tail:** `…ile the owners undertake a multimillion-dollar renovation of 211 guest rooms. ⏎ The 346-room beachfront hotel will reopen Jan. 8 at `
**evidence:** `1` ← "211 guest rooms." (strong); ` ` ← "8 at" (strong); ` rooms` ← "11 guest rooms.\nThe" (strong); ` at` ← ". 8 at" (strong)
**label:** The reopening date "Jan. 8" and the phrase "at" anchor the completion of the sentence structure "will reopen Jan. 8 at." Alongside this, numerals from earlier in the text—"211" guest rooms and "346" beachfront rooms—are pulled in, **likely to maintain numerical coherence** within the property description. The word "rooms" echoes the guest room context, reinforcing the hotel's scale as the sentence moves toward specifying the reopening condition or venue.

*Note: "likely to maintain numerical coherence" is interpretation the evidence cannot support — the exact category the hedging ban was meant to kill. The retrieved tokens themselves are accurately reported.*

### 12. i=12150 — target `','` (weakest/most generic; banned-word slip: **preparing**)
**context tail:** `…child in our Saturday Player Training Camps. For a minimal fee of $50.00 your child will learn principles of defending, attaching,`
**evidence:** ` defending` ← "learn principles of defending, attaching" (strong); ` attaching` ← "of defending, attaching," (strong); `,` ← "defending, attaching," (strong)
**label:** The tokens "defending" and "attaching" from the parallel list "principles of defending, attaching," are strongly pulled forward, along with their surrounding commas, to anchor the grammatical structure at the current position. This write gathers the two skill categories that the child will learn in the training camp, reinforcing the list pattern and **preparing** the close of this enumerated phrase.

### 13. i=11550 — target `':\n'` (generic; banned-word slip: **attention**)
**context tail:** `…he basin and explore how water is affected when there are changes in the basin. ⏎ - Course Type: ⏎ - Learning Module ⏎ - Material Type: ⏎`
**evidence:** ` Type` ← "- Material Type:" (strong); `:\n` ← "- Material Type:" (strong); `-` ← "the basin.\n- Course Type:"; `:\n` ← "- Course Type:\n- Learning Module"
**label:** The structure "- Material Type:" is pulled strongly and repeatedly across multiple **attention** reads, anchoring the list format that the current position completes. The colons and dashes from both "- Course Type:" and "- Learning Module" also arrive, reinforcing the parallel list pattern where each label ends with punctuation. This write carries the template for the metadata field being filled in—the label name and its terminating colon.

*Note: faithful to the evidence (the write really is pure list-template), but strip the quoted field names and this label fits any list-formatted position — see assessment.*

### 14. i=11100 — target `' a'` (restates local tail; misses a real distant retrieval; "The write" opener)
**context tail:** `…with practical solutions. We are one of the world’s largest environmental organizations, with more than two million members and a`
**evidence:** ` organizations` ← "’s largest environmental organizations," (strong); ` a` ← "million members and a" (strong); ` members` ← "than two million members and a" (strong); ` Fund` ← "**, Environmental Defense Fund (EDF**" (distant, two heads)
**label:** The write pulls "organizations" and "members" from the nearby phrase "world's largest environmental organizations, with more than two million members," anchoring the enumeration of the group's scale. It also reads the article "a" itself from "members and a," continuing the grammatical setup for the noun phrase that will complete the sentence. Together these retrievals ground the indefinite article in both the cataloging context (size, membership count) and the immediate syntactic frame.

*Note: the evidence shows two heads reaching back to "Environmental Defense Fund (EDF)" — the one piece of content not visible in the immediate tail — and the label omits it entirely while spending three sentences on the locally visible phrase. This is the residual failure mode that matters most for the critic.*

---

## Quality assessment

Would I approve this batch? **Yes, with the planned regen-on-violation pass and
one prompt tweak** — but with eyes open about what these labels are. The
mechanical hygiene is genuinely fixed: leaks are structurally impossible now,
weights are gone from the rendering, groundedness is high (the one sub-0.9
groundedness outlier, i=7950, is an artifact of inch-marks `2"W` inside quoted
dimensions, not a real fabrication), and quoted content checks out against
the bundles essentially everywhere I spot-checked. The residual issues are of
a different, softer kind. **(1) Local restatement crowds out distant
retrievals**: because layer-20 attention genuinely is mostly local, most
labels lead with the position's own phrase — faithfully — but in cases like
i=11100 the label spends all its budget restating the visible tail and drops
the *only* non-obvious retrieval in the evidence (EDF). The critic gets its
discriminative signal from exactly those non-local tokens, so this is the
costliest failure mode; consider a prompt line like "if any read reaches
before the current sentence, it must appear in your label." **(2) Heavy stock
scaffolding**: "anchor" appears in 91/100 labels, "pull" in 90/100,
"complet-" in 80/100, "the write" mid-label in 55/100, "grammatical" in
39/100. Strip the quoted tokens and many labels at structurally similar
positions (list items i=11550/9600/4200, sentence boundaries i=4500/5250) are
near-interchangeable — the distinguishing content lives almost entirely in
the quoted words, which is acceptable for a text→vector critic but means the
connective prose is close to no-op filler. **(3) Soft speculation survives**:
~15/100 labels carry a future-facing clause ("likely to maintain numerical
coherence", "what follows will describe", "ready the structure for what This
will introduce next") — never quoting unseen tokens, so not leaks, but
interpretation the evidence can't support. The 40/100 ban-list slips are
single-word and mechanical (mostly "retrieval(s)"/"attention", vocabulary the
prompt itself uses); the regen pass handles them. None of this blocks the
batch; (1) is the only one I'd actually change the prompt for before
spending the ~14k calls.
