# Hybrid diff-label prompt: v2 (composed) vs v3 — pilot-30 comparison

Same 30 positions, same evidence (`hybrid_inputs_pilot30.json`). v2 = `hybrid_pilot30.json`
(mandatory "Final token X now more strongly expecting…" third snippet, 40-word/snippet cap,
verdict field, quotes mechanically stripped). v3 = `hybrid_pilot30_v3.json` (no verdict, no
mandatory final-token snippet, 50-word cap, confidence-then-importance ordering, sparing
source-only quoting allowed).

## Stats (n = 30)

| Metric | v2 | v3 |
|---|---|---|
| Mean length (words) | 85.0 (69–103) | 98.5 (72–128), +16% |
| Snippets per label | 3/3/3 always | 3/3/3 always |
| Fabricated entities (proper noun in neither source nor evidence) | **0** | **0** |
| Entity mentions from evidence only (not in source) | ~18 hits / 8 labels | ~26 hits / 11 labels |
| Quoted spans | 0 (stripped by design) | 41 total: 19 verbatim in source, 20 quoting evidence text, 2 near-paraphrases verbatim nowhere (pos 6616: "in the log,", "of the history,") |
| Snippets naming a concrete content domain (judgment) | ~75/90 (83%) | ~80/90 (89%) |
| Format: tags well-formed | 30/30 | 30/30 parse, but 5/30 missing closing `</difference>` (4026, 6616, 945, 5606, 1784); none truncated — closer just omitted |
| Verdict field | 30/30 = "different" (degenerate, zero bits) | absent by design |
| v2 final-token snippet: informative / moderate / filler | 25 / 4 / 1 (filler: 9678) | — |
| v3 retains a completion-expectation clause | — | 29/30 (missing only at 479) |
| v3 uses literal phrase "final token" | — | 3/30 |

**Head-to-head quality (my judgment, all four dimensions weighed):**
v3 clearly better: 3 (4026, 8417, 479) · v3 slightly better: 9 · tie: 12 ·
v2 slightly better: 4 (945, 9300, 872, 5606) · v2 clearly better: 2 (725, 399).

**Did dropping the mandatory final-token snippet lose information?** Mostly no. 25/30 of
v2's final-token snippets carried real signal (concrete expected continuations), but v3
re-expresses equivalent completion-expectation content inside its free-form snippets in
29/30 labels — only 479 has no next-token expectation at all. What v3 *did* lose is the
surface form: the literal "Final token X…" template appears in 3/30 v3 labels vs 30/30 in
v2, and the critic init was trained on final-token-style text. That is a distribution-shift
risk in wording, not in content.

## Side-by-side examples

### 1. v3 clearly better — pos 4026 (token " simple", tamalada source)

Source is about a **tamalada** (Hispanic tamale-making tradition); all evidence
hallucinated Christmas/French-feast cooking.

> **v2:** "Sharpened framing of the cooking activity as overcoming initial skepticism …
> Strengthened association of the domestic cooking context with specific cultural or
> seasonal traditions, moving … toward holiday celebration and family feast contexts.
> Final token simple now more strongly expecting continuation that emphasizes accessibility
> or manageability …"

> **v3:** "Sharpened focus on cultural/ethnic food traditions as the core subject matter …
> the later layer consolidates emphasis on a specific cultural practice (**tamalada, a
> Hispanic tradition**) as the meaningful context … Newly consolidated framing of holiday
> or festive gathering traditions … celebration, tradition, and social gathering as the
> primary semantic domain."

v3 anchors the diff to the true source content; v2 stays entirely inside the hallucinated
holiday domain. Same pattern at 479 (cites the source's rates/facilities/reviews structure)
and 8417, where v3 alone reports a **weakening** ("Weakened connection to the original
product-promotion context (Roboform/form-filling)") — a direction v2's template never
expresses.

### 2. v2 better — pos 725 (token " tips", fruit-baking source)

The four diff_reads consistently shift toward *iterative trials and promised follow-up
feedback* on a new ingredient. v2 synthesizes exactly that:

> **v2:** "Sharpened focus on experimental methodology: … initial trials and iterative
> testing cycles before offering feedback … now more strongly anticipates a second round
> of evaluation and documented results after the first attempt …"

> **v3:** "Sharpened framing of the closing clause as part of a **gardening and
> food-growing context** … advice or observations about growing, harvesting, or preserving
> homegrown ingredients from the garden …"

v3's gardening claim comes from expl_b alone; no diff_read mentions gardening. It
paraphrased the wrong evidence stream. Similar miss at 399, where v2 correctly reports the
diff_reads' shift toward NTSB/government-agency framing while v3 calls it a "formal law
enforcement report structure," underplaying the shift the reads actually show.

### 3. Typical — pos 945 (token "1", K'NEX scale-model source)

> **v2:** "… Newly consolidated expectation that the token continues a scale notation
> (e.g., 1:18 or 1:32) … Final token 1 now more strongly expecting … the opening digit of
> a scale ratio (such as 1:18 or 1:24) …"

> **v3:** "Sharpened focus on scale ratios as numerical specifications: … more strongly
> expects a fractional scale designation (like \"1:18\") … Strengthened association with
> model vehicle dimensions … Refined expectation of measurement language in product
> specification clauses … more tightly bound to phrases describing scale factors …"

Identical signal; v3 is ~20 words longer and its third snippet restates the first. The
length increase in v3 is mostly this kind of restatement (also 9300, 3300).

### 4. Typical — pos 6616 (token " started", log-retention boilerplate)

> **v2:** "Strengthened framing of retention policy configuration … Consolidated focus on
> numeric thresholds and duration specifications … Final token started now more strongly
> expecting completion with phrases describing log storage duration, retention table
> specifications, or temporal boundary conditions …"

> **v3:** "… shifts from describing record deletion patterns toward describing
> duration-based retention policies … will be completed with continuations related to log
> storage duration or retention timeout periods (e.g., \"in the log,\" \"of the
> history,\") … narrows to specific temporal constraints … moves from general
> Excel/data-checking contexts toward IBM SAS or similar enterprise analytics software
> documentation …"

Both good. v3 is denser on the earlier-vs-later contrast and names the software domain,
but its quoted example continuations are lifted (and lightly mangled) from expl_b, not the
source — the "sparing source-only quoting" rule was not followed.

## Verdict: adopt v3, with tweaks

v3 wins on the dimensions that matter for the critic: better source grounding (it pulls
real anchors — tamalada, Sinn Féin councillor, Roboform, "Week 17", "driving environmental
force" — where v2 never does), slightly higher domain concreteness, it can express
weakening/de-emphasis, and dropping the verdict costs nothing (v2 emitted "different"
30/30 — zero information). Fabrication is zero in both. Head-to-head v3 wins 12 positions
to v2's 6, with 12 ties. The feared information loss from dropping the mandatory
final-token snippet did not materialize: 29/30 v3 labels still state a concrete
completion expectation. Tweaks before the full run:

1. **Fix the quoting rule.** Half of v3's quotes (22/41) are not verbatim source, mostly
   quoting hallucinated evidence as "e.g." continuations, and twice paraphrase-mangled.
   Either instruct "quote marks only for spans copied verbatim from the source; give
   candidate continuations without quote marks," or drop quoting permission again.
2. **Keep a soft final-token cue.** One snippet should state what the final token now
   expects, ideally using the words "final token" — content survived, but the critic init
   was trained on that surface form, and it costs nothing to keep (29/30 already comply in
   substance; only the phrasing drifted).
3. **Tighten length.** Mean grew 85 → 98.5 words and the marginal words are mostly
   third-snippet restatements (945, 9300, 3300). Either restore a ~40-word cap or add
   "do not restate an earlier snippet."
4. **Require the closing tag** (5/30 omitted `</difference>`) or keep a lenient extractor.
5. Watch the expl_b-paraphrase failure mode (725, 399): consider one line in the prompt —
   "claims about the *direction of change* must be supported by the diff readings, not by
   the second explanation alone."
