# Gate C — worked examples

## Legend

We trained "critic" models to map a text label to the layer-22-minus-layer-18
activation diff of Qwen2.5-7B at a FineWeb position. Three label sources
("arms"):

- **arm0p** — the two *content explanations* (one of the layer-18 activation,
  one of the layer-22 activation) concatenated as `Earlier: ...\n\nLater: ...`.
  No model ever looked at the diff vector itself.
- **armA** — a diff-injection AV's own verbalization of the diff vector
  (the model reads `v22 − v18` directly and describes it).
- **armB** — a Claude-Haiku-written hybrid label synthesizing the two content
  explanations plus four diff readings into a "what changed" summary.

Two metrics per arm, both cosine similarity between the critic's predicted
vector and a target:

- **cos vs residual** — target is the diff *residualized* against a ridge
  prediction from the layer-18 activation, i.e. only the part of the diff
  that is NOT linearly predictable from position content. This is the
  headline metric (higher = the label carries diff-specific information).
- **cos vs raw diff** — target is the raw `v22 − v18`. This is much easier
  because most of the raw diff is predictable from content alone.

Examples drawn from the 144 positions present in all three arms' holdout
sets. Mean cos-vs-residual over those positions: arm0p 0.304, armA 0.276,
armB 0.160.

---

## arm0p clearly better than armA

### Example 1 — tidx 647 (token `0`, mid-year "20…")

> …purchased the Contractor Division of Shurtleff. Shurtleff and
> Washurn-Linder can each trace their roots back to the 1800's! The newest
> member of our Company, Lenape Industries, acquired in 2>>0<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | **0.182** | 0.741 |
| armA | −0.036 | 0.532 |
| armB | 0.123 | 0.705 |

**arm0p label:**

> Earlier: Corporate/educational brochure format with descriptive product descriptions detailing three farm divisions and their acquisitions, establishing chronological history and ownership structure.
>
> The sentence structure "Additionally, the company also acquired two other companies in 2010 including a dairy product supplier and a consulting firm in 2015" mirrors a timeline pattern, implying the final acquisition year "20" is continuing.
>
> Final token "20" is mid-number in "acquired in 20" — part of the third acquisition timeline item ("Another acquisition made in 20"), strongly expecting "15" or "17" to complete the year, continuing the company's acquisitions of financial assets.
>
> Later: Business/educational brochure format with product description listing company history and product lines for a Pennsylvania concrete manufacturing firm, detailing sales and acquisitions over time.
>
> The sentence structure "Our Sales Division began its own independent consulting firm in 2007 and acquired several other companies in 2011" implies a second date completion, likely referencing a recent acquisition or financial year update.
>
> Final token "20" is mid-number in "in 20", part of a date sequence ("Acquired in 2016, our Research Division was acquired in 20"), strongly expecting "17" or "18," completing the latest acquisition year or data timeframe.

**armA label:**

> American political/educational writer presenting a comprehensive report format with charts and data on their company's product line growth, suggesting a concluding date or timeline statement.
>
> The phrase "My company's latest report from January 2021 to February 2022, excluding the 'Basic' category" implies a question about the company's current status or a future dated version, likely a number.
>
> Final token "2022" ends mid-expression ("2021, Question: 2021"), strongly expecting "s," "the," or "solutions," or "June)" completing a date-based claim about combined sales figures and intellectual property status.

**armB label:**

> Sharpened expectation that the incomplete year "20" refers to a concrete manufacturing or industrial production context, rather than generic corporate acquisitions. The later reading consolidates focus on manufacturing operations and material products.
>
> Strengthened association between the acquisition date and a specific recent timeframe (2016-2018 range), narrowing the earlier layer's broader timeline expectations and grounding the narrative in a more recent business history rather than distributed historical acquisitions across decades.
>
> Newly consolidated framing of the document as a factual business report with specific geographic or operational anchoring (Pennsylvania manufacturing reference), moving from a generic corporate brochure toward a more particular industrial sector profile.

*Stands out: armA's label drifts into hallucinated specifics ("January 2021 to February 2022") and its residual cosine goes negative, while arm0p's two grounded content readings still score 0.18.*

### Example 2 — tidx 711 (token `.\n`, end of AP copyright notice)

> …Homicides there have more than double over the past two years amid a turf
> war between the Sinaloa and Zetas cartels. Copyright 2013 The Associated
> Press. All rights reserved. This material may not be published, broadcast,
> rewritten or redistributed>>.<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | **0.283** | 0.602 |
| armA | 0.086 | 0.355 |
| armB | 0.170 | 0.429 |

**arm0p label:**

> Earlier: News headline format with sports video description structure, showing NFL news item with caption and video details.
>
> The phrase "Copyright 2012 CNN. All Rights Reserved. This story includes a video of the tornado that killed 11 people..." suggests a closing disclaimer or metadata block, likely a news article template or website footer, implying the article concludes or a related story follows.
>
> Final token "rights.
> " closes a copyright/legal notice clause ("Copyright 2012 CNN. All Rights Reserved. This story includes..."), strongly expecting continuation like "The Associated Press" or "For more information" or "This story is no longer available" or similar closing punctuation.
>
> Later: News headline format with AP wire service style, a TV station's sports section announcing a death sentence for Hurricane Sandy.
>
> The phrase "The Associated Press story requires this disclaimer: ESPN.com's death toll count ends, but the NFL's official website will continue to track deaths and injuries..." suggests a closing quotation or standard news format, implying a closing sentence or metadata about the broadcast's copyright or news rules.
>
> Final token "of.
> " closes a quoted phrase ("This story and all related stories... and all other news items..."), strongly expecting "the story" or "This story's headline" or "You can now" or "a new headline format" to complete the disclaimer clause.

**armA label:**

> AP-style news headline format with Python code block showing date/time attributes, following a standard Yahoo Sports template pattern about a web site.
>
> The phrase "This is to ensure a 'live' label is shown, but the real thing isn't... The following is the standard format for a 'sports' name" is a common Python exception message template, referencing a structured attribute list for a named entity.
>
> Final token "attribute "
> ends mid-quote ("The way we do things - the name of this page ... but the following attributes aren't"), strongly expecting "your message's" or "._all_.html" or "the player's death cycle" or "your message to be valid."

**armB label:**

> Strengthened association with wire service attribution and legal metadata. The closing period now more sharply anchors to copyright notice conventions (specifically "The Associated Press" framing) rather than generic news template structures.
>
> Sharpened expectation that the text concludes a formal disclaimer clause rather than continuing into additional content. The final token now signals closure of a bounded legal/attribution statement rather than an open-ended template.
>
> Newly consolidated framing of the text as AP wire service material specifically, rather than undifferentiated news headline format. The later representation more distinctly resolves the source's genre toward established newswire conventions.

*Stands out: armA mistakes the legal boilerplate for a Python code template; arm0p's "copyright/legal notice closing" reading wins on both metrics.*

---

## armA clearly better than arm0p

### Example 3 — tidx 355 (token ` The`, start of company name)

> …An investigation by the department determined that The Hartford had
> inaccurately calculated the value of total vehicle losses for 77 claims
> between January 2009 and September 2011. In response,>> The<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | 0.102 | 0.692 |
| armA | **0.210** | 0.707 |
| armB | −0.033 | 0.544 |

**arm0p label:**

> Earlier: News release format with financial regulatory reporting structure, describing a settlement agreement between Nationwide Insurance and a Pennsylvania insurance company over consumer complaints.
>
> The sentence "The company agreed to pay $1.2 million to resolve violations, and Amerisafe has agreed to pay penalties to the Insurance Department. The agreement" establishes a pattern suggesting the company's compliance terms and penalties are being outlined.
>
> Final token "The" is mid-noun phrase ("Thus, The"), directly continuing the second clause ("As a result, The Insurance Company has agreed to pay penalties..."), strongly expecting "Insurance Company has agreed to" or "Eaton has agreed to a corrective action plan" or "Company must refund."
>
> Later: News article format with newspaper reporting tone detailing insurance company's consumer complaints settlement, establishing factual narrative about State Farm's compliance actions.
>
> The sentence structure "A state insurance regulator has agreed to pay fines for State Farm's violations, and The Travelers Insurance has agreed to disclose its financial practices. " implies a parallel clause about Nationwide's compliance agreement, completing the company's formal settlement obligations.
>
> Final token "The" is mid-phrase ("The Insurance Council has agreed... And The"), directly continuing a noun phrase referencing Nationwide's settlement agreement, strongly expecting "Travelers Insurance" or "Company would pay a fund" or "Nationwide has filed a compliance fund," completing the legal explanation.

**armA label:**

> News article format with Q&A structure about a Tennessee newspaper's branding, using math humor to illustrate social concepts about corporations and laws.
>
> The sentence "To buy into a company's culture, you'd expect all its policies to be consistent. But if you want to know all insurers, The State Farm?" establishes a parallel question about the insurance company's "Safeco" name, implying the Corporate Registry.
>
> Final token "The" is mid-phrase ("But not all policies and The"), part of a specific noun phrase completion ("To change that, either all insurers or The"), strongly expecting "Safeco" or "Corporate Registry" or "Company would."

**armB label:**

> Strengthened framing of the settlement narrative as a formal legal compliance obligation: the token now more strongly anchors expectations toward specific insurance company names and their agreed-upon penalty or corrective action, moving from generic "company compliance" to particular institutional actors and their binding commitments.
>
> Sharpened expectation that the next clause will name a specific insurance entity subject to regulatory agreement, rather than remaining in abstract penalty description. The earlier layer expected generic compliance language; the later layer expects a proper noun (company name) to complete the sentence structure.
>
> Consolidated shift from descriptive regulatory reporting toward newspaper-style factual narration of settlement terms: the token now positions itself within a more concrete, named-entity-driven narrative of which specific firms agreed to which specific obligations, departing from the earlier layer's vaguer pattern of "company has agreed to pay."

*Stands out: armA's label is noisier prose but its emphasis on "specific company name comes next" apparently captures more of the residual diff than arm0p's two near-duplicate content readings.*

### Example 4 — tidx 221 (token `.'`, closing a teacher-heroism quote)

> …"The talk of heroism, this is not mythology,'' Robinson said. "They were
> doing what teachers would do to protect their children>>.'<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | 0.220 | **0.551** |
| armA | **0.314** | 0.504 |
| armB | −0.006 | 0.305 |

**arm0p label:**

> Earlier: News report format with official testimony quotes from FBI, establishing factual reporting context around a Senate hearing on police casualties.
>
> The sentence structure "The union leader said the women were defending themselves and others during the shooting. 'The teachers did what any mother would do in a crisis.' " completes a quoted statement explaining heroic actions of teachers during the massacre, suggesting a concluding or reinforcing clause about justified fear/behavior.
>
> Final token ".' " closes a direct quote mid-sentence (" 'Teachers said 'Women teachers 'protected their students and acted in self-defense.'"), strongly expecting "s" or "That's why they acted" or "The actions were not premeditated."
>
> Later: News article format with headline and quotes structure, reporting on a CNN interview about NASA's chocolate cake, blending cultural and political commentary.
>
> The sentence ending "Johnson said she was proud of NASA's women soldiers and said 'They fought the battle and protected the mission.' " mirrors a quoted phrase about women's protective instinct, completing the explanation of her fear motivations.
>
> Final token ".' " closes a quoted phrase mid-sentence ("'My 'fighting for the mission' and 'her colleagues were doing their job and 'protecting the mission.'"), strongly expecting "s" or "s' " or "The women's actions were unnecessary" or "s, she said."

**armA label:**

> Formal article structure with numbered quotes and bold headers establishes an informative, witty tone discussing math and food.
>
> The phrase "The author says the book is 'a delicious dish'...so 'she wants to know the speed of the information, and 'her category'..." mirrors a common math expression format, completing a definition of the blog's scope with the second part showing the opposite.
>
> Final token " ' ' " closes the quoted phrase "her 'speed of information' and 'her category'...and 'eliminate the queue'...," strongly expecting "s" or "s' time" to close the second clause, completing the formal definition of the blog's categorization.

**armB label:**

> Strengthened framing of the closing quotation as completing a statement about protective duty. The later representation more sharply resolves the quoted phrase around teachers' custodial responsibilities ("take care of their students") as the concluded justification for their heroic actions.
>
> Sharpened expectation that the final punctuation closes a direct quote attributing protective motivation to an authority figure (Robinson). The earlier representation left this more open; the later representation more definitively marks this as the endpoint of reported testimony about duty-driven behavior.
>
> Weakened associations with crisis or emergency framing. The later representation de-emphasizes the shooting/massacre context present in the earlier reading, instead consolidating the semantic focus onto institutional responsibility and care as the stable interpretive frame for the quoted material.

*Stands out: armA's label is almost entirely off-topic (math and food) yet scores highest on the residual — its punctuation/quote-closing structure seems to be doing the work; armB is on-topic but near zero.*

---

## Typical / median positions

### Example 5 — tidx 772 (token ` who`, airline price-discrimination clause)

> …getting even $50 for a passenger to fill an empty seat at the last minute
> is profitable to them. The problem is somehow offering the $50 fare only to
> the passenger who would not fly otherwise, and not cannibalizing the
> customers>> who<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | 0.308 | 0.696 |
| armA | 0.280 | 0.675 |
| armB | 0.204 | 0.570 |

**arm0p label:**

> Earlier: Political economics blog post structure with expert quotes illustrating pricing theory, using concrete examples like airline pricing and Airbnb to explain demand elasticity.
>
> The sentence mid-clause ("For instance a price cap that steals sales from cheaper buyers could punish those who buy now while undercutting those who") is completing an example of price discrimination's fairness dilemma, listing consumer behavior conflicts around price tiers and demand overlap.
>
> Final token "who" ends an incomplete clause ("the customers who...and the customers who"), mid-listing exclusion examples ("not take away revenue from the customers who already bought"), expecting "already paid full price" or "would prefer to buy at full price" or similar contrasting customer behavior.
>
> Later: Business travel economics blog format with expert quotes and data-driven analysis, mixing airline pricing mechanics with quantitative examples.
>
> The quoted example mid-sentence ("The discounting model is frustrating for airlines who undercut fares to those who show up last minute, pricing out people who would have paid full price") is completing an economic rationale about price elasticity and consumer behavior comparison, detailing a no-show scenario.
>
> Final token "who" ends an incomplete clause ("people who...but the ones who"), mid-example describing rejected customers versus those willing to pay full price — expects "would have paid higher" or "already planned to buy at full fare" or "are willing to wait," continuing the price discrimination contrast.

**armA label:**

> Business travel analyst report format with detailed technical descriptions of airline pricing models, now quoting a NYT article with concrete examples of seat pricing behavior.
>
> The final sentence "Airline is forcing no-shows to pay full price, when those willing to wait get a discount. The empty seats represent passengers who would have paid full price" is mid-description of customer behavior economics, citing a fare discount scenario.
>
> Final token "who" ends an incomplete clause ("customers who...but many are those who" — a quoted example of potential buyers who refused early pricing, completing with "would have paid the full fare" or "are willing to pay the higher price if not booked."

**armB label:**

> Sharpened contrast between passenger willingness-to-pay segments: the later stage emphasizes distinguishing those who would accept last-minute discounts from those committed to full-price payment, intensifying the price discrimination dilemma at the core of airline revenue strategy.
>
> Strengthened quantitative framing of airline pricing mechanics: the later stage consolidates focus on concrete fare structures and passenger behavior comparisons—full price versus discounted seats—rather than abstract fairness concerns, moving toward data-driven economic analysis.
>
> Tightened expectation of customer preference revelation: the closing clause now more sharply anticipates a completion specifying what rejected customers would have chosen (full fare, higher price tier, or original booking intent) rather than simply listing excluded groups.

*Stands out: all three labels are on-topic and the scores sit almost exactly at each arm's mean — the typical ordering arm0p > armA > armB in miniature.*

### Example 6 — tidx 132 (token ` piano`, sheet-music catalog blurb)

> …Customers who bought this book also bought: Our Editors also recommend:
> The Art of the Fugue & A Musical Offering by Johann Sebastian Bach 19
> canons and fugues (complete with a>> piano<<

| arm | cos vs residual | cos vs raw diff |
|---|---|---|
| arm0p | 0.305 | 0.669 |
| armA | 0.276 | 0.635 |
| armB | 0.163 | 0.550 |

**arm0p label:**

> Earlier: Literary reference format with scholarly music description, presenting a classic German poem with musical settings and commentary.
>
> The phrase "A beautiful edition of Brahms's Songs of Solomon with piano" suggests an appended note or adaptation, likely describing the song's original vocal arrangement with piano accompaniment or a modern arrangement, implying a musical notation or translation context.
>
> Final token "piano" ends mid-phrase ("with a piano"), part of an incomplete noun phrase ("Includes the text (with a piano..."), strongly expecting continuation like "accompaniment" or "version" or "keyboard supplement" or "interpretive accompaniment," completing the descriptive clause about the song's musical adaptation.
>
> Later: Music reference book description with scholarly literary prose, presenting a classic song collection by American composer Samuel Coleridge-Taylor, with authoritative editorial notes.
>
> The phrase "Includes the vocal score of the song cycle (with a piano" suggests a supplementary edition feature, likely describing a piano accompaniment or full score version, implying a musical edition with practical notation details.
>
> Final token "piano" ends mid-phrase ("with a piano"), part of a parenthetical specification ("Includes the text—with a piano..."), strongly expecting continuation like "accompaniment" or "score version)" or "footnote)" or "accompaniment part," completing the description of the musical arrangement.

**armA label:**

> Literary reference book description with scholarly music notation format, presenting a distinguished classical song collection by American composer Samuel Barber.
>
> The sentence "Accompanying score includes a full piano-part study edition, with piano" suggests a final item in a list of published accompaniment details, implying a warning or additional notation for the orchestra's performance version, likely "performance score" or similar.
>
> Final token "piano" ends mid-phrase ("includes a piano...with piano"), part of a warning clause ("An annotated text version, with an 'x' that requires a piano"), strongly expecting "performance score" or "score" or "version, but" describing the textual accompaniment.

**armB label:**

> Sharpened focus on the musical score as a physical artifact—the closing clause now emphasizes a printed or notated edition with accompaniment details, rather than a generic vocal arrangement.
>
> Strengthened association with scholarly editorial apparatus: the later reading consolidates expectation that "piano" introduces specifics about how the musical text is presented and annotated for performers.
>
> Newly sharpened expectation that the incomplete phrase will resolve into naming a concrete musical format or publication feature (accompaniment, score version, or performance edition) rather than remaining abstract.

*Stands out: another near-median position; all arms converge on "piano accompaniment of a printed score" and the residual scores track the global means almost exactly.*
