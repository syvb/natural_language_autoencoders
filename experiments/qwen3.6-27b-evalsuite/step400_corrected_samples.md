# av_rl_lora_step400 — CORRECTED protocol samples

Protocol: chat template with `enable_thinking=False`, **prefill `<explanation>\n`**, T=1 sampling, **cut at `</explanation>`** (the RL harness's stop string; EOS is untrained past the close). Same 12 held-out rows/seeds as all prior sample sets. Mean extracted-content MSE **0.1144** (best arm: raw-unprefilled was 0.118, dash 0.121); 12/12 closed the tag; 0 CJK.

## row 0 — `openbmb/Ultra-FineWeb:en:89861`  (MSE 0.081, closed ✓)

**Gold:**

> "Also" signals continuation of a list or additional function
> DMX cable tester's capabilities still being enumerated
> XLR DMX lighting cable types being specified
> Product description genre: feature-listing sentence structure expected
> Parallel structure: "checks and [verifies/tests] X cables and also [Y]"
> "Pin XLR" implies specific connector types, more expected
> Retail/product catalog tone throughout, technical specs follow
> Multiple product listings pattern: another feature or cable type likely
> Incomplete verb phrase "checks and pin" suggests numbering like "5-pin"
> Brand/model names recurring, possible next product entry follows

**AV output (extracted):**

> Product catalog/description format: sequential listings of professional wire/audio testing tools with parliamentary compass style descriptions, each detailing features of DMX/LED/tester products.
> 
> Continuation pattern: Each product entry introduces a tester's functions; "DMX cable tester with LED indicator. This little tester will also test DMX 6 pin XLR cable faults and also" promises additional capabilities yet to be listed (e.g., short/open faults, polarity).
> 
> Final token "also" is mid-sentence, part of a compound "DMX tester by Gallagher Pro. This tester tests both 5 & 3 Pin DMX cables and also" requiring continuation of additional compatible functions or test modes. The connector "also" introduces an additive feature clause.
> 
> "locomotive pilot cable" context: functional testing language. "Six-wire...also" demands another testing capability clause.

## row 1 — `openbmb/Ultra-FineWeb:en:83422`  (MSE 0.067, closed ✓)

**Gold:**

> "several" signals an incomplete noun phrase requiring continuation
> List or quantity of dating service features/options expected next
> Promotional marketing copy tone continues, upbeat and persuasive
> Geographic/demographic specificity pattern: Massachusetts singles focus maintained
> Match.com advertisement genre conventions: features, tools, or membership options
> Plural noun expected immediately after "several" (e.g., "tools," "ways," "options")
> Call-to-action marketing momentum building toward user engagement
> Sentence likely concludes with benefit or feature description
> Formal yet accessible register consistent throughout promotional text
> Pattern of superlatives/positives ("thousands," "today") likely continues

**AV output (extracted):**

> Dating site promotional/register context: Match.com or similar platform copy, advertising free Chicago IL singles and MIL dating services.
> 
> Marketing continuation pattern: sentence structure "Chicago Singles Search is easy to find your ideal match. We have thousands of single women & men in Illinois. We have several" promises an enumeration of options, features, or location-specific subcategories following the pattern of building user interest in the platform.
> 
> Final token "several" opens a plural noun phrase within promotional copy; "Chicago is the place to find love. Chicago has thousands of single women and men. We have several" sets up continuation listing location profiles, events, packages, or community subsets. "We have several" strongly anticipates a plural noun like "locations," "profiles," or "ways to meet."
> 
> "We have several" — incomplete noun phrase requiring an immediate […]

## row 2 — `openbmb/Ultra-FineWeb:en:86506`  (MSE 0.107, closed ✓)

**Gold:**

> Final token "rate" begins a compound noun phrase requiring completion
> "pace of rate" strongly expects "increases," "growth," or "appreciation"
> Yun quote is unclosed, requiring closing quotation mark soon
> Formal NAR press release genre with economist quote conventions
> Pattern of hedged future predictions ("anticipate," "should") continues
> Momentum from "markedly lower the pace" needs object complement
> Statistical/housing market domain; price trajectory language expected
> Paragraph structure suggests quote conclusion then possible new section
> Contrast pattern established: strong growth now moderating/slowing
> Nationwide Association of Realtors report tone: measured, professional

**AV output (extracted):**

> Financial news reporting pattern: article systematically covering RegionalMonster.com reports with mortgage rate data and housing statistics citations.
> 
> Narrative momentum: article has transitioned to CoreLogic's year-to-date report, focusing on Q2 2023 home price growth data; statistical finding now expected to complete.
> 
> Specific content constraint: "Fifty-eight percent of MSA markets posted double" strongly signals "double-digit gains/percentage" completion, a standard statistic phrasing in real estate newsletters ("experienced double-").
> 
> Final token "double" is mid-phrase within a SDK bullet point explaining growth patterns, requiring an adjective or noun continuation ("double-digit" gains/percentage increase logic).
> 
> "Over the 12-month period, 58% of metro areas experienced double" — syntax demands immediate continuation with digit/hyphen completing the statistic.

## row 3 — `openbmb/Ultra-FineWeb:en:32382`  (MSE 0.113, closed ✓)

**Gold:**

> "Contrary" begins a contrastive clause requiring completing argument
> Previous sentences described Amano behavior/traits; contrast now expected
> Factual/informational tone about shrimp species continues throughout
> Pattern of describing physical or behavioral traits being continued
> The word "contrary" typically precedes "to" in English constructions
> Common contrast: peaceful nature vs. predatory opportunism already mentioned
> Hobbyist aquarium blog register: casual yet informative tone maintained
> Specific species facts pattern: size, temperament, diet already covered
> First-person narrator sharing observational experience about tank inhabitants
> Latin binomial header suggests semi-formal species profile structure

**AV output (extracted):**

> Informal aquarium/hobbyist writing in German/English mix, with personal anecdote style about discovering wild neocaridina shrimp in a wild fish tank.
> 
> The text follows a two-part structure: introduction of animal and its origin story, then biological characteristics — size noted (3.5cm), trophic behavior mentioned ("feed on algae," "very materialistic cleaners"), next logical topic likely contrasts with Neon tetras or other popular shrimp/carcinus attitudes.
> 
> "Contrary" opens a comparative or corrective contrast clause, likely "Contrary to [belief/to some species]," expecting a dependent clause describing a contrast — specifically about dietary needs, availability, or hardiness of dwarf shrimp, building on the prior "feed on everything."
> 
> Final token "Contrary" begins a contrastive construction, requiring "to [X]" or "to popular belief" next.

## row 4 — `openbmb/Ultra-FineWeb:en:97577`  (MSE 0.181, closed ✓)

**Gold:**

> "The" opens a noun phrase describing the Tibet Hotel
> Hotel's notable features or history likely described next
> Journalistic AP news style, factual descriptive prose expected
> "Closed loop" system contrast with hotel's Tibetan theme continues
> Irony/juxtaposition narrative: experiencing Tibet without visiting Tibet
> Specific details about hotel décor, atmosphere, or Tibetan elements expected
> Third-person objective reporting tone maintained throughout
> Article introducing a human-interest angle within Olympics coverage
> Singular definite noun phrase requires specific hotel attribute noun
> Background context about Tibet's political sensitivity likely woven in

**AV output (extracted):**

> News article format with anecdotal structure — AP/wire style journalist sharing an unexpected observation about U.S., TikTok, and Ukraine geopolitical context. The headline's irony ("US Capitol complex has a Tibetan restaurant") sets up a forthcoming explanation.
> 
> The narrative promises to elaborate on visitors experiencing Tibetan cuisine/culture at the State Department's Moscow hotel, which is low-profile but notable to journalists covering Ukraine. "A few journalists have done just that. The" signals an upcoming detail or description qualifies/contradicts the headline's teasing tone.
> 
> Final token "The" opens a definitive statement about the hotel or the experience, immediately requiring a noun phrase. It likely introduces an ironic or explanatory elaboration about the building or article, completing the sentence introduced mid-thought ("U.S. State Department's... This week, elected […]

## row 5 — `openbmb/Ultra-FineWeb:en:94024`  (MSE 0.165, closed ✓)

**Gold:**

> "been" signals past perfect completion, next word imminent
> The Tonight Show's ending time is the expected next concept
> Logical conclusion: show would have ended before midnight
> Specific time phrase expected, likely "over" or "finished" by midnight
> Continuity error argument being built toward its conclusion
> Factual/analytical tone identifying a plot inconsistency in a film
> Texas/Central Time Zone detail constrains the timing arithmetic
> The midnight timestamp Sue Ann notes contradicts Carson still airing
> Film criticism register, methodically presenting evidence of anachronism
> Bobby Lee and Sue Ann scene context no longer directly relevant

**AV output (extracted):**

> Temporal/corrective context: author clarifying a factual error about a TV show John Singer saw live on Dec. 25, 1969 at "9:50 PM."
> 
> Logical reasoning completion: the parenthetical explains the timing constraint — "if it had begun at 8:00, it would have been" logically requires a conclusion about the show ending, e.g. "over by 20 minutes before" or "finished by 9PM."
> 
> Final token "been" is part of a counterfactual conditional construction ("since it began at 8:00 PM would in reality have been"), forcing a past-perfect predicate completing the consensus about Typical Tonight's standard 30-minute runtime ending.

## row 6 — `openbmb/Ultra-FineWeb:en:13710`  (MSE 0.116, closed ✓)

**Gold:**

> Mid-word or phrase cut: "signal" likely starts a compound noun
> Introducing second model's features, parallel structure to first model expected
> List/contrast pattern: A8GS had one feature, A8GS-T has additional features
> Technical specification language continues: voltages, functions, measurements expected
> The word "signal" likely precedes "contact" or "output" in electronics context
> Both models compared, A8GS-T described as enhanced/superior variant
> Formal product-description register: precise technical noun phrases throughout
> Secondary feature being introduced after "and", completing the conjunction
> Electronics/relay switch domain: terminal, contact, output vocabulary likely
> Parenthetical or clause may follow completing the signal-related specification

**AV output (extracted):**

> Technical product description for a Panasonic switch, establishing bilingual component specs for electronics/automation readers. The text contrasts two features: resistor switching and electromagnetic type switches.
> 
> The sentence structure "Additionally, one of the panel miniature pushbutton switches..." continues listing innovations; the word "Additionally" signals another feature of electromagnetic switches being described.
> 
> The text is mid-sentence explaining Panasonic's dual-switch configuration: "combines operation control electrodes and a separate 'signal" — the single-quoted incomplete term strongly predicts a closing quoted compound noun like "signal output" or "contact/switch" describing feedback functionality.
> 
> Final token "signal" is part of an incomplete quoted compound noun phrase in a structured description — requiring a closing quote and noun, likely "pin" or "output" to […]

## row 7 — `openbmb/Ultra-FineWeb:en:58648`  (MSE 0.089, closed ✓)

**Gold:**

> Numbered list item "6." expects a title/heading next
> Pattern of numbered tips (1-4, now 6) continues, skipping 5
> Each tip section begins with a bold or short imperative title phrase
> Previous tip headings: "Find", "Take", "Inspect", "Agreeing" — next likely imperative or gerund
> Consumer advice article genre: practical, directive tone throughout
> Topic likely covers financing, documentation, or vehicle history checks
> Tips alternate between dealer interaction and buyer due diligence themes
> Section 4 ended with written confirmation advice; next tip likely shifts topic
> Formal-but-accessible register, second-person address expected to continue
> Numbered list skipped 5, suggesting possible editorial error or continuation

**AV output (extracted):**

> The article is a comprehensive consumer guide to buying a used or second-hand car, structured with UK Civil Justice Helpline bullet-point tips throughout; four numbered/titled sub-sections preceding this section addressed "Where and what to buy," "Financing," "Negotiating," and "Viewing and test driving."
> 
> The checklist pattern signals another post-purchase or transaction-step tip is expected next, likely covering paperwork, insurance, service history, or finalising the deal.
> 
> "6." is the final token — a numbered list item mid-sequence in the "useful tips when buying your used or part exchange car" section, immediately requiring a short tip title (e.g., "Test drive and paperwork," "Taking ownership").

## row 8 — `openbmb/Ultra-FineWeb:en:70163`  (MSE 0.186, closed ✓)

**Gold:**

> "Sure" signals near-certain continuation with noun/adjective phrase
> Gift/present context expects items like "to be found" or similar
> Listicle/deals article format expects promotional concluding sentence
> Cyber Monday/Black Friday shopping framing continues deal recommendations
> German news aggregator format with multiple unrelated story snippets
> Informal, reader-addressed register ("for you", "we have compiled")
> Christmas shopping theme suggests seasonal gift-related continuation
> Sentence grammatically incomplete, requiring predicate complement
> "One or the other" idiomatic phrase typically resolves with discovery/find
> Article closes with lighthearted consumer-friendly promotional tone

**AV output (extracted):**

> Aggregated news header fashion: multiple short unrelated German health/lifestyle snippets blur together, suggesting a scraped portal page.
> 
> The article is concluding with an advertising pitch for a Christmas catalog/special offer, listing broader Christmas news categories and quoting HSE provider news. The closing teaser ("Browse through the Catalogue 2.0 and let your heart beat for the variety of products. The right souvenir is sure") signals an enthusiastic closing sentence.
> 
> Final token "sure" ends an incomplete clause ("We have collected numerous products especially for you. So a hit is sure"), requiring completion — "to be a hit" or similar affirmation. "Sure" strongly anticipates "to be found/included" or idiomatic: "surely" continuing the compound "so many diverse products for everyone, a hit is sure."

## row 9 — `openbmb/Ultra-FineWeb:en:94688`  (MSE 0.078, closed ✓)

**Gold:**

> "3D" is incomplete, expects a noun (environment/world/game)
> App store listing genre, feature description continues
> Exciting gameplay description mid-sentence, noun phrase required
> List of features being enumerated one by one
> Child-friendly educational game context, age-appropriate language
> Previous examples show numeric difficulty progression pattern
> Sentence structure "exciting 3D [noun]" strongly constrains next token
> Marketing/promotional register with enthusiastic descriptive language
> Game mechanics being explained, visuals/setting likely described next
> Earlier mention of "3D" implies immersive environment terminology follows

**AV output (extracted):**

> Software description format: educational app listing following conventions of age/gender, price, and feature highlights for a children's math learning game from 2007 UK developer.
> 
> Gameplay mechanic description building momentum: "Christian collects balls as he solves fun rapid fire maths problems. Player personally helps Christian by tapping the correct solution in an exciting 3D" — narrative is mid-sentence, describing the game's visual/environmental setting; "3D" strongly predicts "world" or "rendered environment."
> 
> Final token "3D" is part of an incomplete sentence fragment "Players have fun learning while playing in an exciting 3D," requiring an immediate noun referencing the game's graphical environment (e.g., "world," "engine," "animated surroundings"), consistent with casual animal-fetcher game marketing language from that period.

## row 10 — `openbmb/Ultra-FineWeb:en:42485`  (MSE 0.073, closed ✓)

**Gold:**

> "The" opens a new sentence, likely a noun phrase follows
> Second sentence continues elaborating on the announcement
> Formal government/policy document register throughout
> Acronym-heavy domain: HUD, NOFO, CoC, DV established
> New sentence likely introduces deadline, eligibility, or purpose
> Passive/institutional tone expects bureaucratic noun phrases next
> Large funding figures suggest financial details may continue
> Homelessness/housing policy domain constrains vocabulary
> Prior sentence was long; new sentence likely shorter, specific
> Document likely describes application process or eligible applicants

**AV output (extracted):**

> Government/organizational document describing HUD funding opportunities, establishing factual announcement tone about CHAR 2023 NOFO release for community housing organizations.
> 
> Narrative momentum: the text has introduced the NOFO for Continuum of Care funds and noted its key features (returning to one-year implementation cycle), signaling more specifics about eligibility, deadlines, or provisions are expected next.
> 
> "The" begins a new sentence continuing elaboration about the NOFO's details — likely referencing a specific component, deadline, or requirement, e.g. "The NOFO was released on..."
> 
> Final token "The" opens a noun phrase mid-announcement, most immediately constrained to a capitalized noun referencing the NOFO, HUD, or a deadline/notice detail. "The November 2022 NOFO" formula common in such posts suggests specifics about the notice's provisions follows.

## row 11 — `openbmb/Ultra-FineWeb:en:86506`  (MSE 0.116, closed ✓)

**Gold:**

> Final token "rate" begins a compound noun phrase requiring completion
> "pace of rate" strongly expects "increases," "growth," or "appreciation"
> Yun quote is unclosed, requiring closing quotation mark soon
> Formal NAR press release genre with economist quote conventions
> Pattern of hedged future predictions ("anticipate," "should") continues
> Momentum from "markedly lower the pace" needs object complement
> Statistical/housing market domain; price trajectory language expected
> Paragraph structure suggests quote conclusion then possible new section
> Contrast pattern established: strong growth now moderating/slowing
> Nationwide Association of Realtors report tone: measured, professional

**AV output (extracted):**

> Ongoing data-driven real estate report format, mixing NAR statistics, charts, and quotes about 2024 U.S. housing market outlook.
> 
> Established pattern of describing price momentum, affordability issues, and regional trends; the passage is building toward explaining consequences of constrained supply driving unsustainable price growth.
> 
> Ronald Chan's quote is mid-sentence, explaining the relationship between mortgage prices, buyer demand, and "it mid considerably dampen the speed of rate..." — the sentence promises a concluding phrase about price gains/acceleration completing the thought. The endpoint formula ("coupled with limited new building... have in flip placed a cap on the tempo of rate") requires a noun completing "temper the rate," such as "bills" or "gains."
