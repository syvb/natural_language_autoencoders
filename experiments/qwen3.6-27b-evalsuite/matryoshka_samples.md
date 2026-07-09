# MATRYOSHKA model samples — ceselder/nla-qwen36-27b-matryoshka, rl_av_lora_iter400

**Protocol** (correct for THIS checkpoint): raw text base + `rl_av_lora_iter400`, chat template with `enable_thinking=False`, **no prefill, no tags** (verified: 0/48 generations show any tag traces), T=1 sampling. The model almost never emits EOS (truncation-reward training never incentivized stopping) — cut at your token budget; the trained reward horizon was 120 content tokens, so everything past ~120 is unoptimized tail.

12 held-out rows, 256-token budget, own-critic reconstruction MSE (mean **0.1359**). Note the matryoshka signature in the openers: the first line IS the predicted next-token completion, then lines broaden outward.

## row 0 — `openbmb/Ultra-FineWeb:en:89861`  (MSE 0.118)

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

**matryoshka AV output:**

> also tests wiring functionality additional modes described tester tool description "tests XLR cables" and also checks polarity or other parameters
> List of audio test tools continues, vintage UK catalogue/product snippet style
> Third item description fragment "tests standard RS485 signals between two points and also" implies continuation: polarity checking, LED detection
> www.electronic-component supplier tool specifications typically multi-function: warn/format/diagnose multiple parameters
> Tester checks DMX cables broken cable detection and also battery/power continuity check
> "Hand Held DMX Cable tester tests standard DMX512 cables between two points and also" mirrors probe/second function clause
> Specific product: "small electronic tester detects broken DMX cables between two DMX addresses this tester tests XLR wiring faults" — also tests address or short/open circuit
> 1990s/early 2000s lighting technical reference database, multiple fragmented excerpts about DMX testers
> "tests standard […]

## row 1 — `openbmb/Ultra-FineWeb:en:83422`  (MSE 0.100)

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

**matryoshka AV output:**

> several dating site categories/options webpage marketing
> "We have several" signals city subsections or membership tiers follow
> Dating site genre promotional text "San Diego Singles hook up site" introduction formula
> "San Diego is a large area. We have several" implies website offers local matching options by neighborhood
> Promotional claim "We are the best free dat site... ...!" typical match.com copy continues "We have several" profiles or events sections
> Classic dating site intro structure: "San Diego is big city. We have several [free ways/groups/types] to help you find love nearby."
> Registration prompt context suggests "city profiles," "age groups," or filter options enumeration
> "California area" referenced twice, suggesting geographic segmentation emphasis follows
> "Match.com has many singles in San Diego." We have several — likely a street/membership variant or search tips approach
> Early 2000s adult dating copy tone, simple formulaic continuation "We have several" online […]

## row 2 — `openbmb/Ultra-FineWeb:en:86506`  (MSE 0.132)

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

**matryoshka AV output:**

> double-digit percentage growth implication "double" begins number
> US Labor statistics report genre signals specific numeric threshold
> "every state experienced double" requires single/double-digit adjective completion
> Inflation report context "2020 Q1.. so far this year" demands percentage growth magnitude
> Standard economics press release format: "record-breaking… this year meant… double" mirrors S&P Global phrasing
> S&P Global state-level wage data press release typically continues "some states achieved double-digit" growth
> "that quarter saw single-digit percentage growth" ruled out; structure "grow led to double" expects digits ≥10%
> Parallel phrasing: "each state recorded double" almost always followed by "digit growth" in US wage reports
> Formal financial press release opening, precise metrics framing continues "areas saw double-digit percentage hurdle"
> Historical escalation pattern: 2021 wage gains consistently described as multi-digit increase breakdowns begin with "double"
> 
> Final […]

## row 3 — `openbmb/Ultra-FineWeb:en:32382`  (MSE 0.112)

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

**matryoshka AV output:**

> contrary misconception aquarium fish differs claim begins
> "Contrary" introduces comparison to another species discussed
> German informal scientific enthusiast register continues about feeding habits
> "Contrary" signals correction: e.g., "Contrary to popular belief" or requirements
> Humble lizard mentioned size/diet caveat upcoming
> Mantis Shrimp species introduction wrapped, "Contrary" signals this lizard doesn't share a commonly assumed trait mentioned prior
> Contrast with previous species implies breeding or availability difference
> Author's amusing first-person field notes tone continues
> "Contrary" likely precedes "to [other shrimps / common assumption]" about freshwater requirements
> Keep it short: "Contrary" opens sentence about shrimp generally needing saltwater but this one doesn't or requires different approach
> German blog hobbyist context suggests informal factual continuation about the Blue Ring Shrimp's dietary vs behavioral traits
> 
> "Contrary" likely contrasts with preceding claim […]

## row 4 — `openbmb/Ultra-FineWeb:en:97577`  (MSE 0.190)

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

**matryoshka AV output:**

> The article begins explaining US restaurant visit to China's Moscow embassy district
> Journalistic feature setup "The" introduces surprising discovery/description
> Lighthearted NBA beat writer tone describing unusual geographic/cultural irony
> "UPI reporters happen to be visiting Beijing this week ... eating Chinese food at Chinese-themed hotels in Washington DC. The" continues with factual exploration of the hotels' existence
> Description of how the hotel section was accessed or what reporter found
> "Five Chinese cities are named as pandemic hotspots this week. The" sentence structure expects continuation of historical/cultural information, not a punchline
> Article excerpt introduces premise, "We went inside ☺" suggests lighthearted investigative follow
> Context shift: embedded reporter experience describes what the restaurant/room actually represents politically or culturally
> "Washington DC has a hotel with a China street-name sign. we visited it during Asia trip. The" signals section […]

## row 5 — `openbmb/Ultra-FineWeb:en:94024`  (MSE 0.139)

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

**matryoshka AV output:**

> been aired, ending time logically should be concludedshow
> TV scheduling math error explanation mid-argument "would have been" finished or over by certain time
> Late night show originally broadcast 7pm-8pm so by 10pm it was already over
> Parenthetical clarification "starting at 7pm EST; it's a one-hour show and would have been" ends with concluded/off air
> Conversational explain vs correction tone about old TV airing times
> "if it aired at 7pm would have been" implies hour later ending time "5:30pm already ended"
> Humorous嘲讽 register "Dumb question" suggests mild point about impossible time
> US network TV specific context: old Saturday Night Live tape times
> "started at 7pm EST" and "was a one hour show" implies it would have ended 8pm
> " aired at 7pm EST and by 10pm would have been" completes "off air/finished/long over"
> 
> "was broadcast at 7pm and would have been" strongly expects ending time
> Math logic: 7pm start + 1 hour = 8pm end, by 10pm absurd
> Forum post correcting user's […]

## row 6 — `openbmb/Ultra-FineWeb:en:13710`  (MSE 0.122)

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

**matryoshka AV output:**

> signal output relay component electronics "also 'signal" incomplete word sensor feedback contact term
> Japanese sensor switching module specification "In addition to activation force type it has a 'signal function detecting valve state"
> Self-contained miniature pushbutton relay "also has a second output pin labeled 'signal" electrical confirmation output
> "push-type switch also has a corresponding 'signal output/contact cutoff mechanism" typical for Omron industrial components
> Technical spec description register
> Component description continues valve cylinder or position detection signal circuit “a ‘signal...” electrical output term e.g. detection pulse
> "open/close button” plus “also a 'signal output" pin implies switch state readback/output feedback electronics function
> Translated Japanese electronics/appliance manual context, likely IoT, pneumatic switch module
> The quoted ‘“signal” word suggests a second terminal like output indicator or limit switch logic signal
> Relay actuator […]

## row 7 — `openbmb/Ultra-FineWeb:en:58648`  (MSE 0.110)

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

**matryoshka AV output:**

> ." buying steps checklist continues paperwork/checking the vehicle contract or final advice
> British consumer advice article nearing finish of used car buying steps
> "6." section begins final inspection or payment/documentation trick
> Pattern is: inspect car, check history, test drive, negotiate, then paperwork section
> Previous sections advise checking MOT, price, dealer credentials, test drive details
> Logical next step: what buyer should obtain after agreeing purchase — insurance, contract, warranty documentation
> "." numeral signals "7." or similar missing transitional step like "Check finance/ownership documents"
> Article tone is practical, second-person, consumer-protection forum like Which?
> Earlier advice mentions paperwork resumes: earlier "You should..." verb phrase continues
> Specific UK-used-car context: V5 registration document, warranty, condition checklist expected next
> Bullet sequence momentum "6." suggests signing/legal/financing section follows test drive completion advice

## row 8 — `openbmb/Ultra-FineWeb:en:70163`  (MSE 0.161)

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

**matryoshka AV output:**

> sure German sentence "something is sure" shopping bargain conclusion imminent
> "Advent calendar" article ending "Among these offers, a suitable gift is sure" requires completion "is sure to" find match
> Multiple German news snippets about automotive/corporate news, ends with gift tips
> "10 super fun games for kids and adults are on offer, and a Christmas hit is sure" implies probability: "and趣味性 or favorite pick will be found"
> Idiomatic filler conclusion of German Door-to-door magazine promotion text
> "Among the wide range of products available, a Christmas hit" → "is sure" + specific promise "to fit" someone
> "Sure" German idiom pattern "da ist sicher" something awaited directly before "among them"
> Final clause "with all categories covered, a suitable [item]" completing humorous light editorial tone
> "Browse this list of tips and games — something definitely is certain that among the holidays..."
> Translation artifact "sure" signals "sure" → "at least one wish" satisfies German "ist sicher […]

## row 9 — `openbmb/Ultra-FineWeb:en:94688`  (MSE 0.125)

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

**matryoshka AV output:**

> 3D game environment casual educational description "playfully in a 3D" scene
> Old Mac game cabinet - vintage interactive software genre
> Mouse-catching game mechanics with animated 3D rendered digits/math environments
> "fun interactive educational activity in a fun 3D" likely world/scene visuals
> 2002-era children's app marketing description continues
> "chase mice" described animated cartoon environment in first sentence mentions
> "fun 3D" typically precedes visuals showing environment description
> Voice recognition game: "interactive" suggests immersion in 3D
> Flash/MathMagic app description style, Apple store or demo page
> "noise cancellation" context ends; gameplay opening "the fun takes place in a cool 3D" suggests characters/environments rendered surroundings
> Single-demo educational animal/mouse theme suggests cartoon arena
> "jude" likely "you chase" mice in 3D rendered world is classic spoken math interface games era
> 
> 3D renders were notable selling point circa 2002 mobile/virtual […]

## row 10 — `openbmb/Ultra-FineWeb:en:42485`  (MSE 0.092)

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

**matryoshka AV output:**

> The announcement deadline details document funding cycle announcement specifics
> Federal grant announcement context: eligibility window and adjustments just stated
> "The" opens new announcement department guidance noun phrase about issuance
> 2022 USDA Rural Development announcement cycle version details official specifications
> Announcement content structure: first sentence stated total funding, now webinar/application deadline specifics follow
> RFP deadline and modified announcements are subject — "The" begins "The new Funding Request Form..." or similar guidelines document release
> Specific publication date or guidance document reference expected "The [application window opens/releases/requirements are]..."
> COVID ERA funding cycle specifics include priorities, scoring criteria, submission dates
> Pro grant announcement summary genre: formal press release framing "USDA has released..."
> Deadline or eligibility criteria section begins next: standard RFP opening paragraph structure […]

## row 11 — `openbmb/Ultra-FineWeb:en:86506`  (MSE 0.229)

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

**matryoshka AV output:**

> rate housing price growth slowing reduces expected rate inflation continuation
> Real estate economics sentence completing "curbing of rate" mirrors earlier claim about rapid price increases
> NAR report context requires continuation describing home price increases metric
> " constrains future rate of value increases" structure parallel "rate" needs noun like "home prices" or "price escalation"
> US housing market report genre: direct quote format or paraphrase nearing conclusion
> "Supply constraints have been key cause of lately rapid escalation, so sluggish sales quantity would be beneficial for curbing of rate" - this repeats "price formula/increase" concept completing the phrase "rate of rate..." implies earlier "speed/acceleration" repetition typo correction needed: "rate growth" or "price growth"
> Fixed quantitative claim "limited slowdown in sales would help restrict pace of rate" expects specific buyer affordability/read-only metrics concluding
> NAR administrator quoted earlier about […]
