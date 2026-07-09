# Matryoshka NLA samples with FVE — rl_av_lora_iter400 (reference: av_rl_lora_step400)

**Sampling** (each model under its verified-correct protocol, same rows and seeds, T=1, `top_p=1.0`, top-k off, `enable_thinking=False` chat template): matryoshka = no prefill, no tags, output cut at the 256-token budget; step400 reference = `<explanation>\n` prefill, cut at `</explanation>`. Per-sample FVE = 1 − dirMSE/var̄ (both sides L2-normalized to mse_scale; var̄ from these 12 golds = mean-predictor baseline). **Each model is scored by its own co-trained critic** — cross-model FVE comparisons carry that caveat.

Aggregate FVE on these 12 rows: **matryoshka 0.654**, step400 reference 0.709. 'Source tail' = the end of the document prefix whose final-token L42 activation was injected (the state the AV describes).

## row 0 — `openbmb/Ultra-FineWeb:en:89861` @ 172 tokens

**Source tail:**

> …nd able to diagnose . Speakon, Pole Speakon, Banana, 3. VintageModified: Pyle Pro Audio Cable Tester , ESSENTIAL GIG GEAR! SoundLab – Audio Cable Tester. Here are a few videos on. Professional Cable Tester.
> DMX 35T – DMX Cable Tester.
> This cable tester checks and pin XLR DMX lighting cables and also

**matryoshka (FVE +0.699):**

> also tests wiring functionality additional modes described tester tool description "tests XLR cables" and also checks polarity or other parameters
> List of audio test tools continues, vintage UK catalogue/product snippet style
> Third item description fragment "tests standard RS485 signals between two points and also" implies continuation: polarity checking, LED detection
> www.electronic-component supplier tool specifications typically multi-function: warn/format/diagnose multiple parameters
> Tester checks DMX cables broken cable detection and also battery/power continuity check
> "Hand Held DMX Cable tester tests standard DMX512 cables between two points and also" mirrors probe/second function clause
> Specific product: "small electronic tester detects broken DMX cables between two DMX addresses this tester tests XLR wiring faults" — also tests address or short/open circuit
> 1990s/early 2000s lighting technical reference database, multiple fragmented excerpts about DMX testers
> "tests standard […]

**step400 reference (FVE +0.795):**

> Product catalog/description format: sequential listings of professional wire/audio testing tools with parliamentary compass style descriptions, each detailing features of DMX/LED/tester products.
> 
> Continuation pattern: Each product entry introduces a tester's functions; "DMX cable tester with LED indicator. This little tester will also test DMX 6 pin XLR cable faults and also" promises additional capabilities yet to be listed (e.g., short/open faults, polarity).
> 
> Final token "also" is mid-sentence, part of a compound "DMX tester by Gallagher Pro. This tester tests both 5 & 3 Pin DMX cables and also" requiring continuation of additional compatible functions or test modes. The connector "also" introduces an additive feature clause.
> 
> "locomotive pilot cable" context: functional testing language. "Six-wire...also" demands another testing capability clause.

## row 1 — `openbmb/Ultra-FineWeb:en:83422` @ 77 tokens

**Source tail:**

> …tinue, I agree to receive transactional and promotional emails from Match.com. I understand that I am free to withdraw consent at any time.
> 35 - Dorchester Center, MA
> Active within 3 days
> The Bay State of Massachusetts has thousands of singles looking for love. Find your match today! We have several

**matryoshka (FVE +0.745):**

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

**step400 reference (FVE +0.829):**

> Dating site promotional/register context: Match.com or similar platform copy, advertising free Chicago IL singles and MIL dating services.
> 
> Marketing continuation pattern: sentence structure "Chicago Singles Search is easy to find your ideal match. We have thousands of single women & men in Illinois. We have several" promises an enumeration of options, features, or location-specific subcategories following the pattern of building user interest in the platform.
> 
> Final token "several" opens a plural noun phrase within promotional copy; "Chicago is the place to find love. Chicago has thousands of single women and men. We have several" sets up continuation listing location profiles, events, packages, or community subsets. "We have several" strongly anticipates a plural noun like "locations," "profiles," or "ways to meet."
> 
> "We have several" — incomplete noun phrase requiring an immediate plural noun, likely reinforcing site quality or community breadth.

## row 2 — `openbmb/Ultra-FineWeb:en:86506` @ 152 tokens

**Source tail:**

> … and confined housing offer led to median sales prices rising for existing single-spouse and children houses in all but just one of 183 markets measured by the Countrywide Association of Realtors® (NAR).
> NAR’s most recent quarterly report found that four out of 5 (78%) of 183 markets observed double

**matryoshka (FVE +0.664):**

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

**step400 reference (FVE +0.728):**

> Financial news reporting pattern: article systematically covering RegionalMonster.com reports with mortgage rate data and housing statistics citations.
> 
> Narrative momentum: article has transitioned to CoreLogic's year-to-date report, focusing on Q2 2023 home price growth data; statistical finding now expected to complete.
> 
> Specific content constraint: "Fifty-eight percent of MSA markets posted double" strongly signals "double-digit gains/percentage" completion, a standard statistic phrasing in real estate newsletters ("experienced double-").
> 
> Final token "double" is mid-phrase within a SDK bullet point explaining growth patterns, requiring an adjective or noun continuation ("double-digit" gains/percentage increase logic).
> 
> "Over the 12-month period, 58% of metro areas experienced double" — syntax demands immediate continuation with digit/hyphen completing the statistic.

## row 3 — `openbmb/Ultra-FineWeb:en:32382` @ 118 tokens

**Source tail:**

> …sh and snails it also contains about a dozen Amanos, and countless Crystal Reds.
> Amanos are quite impressive. Their bodies, without the antennae, are about five centimeters long, and even though they're generally peaceful they don't decline a snack of newborn fish if the opportunity arises.
> Contrary

**matryoshka (FVE +0.716):**

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

**step400 reference (FVE +0.713):**

> Informal aquarium/hobbyist writing in German/English mix, with personal anecdote style about discovering wild neocaridina shrimp in a wild fish tank.
> 
> The text follows a two-part structure: introduction of animal and its origin story, then biological characteristics — size noted (3.5cm), trophic behavior mentioned ("feed on algae," "very materialistic cleaners"), next logical topic likely contrasts with Neon tetras or other popular shrimp/carcinus attitudes.
> 
> "Contrary" opens a comparative or corrective contrast clause, likely "Contrary to [belief/to some species]," expecting a dependent clause describing a contrast — specifically about dietary needs, availability, or hardiness of dwarf shrimp, building on the prior "feed on everything."
> 
> Final token "Contrary" begins a contrastive construction, requiring "to [X]" or "to popular belief" next.

## row 4 — `openbmb/Ultra-FineWeb:en:97577` @ 70 tokens

**Source tail:**

> …AP) — People attending the Beijing Winter Olympics can’t visit Tibet because they’re in China’s “closed loop” system for foreign visitors. But some visitors, including part of the Associated Press’ Olympics team, are getting a taste of the region because they’ve been assigned to the Tibet Hotel. The

**matryoshka (FVE +0.516):**

> The article begins explaining US restaurant visit to China's Moscow embassy district
> Journalistic feature setup "The" introduces surprising discovery/description
> Lighthearted NBA beat writer tone describing unusual geographic/cultural irony
> "UPI reporters happen to be visiting Beijing this week ... eating Chinese food at Chinese-themed hotels in Washington DC. The" continues with factual exploration of the hotels' existence
> Description of how the hotel section was accessed or what reporter found
> "Five Chinese cities are named as pandemic hotspots this week. The" sentence structure expects continuation of historical/cultural information, not a punchline
> Article excerpt introduces premise, "We went inside ☺" suggests lighthearted investigative follow
> Context shift: embedded reporter experience describes what the restaurant/room actually represents politically or culturally
> "Washington DC has a hotel with a China street-name sign. we visited it during Asia trip. The" signals section […]

**step400 reference (FVE +0.539):**

> News article format with anecdotal structure — AP/wire style journalist sharing an unexpected observation about U.S., TikTok, and Ukraine geopolitical context. The headline's irony ("US Capitol complex has a Tibetan restaurant") sets up a forthcoming explanation.
> 
> The narrative promises to elaborate on visitors experiencing Tibetan cuisine/culture at the State Department's Moscow hotel, which is low-profile but notable to journalists covering Ukraine. "A few journalists have done just that. The" signals an upcoming detail or description qualifies/contradicts the headline's teasing tone.
> 
> Final token "The" opens a definitive statement about the hotel or the experience, immediately requiring a noun phrase. It likely introduces an ironic or explanatory elaboration about the building or article, completing the sentence introduced mid-thought ("U.S. State Department's... This week, elected officials). The").

## row 5 — `openbmb/Ultra-FineWeb:en:94024` @ 91 tokens

**Source tail:**

> …ow is on with Johnny Carson doing Carnac. When they finish making love, Sue Ann points out that it is past midnight. Carson is still on, which would be fine on the East or West Coast, but the movie is set in Texas, in the Central Time Zone, where the Tonight Show came on at 10:30 and would have been

**matryoshka (FVE +0.646):**

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

**step400 reference (FVE +0.580):**

> Temporal/corrective context: author clarifying a factual error about a TV show John Singer saw live on Dec. 25, 1969 at "9:50 PM."
> 
> Logical reasoning completion: the parenthetical explains the timing constraint — "if it had begun at 8:00, it would have been" logically requires a conclusion about the show ending, e.g. "over by 20 minutes before" or "finished by 9PM."
> 
> Final token "been" is part of a counterfactual conditional construction ("since it began at 8:00 PM would in reality have been"), forcing a past-perfect predicate completing the consensus about Typical Tonight's standard 30-minute runtime ending.

## row 6 — `openbmb/Ultra-FineWeb:en:13710` @ 107 tokens

**Source tail:**

> …eature. There are two basic models in the range. The standard A8GS has a built-in actuator which allows the rocker switch to be flipped off by applying a 3.3 or 5 V control pulse. It is the smallest switch yet to offer this reset feature.
> The A8GS-T also has an ‘off’ actuator and a secondary ‘signal

**matryoshka (FVE +0.690):**

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

**step400 reference (FVE +0.705):**

> Technical product description for a Panasonic switch, establishing bilingual component specs for electronics/automation readers. The text contrasts two features: resistor switching and electromagnetic type switches.
> 
> The sentence structure "Additionally, one of the panel miniature pushbutton switches..." continues listing innovations; the word "Additionally" signals another feature of electromagnetic switches being described.
> 
> The text is mid-sentence explaining Panasonic's dual-switch configuration: "combines operation control electrodes and a separate 'signal" — the single-quoted incomplete term strongly predicts a closing quoted compound noun like "signal output" or "contact/switch" describing feedback functionality.
> 
> Final token "signal" is part of an incomplete quoted compound noun phrase in a structured description — requiring a closing quote and noun, likely "pin" or "output" to name the second switch's signal/feedback function. "Also called a 'signal"

## row 7 — `openbmb/Ultra-FineWeb:en:58648` @ 1058 tokens

**Source tail:**

> …. Remember, the worst that can happen is they say no.
> Also, make sure the price you are being quoted includes VAT, number plates, delivery charges and road tax.
> If your dealer offers you an incentive (such as free insurance or an extended warranty) then ask them to put everything down in writing.
> 6.

**matryoshka (FVE +0.719):**

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

**step400 reference (FVE +0.774):**

> The article is a comprehensive consumer guide to buying a used or second-hand car, structured with UK Civil Justice Helpline bullet-point tips throughout; four numbered/titled sub-sections preceding this section addressed "Where and what to buy," "Financing," "Negotiating," and "Viewing and test driving."
> 
> The checklist pattern signals another post-purchase or transaction-step tip is expected next, likely covering paperwork, insurance, service history, or finalising the deal.
> 
> "6." is the final token — a numbered list item mid-sequence in the "useful tips when buying your used or part exchange car" section, immediately requiring a short tip title (e.g., "Test drive and paperwork," "Taking ownership").

## row 8 — `openbmb/Ultra-FineWeb:en:70163` @ 645 tokens

**Source tail:**

> …git amount. It doesn’t matter whether the officials live in rented accommodation or in their own home.
> Also interesting for you:
> Cyber Monday is in no way inferior to Black Friday: the bargain carousel keeps turning. We have compiled the best deals for you. One or the other Christmas present is sure

**matryoshka (FVE +0.591):**

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

**step400 reference (FVE +0.526):**

> Aggregated news header fashion: multiple short unrelated German health/lifestyle snippets blur together, suggesting a scraped portal page.
> 
> The article is concluding with an advertising pitch for a Christmas catalog/special offer, listing broader Christmas news categories and quoting HSE provider news. The closing teaser ("Browse through the Catalogue 2.0 and let your heart beat for the variety of products. The right souvenir is sure") signals an enthusiastic closing sentence.
> 
> Final token "sure" ends an incomplete clause ("We have collected numerous products especially for you. So a hit is sure"), requiring completion — "to be a hit" or similar affirmation. "Sure" strongly anticipates "to be found/included" or idiomatic: "surely" continuing the compound "so many diverse products for everyone, a hit is sure."

## row 9 — `openbmb/Ultra-FineWeb:en:94688` @ 181 tokens

**Source tail:**

> …It is designed to improve accuracy and speed when doing addition and subtraction skills. Got 2+2 down...and 13-7...ok. Quickly, what is 42+78? ...ok,now try... 93-27? Children can select their level: random ones&ones, tens&ones, and tens&tens questions. The player answers questions in an exciting 3D

**matryoshka (FVE +0.683):**

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

**step400 reference (FVE +0.800):**

> Software description format: educational app listing following conventions of age/gender, price, and feature highlights for a children's math learning game from 2007 UK developer.
> 
> Gameplay mechanic description building momentum: "Christian collects balls as he solves fun rapid fire maths problems. Player personally helps Christian by tapping the correct solution in an exciting 3D" — narrative is mid-sentence, describing the game's visual/environmental setting; "3D" strongly predicts "world" or "rendered environment."
> 
> Final token "3D" is part of an incomplete sentence fragment "Players have fun learning while playing in an exciting 3D," requiring an immediate noun referencing the game's graphical environment (e.g., "world," "engine," "animated surroundings"), consistent with casual animal-fetcher game marketing language from that period.

## row 10 — `openbmb/Ultra-FineWeb:en:42485` @ 97 tokens

**Source tail:**

> …n in FY 2021 Continuum of Care Program (CoC) competitive funding available to homeless services organizations across the country for supportive services and housing programs for people experiencing homelessness, and including up to $102,000,000 available for Domestic Violence (DV)Bonus projects.
> The

**matryoshka (FVE +0.765):**

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

**step400 reference (FVE +0.815):**

> Government/organizational document describing HUD funding opportunities, establishing factual announcement tone about CHAR 2023 NOFO release for community housing organizations.
> 
> Narrative momentum: the text has introduced the NOFO for Continuum of Care funds and noted its key features (returning to one-year implementation cycle), signaling more specifics about eligibility, deadlines, or provisions are expected next.
> 
> "The" begins a new sentence continuing elaboration about the NOFO's details — likely referencing a specific component, deadline, or requirement, e.g. "The NOFO was released on..."
> 
> Final token "The" opens a noun phrase mid-announcement, most immediately constrained to a capitalized noun referencing the NOFO, HUD, or a deadline/notice detail. "The November 2022 NOFO" formula common in such posts suggests specifics about the notice's provisions follows.

## row 11 — `openbmb/Ultra-FineWeb:en:86506` @ 277 tokens

**Source tail:**

> …but the amount at which they ascended slowed in the 3rd quarter,” suggests Lawrence Yun, NAR chief economist. “I anticipate extra homes to hit the market place as early as following calendar year, and that further stock, blended with larger mortgage premiums, ought to markedly lower the pace of rate

**matryoshka (FVE +0.418):**

> rate housing price growth slowing reduces expected rate inflation continuation
> Real estate economics sentence completing "curbing of rate" mirrors earlier claim about rapid price increases
> NAR report context requires continuation describing home price increases metric
> " constrains future rate of value increases" structure parallel "rate" needs noun like "home prices" or "price escalation"
> US housing market report genre: direct quote format or paraphrase nearing conclusion
> "Supply constraints have been key cause of lately rapid escalation, so sluggish sales quantity would be beneficial for curbing of rate" - this repeats "price formula/increase" concept completing the phrase "rate of rate..." implies earlier "speed/acceleration" repetition typo correction needed: "rate growth" or "price growth"
> Fixed quantitative claim "limited slowdown in sales would help restrict pace of rate" expects specific buyer affordability/read-only metrics concluding
> NAR administrator quoted earlier about […]

**step400 reference (FVE +0.704):**

> Ongoing data-driven real estate report format, mixing NAR statistics, charts, and quotes about 2024 U.S. housing market outlook.
> 
> Established pattern of describing price momentum, affordability issues, and regional trends; the passage is building toward explaining consequences of constrained supply driving unsustainable price growth.
> 
> Ronald Chan's quote is mid-sentence, explaining the relationship between mortgage prices, buyer demand, and "it mid considerably dampen the speed of rate..." — the sentence promises a concluding phrase about price gains/acceleration completing the thought. The endpoint formula ("coupled with limited new building... have in flip placed a cap on the tempo of rate") requires a noun completing "temper the rate," such as "bills" or "gains."
