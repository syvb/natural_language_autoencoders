# Left-matryoshka NLA — example outputs

15 held-out activations (doc-disjoint `av_eval_v3`), verbalized by the **v3 warm-start AV** and the **suffix-RL iter_50 AV** (T=1, 120-token cap, as in training). Per-sample cosine through each pair's own critic: full text / first-10-tokens / last-10-tokens.

---

## Sample 1

**Context (last 300 chars):** `…First Song Before First Grade She has a Master’s Degree in Music and is not afraid to use it. As she demonstrated during her talent portion of the competition in which she performed a selection from the opera La Boheme, music is in her DNA. Not only can Nia sing like a dream, she also writes some of`

**Warm-start AV** — cos full 0.755 · first-10 0.612 · last-10 0.388

```
"some of" opens an incomplete list or corpus phrase
Perri's own songwriting/pencil activity expected next
Pulsing rhythm metaphor continues; creative production implied
Parallel structure: borrowing lyrics, now lending/singing her own tracks
Celebrity name-dropping pattern continues; more collaborators likely
Biographical profile format; historical narrative momentum continues
Rap/Hip-hop genre context shapes lyrical/deep background described
Formal yet accessible celebrity biography register maintained throughout
Her collaboration/accolades narrative building toward a climactic achievement
Third-person factual tone; specific details about her craft expected
```

**Suffix-RL iter_50 AV** — cos full 0.852 · first-10 0.651 · last-10 0.730

```
Final token "some of" begins a phrase listing Emily's songwriting abilities.
Sentence "And not only does she sing some of" requires completion.
Subject "she even composes some of" expects a noun phrase describing her music/songs.
Biographical profile of youthful singer/bassist continues with achievements.
"not only does she sing some of" implies a "but also" corollary listing her songwriting.
Subject "some of" expects "her own music/songs" or "her own compositions."
Article about teen rock singer profiles her talents and contributions.
Phrase "writes some of"
```

---

## Sample 2

**Context (last 300 chars):** `…he most complex and demanding situations.
Whatever it takes. Wherever it’s going.
Delivering to anywhere in the world -
even most remote locations
Montreal, May 18, 2021 – Bombardier today announced its collaboration with Sterling Global Aviation Logistics, a global leader in aviation transportation`

**Warm-start AV** — cos full 0.609 · first-10 0.565 · last-10 0.447

```
Final token is mid-sentence about Rightech's expertise
Organization introduction mid-sentence, noun phrase completion expected
"Increased competition" frames Rightech advantage narrative continuing
Company description formula: experience/background details follow introductions
Formal business/marketing register maintained throughout text
Pattern of completing embedded clauses in serial sentences
FlyNet and Rightech as twin entities, parallel description likely
Industry-specific jargon: aviation charter, transport logistics domain
Superlative/competitive framing suggests further capability evidence follows
Seniority/leadership narrative building toward partnership announcement climax.
```

**Suffix-RL iter_50 AV** — cos full 0.890 · first-10 0.718 · last-10 0.758

```
Final token "transportation" is mid-phrase, noun in compound noun phrase.
Company profile sentence about Transfleet's expertise continues.
"leading air freight transportation" phrase expects noun phrase completion.
Press release "About" section describing company background follows.
"recognized globally as experts in aeronautical logistics transportation" expects "services" or "and logistics solutions"
Biographical/company boilerplate structure describing company's specialization.
"specializes in aeronautical transportation" is incomplete, describing company's expertise.
Press release genre: company description section follows brand name.
"industry leader in air freight transportation" phrase
```

---

## Sample 3

**Context (last 300 chars):** `…nd waiving commercial rent for 6 months for small businesses that lease space through the Ministry of Tourism and Transport. Additionally, the VAT rate will be reduced to 5% for 3 months on commercial rent for small businesses and a reduction of the travel tax for 6 months to boost visitor arrivals.`

**Warm-start AV** — cos full 0.852 · first-10 0.812 · last-10 0.574

```
Final stimulus measures section likely continues or concludes
Blue economy tourism relief programs being enumerated in list
XPRS overarching relief package being itemized with dollar amounts
Formal government press release genre, policy announcement register
Parallel structure: stimulus measure name, agency, dollar amount pattern
Blue economy likely includes fisheries, marine, or island tourism measures
Specific financial figures cited suggest more may follow
Document likely ending or transitioning to next relief category
West African regional context suggests limited stimulus scope
Date/time-stamped release signals complete or nearly complete content block ending soon. Cut-off character likely follows next element. New
```

**Suffix-RL iter_50 AV** — cos full 0.899 · first-10 0.726 · last-10 0.690

```
Final token is a period ending the list of relief measures.
Enumerated list of tax/financial relief measures just completed.
Government press release format; additional policy details may follow.
"Additionally, tax relief on fuel will be doubled for three months" ends the list.
Parallel structure: VAT, duty waiver, entry fees, tourism relief mentioned.
Sentence-ending period signals list completion; new paragraph expected.
Sequential policy announcements; further measures or summary may follow.
The article details a package; summary or next government actions expected.
"Also included relief on VAT... and fuel excise duties..." list closed.
Montserrat
```

---

## Sample 4

**Context (last 300 chars):** `…ll pipe (resistance strength greater than 1100 Mpa) are available. (2) Large pitch threads can effectively improve the tension resistance of threaded connections. (3) The drill pipe adopts equal strength design, which reduces the overall weight of the drill pipe and effectively improves the drilling`

**Warm-start AV** — cos full 0.875 · first-10 0.690 · last-10 0.412

```
Final token "drilling" mid-phrase, noun expecting continuation
Technical description of XC2.5 product specs continues
Non-commutator motor advantages being enumerated, more likely
Product specification list pattern: efficiency, performance, now drilling outcome
Formal technical marketing register with precise engineering vocabulary
"Enhancing the drilling" phrase requires object: speed, capacity, etc.
Parallel structure: prior sentences end with noun phrases after run-on lists
Chinese manufacturing export catalog genre conventions throughout
Cubic meters/tonnes capacity figures pattern likely to recur
Sentence completion expected before new product feature paragraph begins
```

**Suffix-RL iter_50 AV** — cos full 0.908 · first-10 0.701 · last-10 0.523

```
Final token "drilling" begins a noun phrase describing percussion effect benefits.
Technical sentence structure: "can improve the drilling" expects noun phrase completion.
Product feature explanation: mining drill performance improvement being described.
List of advantages of cone bit replacing QJ2150 is ongoing.
"improve the drilling" phrase expects "efficiency" or "drilling speed/efficiency/ability."
Technical specification pattern: improving drilling performance in down-the-hole drilling context.
Sentence structure "can improve the drilling" expects noun like "efficiency" or "speed."
Chinese mining machinery product description, technical
```

---

## Sample 5

**Context (last 300 chars):** `…ith part-to-part gain tolerance of ±1%. The ACPL-C784 utilizes sigma-delta (Σ-∆) analog-to-digital converter technology, chopper stabilized amplifiers, and a fully differential circuit topology. Together, these features deliver unequaled isolationmode noise rejection, as well as excellent offset and`

**Warm-start AV** — cos full 0.896 · first-10 0.677 · last-10 0.600

```
"and" signals continuation of parallel list structure
List of high-precision measurement qualities being enumerated
"offset and X" pattern expects a noun phrase like "drift"
Technical specification prose: dense noun phrases, passive constructions
ADC datasheet context: precision metrics dominate vocabulary choices
Earlier terms "offset and gain" suggest gain-related noun follows
Formal datasheet register: Latinate precision terms throughout
Sentence likely continues with additional isolation feature claims
Closed-loop video equipment application just stated, supporting specs follow
Superlative/advantage framing ("highest...highly precise") continues throughout paragraph
```

**Suffix-RL iter_50 AV** — cos full 0.932 · first-10 0.691 · last-10 0.782

```
Final token "and" begins a list continuation after "offset and"
List pattern: "offset and" expects parallel noun phrase following
Product specification document listing sensor performance qualities
"优秀的 offset and" requires completing "offset and [drift/accuracy]" noun phrase
Technical datasheet enumeration: "excellent offset and" expects "non-linearity" or "linearity"
Phrase "excellent offset and" begins a paired deficiency metric list
PMT-100 section lists temperature stability benefits, accuracy next expected
"low offset and" implies paired performance metric: "non-linearity
```

---

## Sample 6

**Context (last 300 chars):** `…of fresh housing. Pankaj Bansal, Director of M3M said “We focused on selling our under-construction projects; this segment moved very fast for us. Our focus remained on timely delivery of the projects to our customers”.
Inventory levels in realty markets such Mumbai, Pune and Thane are registered at`

**Warm-start AV** — cos full 0.666 · first-10 0.528 · last-10 0.450

```
Incomplete sentence ending mid-phrase requires immediate noun phrase
Quantitative data expected, likely percentages or specific figures
Pattern of providing stat/citation after claim, mirroring "Source says" format
Regional market broken up by regions, data per market expected
Overall market context stated at start, now elaborating component metrics
Formal market research report register continues throughout
Prior data points suggest "under 15 percent" as plausible figure
Low prices driving growth narrative supports optimistic numerical claims
Comparative structure established, pre-pandemic figures may follow
CEVA/analyst quotes pattern may continue
```

**Suffix-RL iter_50 AV** — cos full 0.888 · first-10 0.601 · last-10 0.717

```
"At" signals a quantitative/descriptive phrase beginning.
Inventory levels stat about to be stated numerically.
Sentence "The levels of inventory across all cities are at" expects a number or low figure.
Article structure: data points about inventory levels being summarized.
Pattern: "inventory levels across all cities are at" precedes a specific value.
Industry report tone: statistics about Mumbai home prices inventoried.
"are at" expects quantity or percentage describing inventory lows.
"inventory level across selected cities have been recorded at" signals specific figures follow.
Article consistently cites low inventory numbers; continuation expected.
"at
```

---

## Sample 7

**Context (last 300 chars):** `…heduled Passenger|
|Departure airport:||Nizhnevartovsk Airport (NJC/USNN), Russia|
|Destination airport:||Tyumen Airport (TJM/USTR), Russia|
PROBABLE CAUSE: Reduction in engine thrust during take-off, probably because the crew closed the throttle.
» Soviet Transports
» ICAO Adrep Summary 1/88 (#52)
`

**Warm-start AV** — cos full 0.658 · first-10 0.542 · last-10 0.502

```
Final token closes a proper noun/title bibliographic citation
Catalog entry format suggests numbered list continues next
Aviation accident database entry pattern throughout text
Medallist or NTSB may continue as related citation entries
Next entry likely similar disaster/event title follows
Numerical or date-based accident report listings pattern
FAA citations, references, or related incidents plausible next
Formal archival catalog register maintained consistently
Geographic or operator attribution typically follows accident citation
List appears to be unclosed, implying more entries follow "Related" category label. Authorships expected again. Second citation follows same format pattern
```

**Suffix-RL iter_50 AV** — cos full 0.687 · first-10 0.617 · last-10 0.514

```
Final token is a table/reference section label ending. "Accident:Estimated" ends a bibliography/source citation entry.
Aviation accident report reference section, bibliography format expected.
Table of contents or linked citations pattern suggests document ends here or transitions to new section.
Structured accident report format: facts, context, reference.
"Aviation Safety Network" citation implies accident database reference entry.
Pipe-delimited reference label "Accident:Estimated" indicates formatting for database metadata.
Document likely ends or new section titled "References" follows.
No further narrative content expected; references are final.
Structured metadata pattern: "Acc
```

---

## Sample 8

**Context (last 300 chars):** `…he well-known columnist, "I read you every day. I love your articles." A hardly enthusiastic Miguel reluctantly takes the piece of paper to sign his name and tells her, "I don't write to be read. I write to be paid."
Enjoying This Commentary? There's More to Love
Subscribe to MediaVillage to receive`

**Warm-start AV** — cos full 0.735 · first-10 0.609 · last-10 0.211

```
"receive" expects direct object: newsletter, updates, news
Email signup/subscription call-to-action pattern continues
Marketing/promotional boilerplate language register expected
Free trial or newsletter editorial service being promoted
Interrupted mid-sentence requiring noun phrase completion
Press release boilerplate footer convention, formulaic continuation
"Businessweek BusinessNewsDaily" as subject demands full predicate
Signaling exchange: email input field prompt likely follows
Specific newsletter type or frequency often named next
Formal news wire structure ending with site promotion textture text described next. Frequency drops significantly after token. Stable compression with no
```

**Suffix-RL iter_50 AV** — cos full 0.806 · first-10 0.444 · last-10 0.513

```
"to receive" expects object/complement after "subscription to receive"
Email newsletter signup prompt requesting reader subscription benefits
Call-to-action phrase "subscribe to Broad //
Tech Investor Net to receive" expects newsletter content listing
Formal subscription/registration prompt grammar: "to receive" needs object
"Subscribe to Broad //
Tech Investor Net to receive" implies receiving alerts, updates, or content
Email newsletter signup template: "to receive" begins benefit description
Phrase "to receive" in subscription prompts typically precedes "the latest news"
The promotional signup block signals reader benefit description follows
"subscribe to
```

---

## Sample 9

**Context (last 300 chars):** `…so used in ABS, PS,PVC, HIPS, PA, PC, PP, PE,PET,EVA, PVC, etc.
Rate of use: (per 1000kg of substrate) Polyester: 75-300g EVA, PP, nylon: 75-200g
25kg per fiber drum
STOREAGE: Optical brightener OB-1 (TDS) should be stored in a cool, dry area. Extended storage at elevated
temperatures or exposure to`

**Warm-start AV** — cos full 0.908 · first-10 0.634 · last-10 0.511

```
Final token "to" begins a prepositional phrase listing exposure conditions
Environmental stress conditions being listed (humid/damp, or contaminant exposure)
Fourth bullet point continuing storage/preparation precautions list
Parallel structure: each bullet ends with specific handling directive
Technical product safety specification document genre
Pronoun inconsistency ("its") revisited: likely transitional artifact
Additional exposure to outdoor/environmental elements typically follows
Storage conditions being enumerated (light, moisture, contamination)
Formal technical/commercial food ingredient specification register
Numbered or bulleted safety warnings near document conclusion section endpoint.5 section)
```

**Suffix-RL iter_50 AV** — cos full 0.940 · first-10 0.634 · last-10 0.859

```
Final token "to" begins a prepositional phrase after "or exposure to"
Warning about exposure conditions completing理解: moisture/humidity expected next
Chemical storage/compatibility warning sentence structure: "exposure to" needs noun phrase
Product specification safety data listing incompatibilities and storage hazards
"Exposure to" phrase typically followed by moisture, light, or high temperatures
Sentence: "When stored at low temperatures or exposure to" needs completing
Corrosion inhibitor product datasheet listing storage/compatibility contraindications
"Exposure to" precedes moisture, sunlight
```

---

## Sample 10

**Context (last 300 chars):** `…les) from Maldives to Vietnam, It takes 4.49 hours to arrive.
|GPS Coordinates (DMS)||1° 58´ 38.1000'' N |
73° 32´ 9.9600'' E
Map of Maldives
Maldives Distances to Countries
|Distance from Ireland to Maldives||9,287 km|
|Distance from Maldives to Tajikistan||4,108 km|
|Distance from Fiji to Maldives`

**Warm-start AV** — cos full 0.723 · first-10 0.598 · last-10 0.515

```
Final token is an incomplete table cell, value expected next
Pipe-delimited table row structure requires closing pipe character
Distance value expected between 3594.999 and 4094.409 range
Pattern: "Distance between City Country and City Country = km" repeating
Established table rows incrementing via China and India city combinations
Trailing pipe character expected to close the final cell
Numeric distance value in parentheses typically follows table header
Tabular content listing-distances section mirrors distance module header exactly
Next row will likely increment distance and add more destination
```

**Suffix-RL iter_50 AV** — cos full 0.673 · first-10 0.598 · last-10 0.714

```
Final token is a table cell starting a "distance to Maldives" entry.
Table row pattern: "Distance from India to Maldives" expects distance value/pipe.
Pipe-delimited table row structure: "|Distance from Bangladesh to Maldives||"
List of distance entries repeating: "Distance from X to Maldives" pattern continues.
Table column pattern: "|Distance from [Country] to Maldives|Distance from India to Maldives"
Pipe character indicates another cell value or distance value follows.
Previous table entries show distance values with miles/kilometers after pipe.
"Distance from Maldives to India
```

---

## Sample 11

**Context (last 300 chars):** `…tations and traps, resulting in more optimum well locations.”
Traps that had been overlooked or misinterpreted with 2-D lines spaced 1,500 to 2,000 feet apart were delineated and drilled, Johnson said.
“Careful record editing, velocity analysis and first-break muting are essential to optimize 1- and`

**Warm-start AV** — cos full 0.499 · first-10 0.445 · last-10 0.400

```
Truncated mid-phrase near a mathematical value "1"
Numeric continuation expected: scale value like "2-krm"
List of three factors: spatial stratification precedes resumption
Scientific agricultural domain; formal technical register continues
Previously established depletion sequence: reduction from 1-commercial through 1- and 2-krm
Pattern of paired measurements likely: "1-commercial and 2-krm" mirroring earlier phrasing
Sentence structure mid-clause; resolution involves explaining impact on canyon development
Causal argument momentum: spatial stratification identified as key runaway production variable
```

**Suffix-RL iter_50 AV** — cos full 0.843 · first-10 0.571 · last-10 0.369

```
Final token "and" begins a continuation of Nr 1- and X-range contrast.
Hyphenated list "1- and" expects a second numeric range following.
Sentence structure: "use of 1- and" anticipates "2 mm datasets" or similar.
The paragraph discusses limitations of 1- and 2 mm datasets progressing.
Pattern of citing "1- and 2- dimensional datasets" throughout text.
Phrase "1- and" begins enumeration of small stripmine dimension ranges.
Academic/industry report on underground mining dimensional data limitations.
Sentence: "end users are
```

---

## Sample 12

**Context (last 300 chars):** `…ms of the environment management at ChTPZ and PNTZ have been named efficient.
SAI Global carried out the audit at ChTPZ and TUV Rheinland – at PNTZ.
This year ChTPZ has started a large-scale project on the construction of the wastewater purifying facilities. Investments total 600 million rubles.
PNT`

**Warm-start AV** — cos full 0.668 · first-10 0.660 · last-10 0.333

```
"PNT" is a location/acronym fragment requiring completion
Question format established, callback to PNT suggested answer follows
Previous pattern: business fields expanded, this mirrors that structure
PNT company name listed earlier, completing acronym "RAM" expected
Question form likely returns, addressing PNT specifically this time
Green initiatives topic continues, shared interests likely mentioned
Baltic/Norascon company introductions typically follow earlier PNT mention
Formal business communication register, polished and professional tone
Glory Finaly founder profile may cite shared environmental ambitions
Video/audio engagement element may reappear as earlier
```

**Suffix-RL iter_50 AV** — cos full 0.768 · first-10 0.638 · last-10 0.693

```
"PNT" is an abbreviation starting a new sentence/phrase.
Organization name "PNT" begins a new sentence, likely repeated entity.
Ministry of Ecology's section mirrors Chernomyrdin Steel's structure.
Pattern: "PNT" here begins "PNTC" company name continuation.
"KNT" likely follows "PNT" to complete the organization name.
Document structure: second organization (PNTC) now described after Chernomyrdin Steel.
Formal industrial/sustainability report comparing two companies.
"PTN" is the abbreviation for PNTC, the second
```

---

## Sample 13

**Context (last 300 chars):** `…inter for your personal use? Do you look for the steps to connect Canon printer to your wireless connection? If so then you have arrived at the accurate platform. Here in the below section learn how to connect your Canon printer to a wireless network. Before you start, the process checks your router`

**Warm-start AV** — cos full 0.857 · first-10 0.779 · last-10 0.473

```
Final token "router" signals ongoing step-by-step technical instruction
Numbered list pattern continues; item 2 expects completion of next instruction
Technical setup instructions sequence: connect router typically precedes Wi-Fi/WAN settings access
How-to guide genre expects UI navigation steps like "settings," "connect," "enable"
Low-quality translated/English-spun text signals awkward phrasing continues
Logical prerequisite setup: router connected before Wi-Fi setup expected
List items follow consistent imperative verb plus object structure
Router configuration often involves ensuring correct network name/password setup
Step numbering suggests possibly more future steps (3,
```

**Suffix-RL iter_50 AV** — cos full 0.919 · first-10 0.801 · last-10 0.830

```
Final token "router" begins a noun phrase about router setup/configuration.
Instructional how-to guide on wireless printer setup steps.
"Check your router" phrase expects adjective or verb phrase next.
List of setup steps: "First, you should check your router" implies verification follows.
Topic: printer connection requires router compatibility and settings.
"Check your router" is incomplete phrase; connectivity or configuration expected.
Guidelines section lists sequential steps; checking router's capability logically follows.
"Please check your router" signals a condition or requirement about WiFi/router.
"Make sure your router" expects noun phrase describing router
```

---

## Sample 14

**Context (last 300 chars):** `…TRENDING: Avalanche swallows skiers; Drivers get a surprise after parking on thin ice
Winter's frigid air can bring with it possible plumbing problems, including frozen pipes.
Snow days are exciting for children, but harsh cold and treacherous wintry weather can make it difficult to get outdoors and`

**Warm-start AV** — cos full 0.709 · first-10 0.697 · last-10 0.540

```
And conjunction expecting continuation of outdoor activity limitations list
Negative outdoor conditions causing indoor-seeking behavior in children
Child safety/health topic requiring practical guidance to follow
List of hazardous activities at home likely elaborated next
Tips format signals concise, actionable safety advice sections
Humid, disturbed summer conditions mentioned but not yet defined
Seasonal home-hazard framing expects fire risks and tips
Parallel structure with flood-preparedness section implies similar guide tone
Audience is parents/caregivers motivating their children's participation
NAHB educators framing suggests family-friendly, practical recommendations aheadazione ergonomica acqua
```

**Suffix-RL iter_50 AV** — cos full 0.886 · first-10 0.687 · last-10 0.812

```
Final token "and" begins a continuation of a list or clause.
Sentence listing indoor activity difficulties during winter weather.
"get outside and" expects second activity or noun phrase.
Snow prevents outdoor activities, indoor alternatives contrast expected.
Health article about senior health impacts of winter weather.
"unable to get outside and" implies paired activity/situation follows.
"can keep children indoors and" requires completing contrast phrase.
Phrase "get outside and" lists two parallel participial actions.
Winter weather causing indoor confinement, indoor activities follow.
"can't get outdoors and" expects "stay indoors" activity phrase.
```

---

## Sample 15

**Context (last 300 chars):** `…iply the expression by a float number to set the scraggle intensity. In the next example, I intentionally use an extreme multiplier to show the difference.
As final step, let’s use a Merge node to combine the two subnetworks.
Controlling the Hair Clumps at the Root
If you look at the root, there are`

**Warm-start AV** — cos full 0.738 · first-10 0.654 · last-10 0.570

```
Final token "are" opens a clause about split-zone polygon edges
Visual observation of edge geometry just made, description expected
Tutorial pattern: state problem, then show solution results
Certificate masking tool explained earlier; it likely addresses splitting issue
Split-zone un-polygonized issue established in Week 5 section
Walkthrough style continues: analytical observation before fix application
Perspective and drawing distortion context frames the issue
Previous Masking Tool workaround may be revisited for this case
Informal instructional register continues with first-person observation
Specific visual artifacts (gaps, intersections, angles) likely described next
```

**Suffix-RL iter_50 AV** — cos full 0.888 · first-10 0.648 · last-10 0.665

```
Final token "are" begins a sentence describing remaining gaps/issues.
A section on Bézier Subdivision Adjustments is now underway.
The sentence "After applying subdivision, there are" expects a noun phrase describing remaining problems or visible imperfections.
Pattern of listing deficiencies: the Bézier row lacks smoothness and clean edges.
"Here are" begins a description of visible artifacts or irregularities after subdivision applied.
Document structure: each technique section explains a problem with subdivision results.
"After applying subdivision, there are" anticipates a noun phrase describing remaining problematic elements.
The tutorial consistently highlights areas needing
```
