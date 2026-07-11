# Quote-penalty RL: sample comparison

16 held-out `av_eval` prompts (distinct docs), identical prompts + per-sample seeds across checkpoints, 200 max new tokens.

| checkpoint | quote chars/sample | zero-quote samples | 'final token' echo | CJK |
|---|---|---|---|---|
| preRL | 15.25 | 0/16 | 16/16 | 0/16 |
| overlap100 | 14.12 | 0/16 | 16/16 | 0/16 |


---

## Sample 1/16 — `openbmb/Ultra-FineWeb:en:62062@227`

**Context tail** (activation = last-token L20 state):
```text
...nders didn’t have to parade on stage in swimsuits. Whatever the cause, viewership dipped to less than 5 million people who actually watched the live broadcast on ABC.
If you were one of the viewers who couldn’t tear yourself away from the Packers-Bears collision on the football field, here’s a short Lightning List of Four Facts You Should Know About Nia Franklin The New Miss America.
#4 Wrote Her First Song Before First Grade She has a Master’s Degree in Music and is not afraid to use it. As she demonstrated during her talent portion of the competition in which she performed a selection from the opera La Boheme, music is in her DNA. Not only can Nia sing like a dream, she also writes some of
```

**preRL** (quote chars: 15):
```text
Blog-style headline and body structure with music/band celebrity stories focusing on Houston's indie culture, blending humor and social commentary around a radio host's powerful presence.

The sentence "While Janee Tierney sings some of the country's most acclaimed hits, the incredibly talented woman also writes some of" continues a boastful self-description listing her musical abilities and songwriting skills.

Final token "some of" ends mid-clause ("but she actually composes some of"), part of a parallel clause about her literally creating songs AND writing them herself — expects "her own lyrics" or "the most beautiful music" or "her own lyrics too," completing the boast about writing and creativity.
```

**overlap100** (quote chars: 13):
```text
Blog-stylelocal news roundup format with Texas music column vibe covers retro entertainers honoring coveted awards — alternating celebrity profile snippets about Austin venue acts.

The second quoted bio item shifts to personality description pairing Scott's performances and songwriting credentials, introducing parallel claim about female influencer's creative output vs. accolade nature — expects continuation about her lyrics or poetic talent or similar creative sources.

Final token "some of" opens a quoted attribution clause describing the writer's compositional or creative sourcing ("While she herself scores many hits, she actually pens some"), immediately expects "her own lyrics" or "theliest imagery" or "his own worst ideas."
```

---

## Sample 2/16 — `openbmb/Ultra-FineWeb:en:64875@638`

**Context tail** (activation = last-token L20 state):
```text
...ueens; and New York City Council Member Karen Koslowitz of Queens.
Last month, the two organizations released the QueensWay plan, a blueprint for the 3.5 mile-long QueensWay. For more information, visit www.thequeensway.org.
In 2013, a public poll of Queens residents showed strong support for the QueensWay.
The Trust for Public Land creates parks and protects land for people, ensuring healthy, livable communities for generations to come. Nearly ten million people live within a ten-minute walk of a Trust for Public Land park, garden, or natural area, and millions more visit these sites every year.
The Friends of the QueensWay (FQW) is a grassroots community organization – consisting mainly of
```

**preRL** (quote chars: 14):
```text
Local NYC civic advocacy article with progressive blog tone presenting a street improvement project, combining planning and environmental/neighborhood empowerment themes.

The paragraph structure "The Greenbelt Coalition is a grassroots advocacy group committed to developing the East River Greenway corridor. Comprised primarily of" introduces a second quote or list, likely naming community members from South Orange County, with "ordinary citizens" context expected.

Final token "of" ends an incomplete noun phrase "including primarily a group primarily consisting of," part of a civic description clause ("This project is a coalition consisting mainly of..."), strongly expecting names like "residents, neighborhood leaders, and local activists" or "citizens living along the route."
```

**overlap100** (quote chars: 12):
```text
Local civic/blog journalism article structure with alternating political advocacy panels documenting a Bay Area transportation bill project; second entry shifts to media event announcement format.

The opening quoted block header introduces Local Movement Caucus campaign focus around Mission Creek corridor history, signaling the second contributor bio description likely lists neighborhood residents or community advocates/local activists with grassroots roots versus formal organizations.

Final token "of" ends an quoted attribution phrase listing contributors origin description ("Primarily a diverse coalition largely comprised, largely consisting including"), immediately expecting noun like "residents" or "neighborhood activists," or "Mayor/Knowledgeholders" or "those who remember" or similar.
```

---

## Sample 3/16 — `openbmb/Ultra-FineWeb:en:10152@120`

**Context tail** (activation = last-token L20 state):
```text
Fiber optics products have a huge variety of applications and uses. For example, products such as repeaters, adapters, and various power supplies fall under this category. So optic receivers, APC and UPC connectors. Solid Signal carries all of these fiber optics products and more. We stock our online inventory with products from manufacturers that includes Blonder Tongue, Cabletronics, and Ripley. Solid Signal is your No. 1 source for the best fiber optics products. We also offer a huge online selection of the best commercial equipment.
The GarrettCom model 5846 Serial Fiber Links are
```

**preRL** (quote chars: 11):
```text
Technical product webpage/blog format with product descriptions and specs, shifting into a second product listing promoting industrial Ethernet devices from Industrial Supply.

The phrase "The ICOM KMF-800 Series Low Profile DBA Adapter Modules are" signals a new product intro, likely presenting a specific model or product feature, likely describing a high-quality test instrument used for multi-channel test equipment.

Final token "are" ends an incomplete noun phrase ("The ICOM KMF-800 High Speed Synthetic Couplers are"), immediately requiring a verb or noun phrase describing the product's functionality, capabilities, or specifications, likely "compact and reliable hobbyist test adapters" or "three high quality 2U rack modules."
```

**overlap100** (quote chars: 12):
```text
Product catalog webpage mixing technical articles with promotional listings, alternating pasteurized industrial/comms hardware content establishes parallel headline listing format with listings shown.

The opening paragraph introduces a new product item header "Ortronics MultiPort TDR123 Series" signals an announcement or spec item description expected next, likely describing functions, features, or bandwidth capabilities for rackmount attenuators.

Final token "are" ends an introductory quoted product description clause beginning "The FS Electronics Two Channel Model MM-PRO III Trackmuster Clips WTW TESTS Blocks," expecting noun phrase like "high quality single-point detectors" or "a great solution" or "made."
```

---

## Sample 4/16 — `openbmb/Ultra-FineWeb:en:14658@179`

**Context tail** (activation = last-token L20 state):
```text
...et. Metro Transit operates the public transportation system for the St. Louis region, which includes MetroBus, a 400-vehicle clean-burning diesel bus fleet that services a 550-square-mile area in eastern Missouri and southwestern Illinois.
“The introduction of BEBs into the MetroBus fleet next year represents our commitment to providing economically and environmentally sustainable mobility options as well as an excellent transit experience for our riders,” said Jessica Mefford-Miller, Metro Transit executive director. “We will be leaner and greener by introducing this BEB technology, not only with the buses but also through the charging infrastructure and operating facilities.”
Metro Transit
```

**preRL** (quote chars: 19):
```text
News release format with transportation/infrastructure equipment manufacturer highlights features of Siemens Rail's hybrid electric bus contract, establishing technical and market context.

The paragraph structure "The project is part of a $9 million electric vehicle procurement contract with Siemens Rail to deliver the full transit fleet" mirrors earlier statements, now introducing Lincoln Transit's decision context with "Transit Authority's prior sustainability goals."

Final token "Transit Transit" repeats an incomplete clause mid-sentence ("Clark County Transit Transit"), directly continuing " transit officials have selected the new fleet," expecting continuation like "selected the project to begin construction" or "has been planning to reduce its diesel fleet" or "is the lead agency for project."
```

**overlap100** (quote chars: 14):
```text
News-media format with promotional/infrastructure report article detailing Siemens energy contract awards for bus station solar canopy project includes imagery captions and exposition tone.

The structured pattern establishes article introduction and second project announcement section with quoted facts, now expected parallel content about owner motivation or selection rationale primes the headline partner'sCompra's project award context and agency's prior infrastructure goals.

Final token "Transit Transit" repeats the naming subject opening quote's source, immediately continuing the official partner description clause — likely "has selected the solution" or "is seeking its sustainable improvement portfolio" or "plans to report annual spend figures" or "already owns sites."
```

---

## Sample 5/16 — `openbmb/Ultra-FineWeb:en:30894@517`

**Context tail** (activation = last-token L20 state):
```text
...etter shade for the soils beneath them but they also have the ability to “sweat” by releasing water through their leaves to stay cool on hot days. This evaporative cooling lowers air temperatures making a field of native grasses significantly cooler than a typical suburban lawn.
3. Prescribed Burns
If you’re on a larger piece of land, you may want to consider prescribed burns. Habitat disturbances such as fire are natural components of the recipe for healthy ecosystems. Periodic fires allow for the germination of seeds for fire dependent species like long leaf pines and bear oaks, help create areas of early successional habitat like grasslands and meadows, and help reduce leaf litter thereby
```

**preRL** (quote chars: 12):
```text
Student written blog post detailing wildlife/land conservation with scientific and educational tone highlighting unique species and species work alongside a park diarist.

The sentence structure "Upland birds, specifically prescribing fire and grazing controls, also aid against forest succession thereby increasing soil nutrients and removing excess leaves thereby" lists ecological benefits of winter burn activities, continuing a third example with soil oxygenation or fire management.

Final token "thereby" ends an incomplete clause mid-sentence ("removing leaves during winter and clearing soil thereby..."), expecting a noun phrase describing soil benefits like "reducing fire fuel" or "improving growing season" or "enhancing nutrient cycling and drainage."
```

**overlap100** (quote chars: 11):
```text
Blog style student article detailing historical/educational nature charter identities with cowboy/wildlife content marrying farming and ecology perspectives via social media posts.

Second quoted bullet-point item's description lists ecological benefit functions of burned species, with a parallel clause about forest management practices reducing fuel/scape or soil health metrics — expects continuation describing seasonal effects or nutrient cycle benefits like cleared space or sunlight access.

Final token "thereby" ends an enumerated parenthetical clause listing ecological impacts ("Removing organic layers and dormant growth reducing winter biomass plus drainage, decreasing soil..."), immediately expecting noun like "enhancing oxygen" or "promoting growing seasons" or "reducing."
```

---

## Sample 6/16 — `openbmb/Ultra-FineWeb:en:85700@56`

**Context tail** (activation = last-token L20 state):
```text
Buy luxury perfumes online in UAE only at Hover8Store
Your Cart is Empty
I Agree with the Terms & Conditions [View Terms]
Stay Safe - Shop Online
Rosendo Mateu Nº 5 Floral, Amber, Sensual Musk luxury niche fragrance for Women and Men
```

**preRL** (quote chars: 18):
```text
Arabic/Egyptian perfume product description with stylish product image format suggesting a perfume or fragrance product listing with attributes like "Vintage" quality.

The phrase "Limited Edition Perfume for Men and Women - Unisex Eau De Toilette for Men and Men" signals a title or product naming promising dual gender appeal, strongly implying a product description continuing about the scent or duo.

Final token "Men" closes an incomplete compound noun ("For Men and Women and Men"), part of a product title or descriptive clause structure ("Limited fragrance for Men and Women"), expecting continuation like "is" or "with Style" or "from Italy" or "a combination of elegance and intensity."
```

**overlap100** (quote chars: 16):
```text
Egyptian fashion/product listing format with tags/image labels suggests retail SEO oratoire post structure around fragrance/trend item description.

The headline introduces a paired product title pattern ("Gothic Exotic Cologne Masculine & Feminine") signaling a product announcement or label description, strongly implying a joint gender descriptor or theme context about mixed style or dual characteristics for an age-old scent.

Final token "Men" closes the targeted demographic label pair ("For Style & Spirit for Boys & Men & Women"), immediately expecting continuation like "." or "is" or "of luxury" or "This unique blend" or "comes with dates" or similar caption.
```

---

## Sample 7/16 — `openbmb/Ultra-FineWeb:en:19231@303`

**Context tail** (activation = last-token L20 state):
```text
...g our perceptions of modern Australian architecture. They comprise a diverse mosaic of the evolving urban environment and a rich resource documenting the progress of architecture of the last forty years.
Gollings has been the photographer of choice, the go-to guy for scores of architects wishing to have their works recorded. He is the consummate professional-for-hire who can be relied upon to capture the essence of a building, time and time again, to best reflect the architect’s design intent. Practical and pragmatic, one of his great skills lies in his ability to almost always compose the best shot, the one defining image that makes a building memorable, etches it into the psyche.
Beautiful
```

**preRL** (quote chars: 21):
```text
Australian architectural historian writing, exploring a photographer's book on Australian landscape architecture, with anecdotes about Riddell's gift of observing topography.

The sentence structure "But the photographs are also beautiful" signals a concluding clause or second item in a list likely continuing the critical theme of the book's qualities—likely "Images" or "His photographs are not only intellectually satisfying."

Final token "Beautiful" opens an incomplete noun phrase ("Beautiful"), part of a third-worded concluding observation ("Beautiful..."), likely followed by "photographs" or "this book is indeed a celebration of that" or "He constructs an aesthetic, but his photographs also embody technical precision and wonder."
```

**overlap100** (quote chars: 16):
```text
Australian newspaper/journalistic account of architecture historian's regional books, chronological register with anecdotes about Queensland landscape imagery's arcadian obsessions.

The paragraph header signals topic shift into quoted label/garland sentiment framing, suggesting a second item introducing the book's aesthetic qualities or photographic values alongside his academic substance—pattern implies a parallel descriptive adjective phrase or essay conclusion about form or image.

Final token "Beautiful" opens quoted noun phrase after "Articles... ", immediately expects " photographs are" or "ly rendered things certainly count here, but" or "this book's prose does not suffice," or "and data constructs aside" — continuing critique framing.
```

---

## Sample 8/16 — `openbmb/Ultra-FineWeb:en:26789@203`

**Context tail** (activation = last-token L20 state):
```text
Machinable Glass-ceramic vs. Lead-alkali-silica Glass
Both machinable glass-ceramic and lead-alkali-silica glass are glass and glass-ceramics. There are 10 material properties with values for both materials. Properties with values for just one material (16, in this case) are not shown.
For each property being compared, the top bar is machinable glass-ceramic and the bottom bar is lead-alkali-silica glass.
|Dielectric Constant (Relative Permittivity) At 1 Hz||6.0|
|Elastic (Young's, Tensile) Modulus, GPa||67|
|Electrical Dissipation At 1 Hz||0.0049|
|Electrical Dissipation At 1 MHz||0.0050|
|Electrical Resistivity Order of Magnitude, 10x Ω-m||14|
|Flex
```

**preRL** (quote chars: 20):
```text
Technical material property data format with a structured chemical product specification listing properties for Aluminum Alloy SA534, using standardized Material Properties tables with labeled columns.

The pattern "Tensile Strength" and "Modulus" items suggest another property column continues with "Flex" likely naming flexural modulus or another mechanical property category, completing the list of composite material attributes.

Final token "Flex" is the start of a truncated word mid-item ("Flex"), part of a property column listing sequence ("Flex..."), immediately expects "ural Modulus" or "ural Strength" or "ility Modulus" or "ural modulus is low" describing bending behavior or mechanical integrity at low temperature.
```

**overlap100** (quote chars: 20):
```text
Technical datasheet listing product material properties with categorized ASTM product codes format, pairing labels showing industrial ceramic data entries for Aluminum Iron Canyonite.

The pattern establishes two property categories header items begun ("Hardness, Thermal"), implying the second property name continues material mechanical characteristic or bending behavior descriptor, likely "ural Modulus" or "ural Strength" or similar bend or toughness metric entry.

Final token "Flex" opens a second listed property name mid-item, immediately expects "ural Modulus" or "ual Strength" or "ility Module" or "ural Crack" or "ure Density" or "Z Index falls under..." descriptor label describing shape.
```

---

## Sample 9/16 — `openbmb/Ultra-FineWeb:en:39563@164`

**Context tail** (activation = last-token L20 state):
```text
...rrelson was the best car sales person I have ever worked with. We went to the dealership on a Thursday evening.. Jamey took the time to listen, understand our needs and did not hesitate to let us look at alternatives and drive as many cars as we wanted. Ended up buying a certified used vehicle for a fare price and closed the deal after hours. Jamie asked us to come back on the weekend so he could explain and setup all the technology in the vehicle, which we took him up on. That was going beyond expectation. Plus we left with a full tank of gas. We had a good experience over all.
by Dott Butler from Quincy, FL
November 26, 2018
I really like my vehicle and I have never owed one that is in the
```

**preRL** (quote chars: 14):
```text
Auto forum post/feedback format with conversational responses discussing car payment/ownership costs and mechanic frustration.

The sentence "Now I have a car that is not like the others its always in the shop or there is nothing wrong with it" appears to be an informal personal anecdote or complaint about a low-quality car loaned to credit union being problematic, suggesting a third item or analogy.

Final token "the" is an incomplete noun phrase mid-sentence ("a car that is not in the"), part of a list of scenarios ("if a car that is not in the"), expecting a noun like "garage" or "top priority list" or "bad condition/debt traps."
```

**overlap100** (quote chars: 15):
```text
Local/autoamerica forum post blending informal tone with consumer feedback sections showing price comparisons/ucksiness anecdotes around sales cars.

The second quote entry appears to introduce a categorical commenter grievance statement or second-hand perception angle contrasting vehicle quality against dealership expectation, likely listing negative attributes or costly maintenance-free status — suggesting a phrase about rust bin or debt-damaged car pigeonhole.

Final token "the" is mid-phrase in a quoted quoted item clause describing "a car that consistently not in my finance logs isn't", expects noun like "bin" or "same category" or "problems" or "top ranked" or "garage weeds."
```

---

## Sample 10/16 — `openbmb/Ultra-FineWeb:en:34537@306`

**Context tail** (activation = last-token L20 state):
```text
... and take up less shelf space
- Specifically formulated: works in all Bissell and dirt devil compact deep cleaning machines
- formula contains biodegradable detergents; it contains no phosphates, dyes, optical brighteners or heavy metals; bottle made from at least 25-percent post consumer recycled plastic; both bottle and cap are recyclable
- DFE: formula recognized for safer chemistry by US epa's design for the environment (DFE) program
Specifications of BISSELL 2X Pet Stain & Odor Portable Machine Formula, 16 ounces, 74R71
|Product Dimensions||2.6 x 4.6 x 6.6 inches|
|Item Weight||1.2 pounds|
|Item model number||74R71|
Write a reviewYour Name:
Note: HTML is not translated!
Rating: Bad Good
```

**preRL** (quote chars: 18):
```text
Product description with promotional/SEO formatting, following e-commerce listing format with product details and review section.

The review feedback widget ("Rate this product: Poor Good") is a standard rating box with a star rating scale, typical of e-commerce sites, immediately after which "Your rating: Bad Good" is the default label text for the feedback form options.

Final token "Good Good" closes a conventional rating label ("Poor Good Bad Good"), part of a rating scale display format ("Rate this product: Poor Good"), immediately expecting continuation like " " or "Rating:" or "Rank:" or "Please rate this product" to complete the rating UI element.
```

**overlap100** (quote chars: 20):
```text
Product catalog listing format with Thai retail/marketing items showing specs and reviews, switching to a customer feedback/outlet section structure typical of e-commerce pages.

The quoted review rating widget pattern mirrors a standard feedback UI template with sentiment icons ("Rate this product"), displaying classic labels "Poor ExCEL" rating scale labels — the closing parenthetical label text is expected to follow convention.

Final token "Good Good" ends an informal rating scale description in braces ("Bad Bad Evil Good -- Evil Good"), immediately expecting continuation like " " or "Rating:" or " | Score" or " " or "Please" or "This," completing the emoticon legend.
```

---

## Sample 11/16 — `openbmb/Ultra-FineWeb:en:91880@202`

**Context tail** (activation = last-token L20 state):
```text
...o the page:
The initial comments that were coming in made us realize that people very much wished it was a co-op game so I got to work on adding that feature in immediately. I’m proud to announce that we now have co-op mode as well as the normal single player mode. The best part is that you can switch between the 2 on the fly so a friend can jump in at any time.
Another big issue that people had with the game was the fact that the climb animation didn’t look great. Our animator had actually never animated anything before this project and he decided to pick up Spine and set out to learn it. I’d say they all look great considering but I believe climb was just a difficult one to pull off in 2D.
```

**preRL** (quote chars: 13):
```text
Gamestreamer posts with casual tone and video thread style showing updates on a modded game's progress.

The sentence structure "Well for a while my models were honestly rough looking for the textures/anim's. I had originally planned on doing them separately but the new one came out much better than I expected. " establishes a personal confession mid-sentence promising the new fix to the animation issue.

Final token "issue." closes an aside clause ("Since originally I was worried about the awkwardness of movement with the models. Since then I had issues with the animation originally..."), strongly expecting "I've now fixed it" or "The new model definitely fixed the problem" or similar.
```

**overlap100** (quote chars: 11):
```text
American Style blog post format with casual/modder thread posts showing weekly updates for the channel's areas with humorous tone throughout.

The first quoted section header promises two update posts changed directions - the writer explicitly addresses a prior flaw/issue with original textures vs new ones, signaling the second quoted tweet explanation statement delivers the designer's improved or corrected artwork solution narrative about the problem concept.

Final token "manner. " closes a causal clause mid-sentence describing previous dissatisfaction rationale ("While I originally argued it was messy/confused aspect... Since technology changed my hate idea..."), strongly expecting "I've now drawn" or "The new model-fixed."
```

---

## Sample 12/16 — `openbmb/Ultra-FineWeb:en:33841@102`

**Context tail** (activation = last-token L20 state):
```text
Living with a pattern of procrastination can be difficult because it is often a silent, invisible, and shameful inner struggle. You may feel that there is a cloud of dread hanging over your head at all times. As the deadline looms, the sense of dread can grow until it becomes too much to bear. You may say to yourself, “This feels awful and I will never do it again!” Only to find yourself back in the cycle of procrastination the next time a deadline comes around.
Procrastination
```

**preRL** (quote chars: 12):
```text
Self-help/therapeutic blog post tone establishes educational momentum around the phenomenon of procrastination and overwhelm, promising insights into managing procrastination.

The opening sentence "Procrastination is a common mental habit and life difficulty." signals a pattern structure, now expected to continue defining or explaining procrastination, likely listing characteristics or consequences, or introducing therapy tips around the process.

Final token "Procrastination" repeats as the subject header of the paragraph ("Procrastination"), directly continuing the definition and scope section, strongly expects continuation such "is not just a problem, but" or "often stems from underlying emotional patterns" or "can be understood as a complex cycle of...".
```

**overlap100** (quote chars: 13):
```text
Blog post style with wellness/psychology framing establishes accessible tone and local context around anxiety and creative mindset themes.

The opening paragraph setup signals article introduction structure, promising heading promises about procrastination's nature and solutions, so the first paragraph header naturally follows with defining concept or key characteristics of procrastination, creating expected continuation about suffering cycle or therapy insights.

Final token "Procrastination" opens the quoted definition paragraph block already begun after intro sentence, immediately expecting continuation like "is a complex" or "can manifest as..." or "has many causes," or "is not just a problem" or "and recovery strategies address...".
```

---

## Sample 13/16 — `openbmb/Ultra-FineWeb:en:68813@337`

**Context tail** (activation = last-token L20 state):
```text
... Directory. High Quality Hot Water Boilers from Turkish Suppliers, Exporters and Manufacturer Companies in Turkey.
ProLine ® XE Combi Boiler. The first-ever residential combination boiler from A. O. Smith comes with an industry-leading 4.8 gal/min domestic hot water delivery and 95% AFUE, as well as a superior space heating performance with up to 74% more heating capa
Turkey Hot Water Boiler, Hot Water Boiler Turkish Hot Water Boiler Turkish Hot Water Boiler, Turkey Hot Water Boiler Manufacturers/Suppliers and Exporters Directory. High Quality Hot Water Boiler from Turkish Suppliers, Exporters and Manufacturer Comp
An electric water boiler, also called a thermo pot, is a consumer electronics
```

**preRL** (quote chars: 13):
```text
Wikipedia-style article listing appliance definitions with numbered subcategories, mixing topics around coffee maker and microwave, transitioning to a third definition about electric kettle.

The paragraph "A coffee kettle is an appliance that heats water to make tea or coffee; an electric kettle is an electronic home appliance" introduces a new definition, mid-sentence describing a small appliance device's function or product category.

Final token "electronics" ends mid-phrase ("a household electronics appliance is a home electronics"), part of a definition clause describing a food processor as consumer electronics product — expects continuation like "device used to heat water" or "product that combines heating elements with a power source" or "home appliance."
```

**overlap100** (quote chars: 12):
```text
Wikipedia-style article listing appliance variant articles with mixed category headers, alternating product descriptions covering coffee pot types; second entry shifts to a definition article topic.

The opening quoted definition paragraph introduces "toaster heater" as a household device article title with informal tone, signaling a second definition or category description phrase about appliance device or gadget function follows the electronics genre pattern.

Final token "electronics" ends mid-phrase describing manufacturer product class ("A dehumidifier is a household leisure appliance, a consumer consumable home appliance gadget"), immediately expecting noun continuation like "device that cooks" or "product item" or "device for displaying water".
```

---

## Sample 14/16 — `openbmb/Ultra-FineWeb:en:41145@298`

**Context tail** (activation = last-token L20 state):
```text
... that dye yarns, fiber, fabric, and carpets. Chemistry is important in all functions, but the technical content tends to be more challenging in the polymer chemistry side than in the dyeing and finishing end.
Textile chemists are generally persons interested in the intersection between chemistry and the kind of engineering that goes into textile marketing. They enjoy the modification and improvement of basic polymers and like using their knowledge of materials to solve problems.
Many chemists in the field are trained as polymer chemists. Although a number of schools specialize in textile chemistry, a textile degree is not a prerequisite for employment in the field. Important courses of study
```

**preRL** (quote chars: 16):
```text
Textbook reference work gives career descriptions and program outlines for mechanical engineering, with sections on scientific prerequisites for petroleum geologists.

The paragraph structure "The student should study premedical courses to prepare for petroleum engineering careers. Courses of study that are useful include" follows a list pattern, now transitioning to "Recommended areas of study," suggesting continuation listing academic subjects or prerequisites for geologists.

Final token "study" ends an incomplete clause ("Important courses of study"), part of a list of recommended academic subject areas ("Primary areas of study. Courses of study"), expecting continuation like "include" or "for this field are: physics, chemistry, mathematics" or "include those leading to..."
```

**overlap100** (quote chars: 16):
```text
Trade-reference directory article on chemical-engineering career field, with organized listing format covering careers and job areas under professional forestry theme.

The paragraph structure shifts from description paragraphs to a second source section header ("Academic preparation background"), signaling parallel advice listing academic disciplines or prerequisite subjects relevant to the field; expects similar enumerated curriculum items continuing the scholarly qualifications context about biology/materials.

Final token "study" closes a quoted subject-description clause phrase ("Relevant areas of key areas of emphasis"), immediately expecting "include" or "for this area are" or "include biology, physics" or "would be include..." or "include potential goals."
```

---

## Sample 15/16 — `openbmb/Ultra-FineWeb:en:96583@215`

**Context tail** (activation = last-token L20 state):
```text
...gas only expands at the speed of sound. Grant that the speed of sound in white hot gas at enormous pressure is much higher than the speed of sound in air, but muzzle velocity in conventional guns is still limited to about 4000 ft/sec.
Railguns have no such limit and can achieve much much higher muzzle velocities. The Navy project is working on 32 Megajoule rail guns. To put that into real numbers, 32 Megajoule is 23.6 Mega foot/pounds. For comparison, the conventional 5 inch cannon has a muzzle energy of 0.185 Mega foot/pounds. The rail gun has 127 times the muzzle energy of the conventional cannon. This translates into amazing range, like 100 miles.
To power the railgun, the Navy is talking
```

**preRL** (quote chars: 16):
```text
Old SF Blog tech/technical article tone with amateur space enthusiast writer discussing fusion reactors and a proposed lunar colony setup, building toward describing scaling issues.

The sentence structure "Now the company doing the heavy lifting is talking" signals a list of proposed power figures, specifically a specific technology or budget figure, continuing the "Big Death Ray" reference to nuclear fusion concepts.

Final token "talking" ends mid-clause ("NASA is talking"), part of a list of technologies and figures being considered ("Tesla is talking...IBM is talking"), strongly expecting continuation like "about megawatt lasers" or "of huge fusion reactors and perhaps superconducting materials" or "about giant..."
```

**overlap100** (quote chars: 12):
```text
Old Blogger-era tech/technical sci-fi column tone covering anomalous Cold War era power hardware. Narrative offers parallel catalogs of escalating proposals around fusion power depletion scenario.

The opening paragraph establishes pattern introducing quoted sources and figures, implying next item is technical spec or project reference material about massive energy numbers or hardware approaches — likely turbine or budget talk terms.

Final token "talking" begins quoted description clause mid-sentence after "NSA crewNS are NS folks, Dixon crew are proposing", immediately expecting "about megatons" or "of carbon fiber" or "about speculative numbers" or "about catalysts and plans."
```

---

## Sample 16/16 — `openbmb/Ultra-FineWeb:en:39779@99`

**Context tail** (activation = last-token L20 state):
```text
Creamy and sweet—and just what your skin needs to feel loved from head to toe. Our best body lotion for normal to dry skin combines Milk, Honey and Vitamin E, along with Coconut and Grape Seed Oils for an ideal blend of nutrients. Designed to soften your skin, this delectable natural combination is clinically proven to moisturize for up to 24 hours. And if that’s not enough, this non-greasy moisturizer absorbs quickly so you can sit back and enjoy
```

**preRL** (quote chars: 12):
```text
Product messaging format with bold branding and health-yogurt tone, promoting a sprayable sunscreen with "refreshing flavor," implying a campaign or product promotion.

The sentence structure "But bye-bye sweat — and let the thirst-quenching mist dry up so you can savor your day...and enjoy" suggests a concluding clause implying the sunscreen has completed its job, then a happy outcome.

Final token "enjoy" ends an incomplete phrase ("and savor your day and enjoy...and enjoy"), part of a contrastive or humorous concluding clause ("But before the moisture clears, take a break from drying and enjoy"), likely followed by "your pristine beauty" or similar.
```

**overlap100** (quote chars: 13):
```text
Marketing-style product copy withWM beauty tone blends humor and chemistry specs — promotional listing format with bold taglines around flavor-focusedloe accolades.

The sentence structure promises a closing paired sentiment clause contrasting "denahrain's rinse action timing" versus satisfying experience/moment — implying the promised leisure or relief outcome, and a concluding image of finished treats being enjoyed or forgotten.

Final token "enjoy" ends a quoted phrase describing aftermath consequence ("Once cleaned and cleared — ditch those worry signals...and step aside to pause, then savor"), strongly expecting "your results" or "the serene surroundings" or "your newfound silence" or similar.
```
