# Quote-penalty RL: sample comparison

16 held-out `av_eval` prompts (distinct docs), identical prompts + per-sample seeds across checkpoints, 200 max new tokens.

| checkpoint | quote chars/sample | zero-quote samples | 'final token' echo | CJK |
|---|---|---|---|---|
| preRL | 15.25 | 0/16 | 16/16 | 0/16 |
| iter50 | 0.06 | 15/16 | 15/16 | 2/16 |


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

**iter50** (quote chars: 0):
```text
Traditional country music blog post format with Texas entertainment news mix, moving into some local celebrity endorsements and a bit of quirky pride around a singer and songwriter.

The sentence structure suggests a funny contradiction about the artist not only sells some of her own music but also allegedly composes some of — an incomplete clause about her musical talent and creative writing skills.

Final token some of is mid-clause in an awkward attribution claim suggesting she not only composes some of but actually some of the music herself — a parenthetical digression likely continues about her own lyrical or artistic ability overlapping-writing activity.
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

**iter50** (quote chars: 0):
```text
New Jersey county advocacy group website promoting a proposed green corridor project in East Bay region, shifting to a citizen-driven organization focused on sustainability and land preservation.

The block quote introduces a community-driven initiative, likely describing a coalition of local residents primarily of (a description of a coalition or community members — likely community advocates).

Final token of a primarily composed of mainly of is mid-sentence introducing a coalition consisting primarily of, likely a description of a diverse group of individuals or community members with direct experience and opposition to the pipeline, suggesting a local civic or activist group formation narrative follows.
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

**iter50** (quote chars: 0):
```text
Product catalog webpage format with technical electronics supplier site. Switch to a new product line suggests a product announcement or sales listing with a specific model for high speed BNC connectors.

An incomplete sentence introduction implies a product line of ANB Scientific Multichannel Acquisition modules are introduced as the Blackfin 2520 Miniature隔离 adapters are

Final token are are a pair of ADP2520 TwinLink Thunderbolt Connectors are — expects a technical description or product benefit statement about these specific high speed switching modules solving a particular application need.
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

**iter50** (quote chars: 1):
```text
U.S. transit product announcement format with a contractor award for a solar-powered light rail project, establishing technical and construction details around a new bus fleet procurement contract.

The company structure includes a statement about TriMet Transit selecting the project and project is moving forward with funding to support expansion of its transit infrastructure in Lincoln County, establishing a partnership context.

Final token Transit Transit Transit is mid-sentence introducing TriMet Transit, expected to continue with a statement about TriMet Transit's decision-making and the company has selected this project as part of its sustainable infrastructure expansion plans, likely repeating partnership or funding details.
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

**iter50** (quote chars: 0):
```text
College student blog format with ecological/educational focused content listing species and restoration benefits of fire in Michigan forests.

The quoted scientific sentence is mid clause describing additional benefits of prescribed fire clearing brush and bacteria thereby reducing soil moisture and thereby — an incomplete ecological process describing soil aeration and removing fire fuel thereby suggesting a second environmental benefit or nutrient cycling effect.

Final token thereby thereby is mid clause describing removing excess moisture and reducing soil moisture thereby — a parenthetical scientific explanation of soil disturbances and fire clearing contributing to health cycles or nutrient availability, likely completing a multi-function environmental benefit explanation.
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

**iter50** (quote chars: 0):
```text
Arabic/Egyptian product description format with perfume product image suggesting a blend for Men and Women or unisex fragrance for Summer season.

The structure implies a product title or label likely introducing a combined fragrance for Men and Women or Men & Men, suggesting a product description or combo offering for a dual gender aroma.

Final token Men and Men is mid-sentence in a product name or tagline context for香料 for Men and Women and Men, likely a product name or dual gender appeal expecting the combined fragrances qualities expected for a specific masculine or unisex fashion fragrance.
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

**iter50** (quote chars: 0):
```text
Australian architectural historian text on Australian landscape photography, a regional book about nature and design — suggests a nuanced argument about the work of a photographer producing excellent images.

The phrase appears to pivot to an additional quality — Beautiful suggests a concluding remark or a qualification likely about the images themselves becoming beautiful or artistic quality beyond purely technical excellence.

Final token Beautiful is the start of an incomplete word or concept, likely a noun phrase continuing the theme of beautiful — expect a continuation addressing the photographs themselves as a complement or further attribute to this process of capturing the natural world, likely a reflective or analytical observation about aesthetic quality.
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

**iter50** (quote chars: 0):
```text
Technical material property database format with a chemical element identifier product page, suggesting property values for Ceramic / Glass Material properties in a comma separated list format.

The pattern suggests a second property category, likely a mechanical property or Flexibility measurement, implying a second property value or mechanical strength number follows.

Flex is the abbreviated term beginning a second property label, likely a symbol for Flexural modulus or similar mechanical property describing bending behavior under strain or flexibility factor, continuing a paired or combined property listing structure.
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

**iter50** (quote chars: 0):
```text
American automotive forum post format with multiple questions and responses about car payments and maintenance costs. A casual comment suggests a frustrated attitude towards a vehicle that is not in the repair cycle.

The quoted text appears to be a odd comparison or personal opinion about a vehicle that is in the awkward position not being in the — likely a humorous or negative remark about a financed vehicle that has frequent problems.

Final token the is mid-clause of an incomplete quote about a vehicle that is not in the, suggesting a comparison or condition about a vehicle that is in frequent repair or not a good value versus depreciated debt.
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

**iter50** (quote chars: 0):
```text
Product description format with Thai herbal food store page switches to a product review form with rating option, following a standard e-commerce template pattern.

The rating widget HTML format with a default evaluation box appears to include a star rating scale with a feedback label that translates to Bad Good, signaling a typical rating evaluation prompt with a numeric score range.

Final token Good Good is mid-value between Bad and Good in a rating gauge conversion label format, part of a standard Magento or similar rating star UI template with a feedback submission field — the text likely completes a score rating format with emotional indicators displayed.
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

**iter50** (quote chars: 0):
```text
American gaming blog post format with a couple announcements about the site and some new changes. Seems to be mixing some updates and explaining a transition in a model that was previously rough looking.

The sentence structure is mid explanation of how I originally had issues with the texture issues and while I originally thought the new style wouldnt work better my effort fixed the issue. suggests a new artist fix.

Final token. is mid clause after admitting that while I did originally have some issues with the original distortions and since wanted to fix the strange feel but the new one was a bit more problematic. implying a second attempt or fix explanation follows.
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

**iter50** (quote chars: 0):
```text
Canadian mental health blog format establishes a helpful, empowering tone around procrastination and anxiety around procrastination, signaling an article about understanding and breaking patterns.

The opening sentence structure implies the text will introduce what procrastination is, strongly suggesting a repetition or explanation follows about the nature or cycle of procrastination as a problem.

Final token Procrastination is repeated, signaling the main topic header or article introduction, establishing that while procrastination is a common and difficult experience, the text will explore connections and strategies around procrastination — likely continuing a definition or explanation of how it relates or fuels mental suffering.
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

**iter50** (quote chars: 0):
```text
Wikipedia article format with multiple definitions and subtopics, transitioning to a separate section on coffee cookers or kitchen appliance products, a definition or description is provided.

The inserted definition is an informal term describing an electric kettle as an household electronics device, a definition of a heating appliance that is a consumer electronics product is attempted, likely a casual definition.

Final token electronics is mid-sentence describing a kitchen appliance is a consumer electronics electronics — a technical product that is a household appliance device or an electronic appliances device, likely a definition or description of a heating method converting liquid water into steam as a household appliance product.
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

**iter50** (quote chars: 0):
```text
Technical reference handbook on chemical engineering career; American agricultural university text presents an overview of careers and prerequisites for oceanography; a section follows on appropriate college major study.

The paragraph shift suggests a list or recommendations of related areas of study may follow, with a statement about important areas of study suggesting courses of study are related to prerequisites.

Final token study is mid-sentence in a statement about important areas of study to prepare for the field. Areas of major study study... A list of relevant areas of study or recommended areas of study may follow, likely completing a scholarly preparation context.
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

**iter50** (quote chars: 0):
```text
American tech war blog tone with sci-fi space nuclear power talk suggesting big numbers and serious fusion solutions for a subsurface super shield project.

The sentence structure implies they are starting to talk about the technology schemes they are talking, suggesting a specific technology or figures are being considered alongside some massive power generation talk numbers.

Final token talking is mid-clause with the military folks talking, they are talking — a likely parenthetical reference to some specific technology or plans involving huge power numbers or specific exotic tech proposals. Expect a technical specification or proposed budget figures in conjunction with radiation and mega watt level talk.
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

**iter50** (quote chars: 0):
```text
Product description format with bold fragrance promises for a sunblock serum — playful tone suggests a fun twist on moisture locking while letting you savor your drink and enjoy.

The final sentence structure suggests a clever dual meaning or closing encouragement to rinse off the lingering benefits and then enjoy — implying a pun or time to savor and enjoy something else after the blocking process.

Final token enjoy enjoy is mid-sentence in a playfully contrasting clause, suggesting a closing punctuation or the pleasure of letting the sunscreen work its magic while you enjoy and enjoy — likely completing a humorous resolution or enjoyable activities following the blocked time.
```
