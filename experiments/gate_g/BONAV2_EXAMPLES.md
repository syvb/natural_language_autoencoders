# Gate G BoN-AV2 — AV2 vs frozen AV, full examples

BoN-AV2 round (gap 16→24) **result: double-null.** Reconstruction 0.8272 vs frozen 0.8273 (p=0.57); diff-focus unchanged (length 103.4 vs 103.5; overlap with the before-text Jaccard 0.271 = 0.271). Below, the frozen AV(h24) description and the BoN-SFT'd **AV2(h24)** for the same positions — they are the same *kind* of full-state description, just paraphrased with different confabulated specifics. AV2 did **not** learn to drop redundant before-content and name the change.

Each shows the layer-24 (after) state described by the frozen AV vs AV2, plus the layer-16 (before) description for reference.


======================================================================

## i=0  ·  token_at_pos = "'t"

**context:** …|Viewing Single Post From: Spoilers for the Week of February 11th|
|Lil||Feb 1 2013, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't


**AV(h16) [before, for reference]:**

> Canadian TV show forum post tone with sarcastic/poetic language ("Don't like this show. Don't like the new season."), suggesting a personal opinion or reaction format.

The phrase "Don't watch this show. Don't care about the plot. Don't" repeats a pattern, implying a second parallel negative declaration or list of unrelated things, likely continuing a dismissive stance.

Final token "Don't" is mid-repeated phrase ("Don't like this show. Don't care about characters. Don't"), strongly expecting a noun or verb continuation like "care about ratings" or "want to remember" or "need this" or "have any interest in the book."


**frozen AV(h24) [after]:**

> Fanfic/forums post format with informal tone and character reference questions about TV show "SG1" — pattern establishes a reader's frustration with the show.

The sentence structure "I love the characters, don't care about the plot. Don't like the DCU, don't" mirrors a second question implying the reader is ignoring or dismissing the book's facts about the universe.

Final token "Don't" repeats the second clause ("If you don't care about the characters. Don't..."), strongly expecting "care about" or "want to know about" or "even care about the wiki" or "need to know anything else about the characters" — continuing the negative stance.


**AV2(h24) [BoN-SFT'd, after]:**

> Fanfic/forums post format with casual tone discussing TV show characters, establishing a pattern of complaints about the show's universe.

The phrase "I love Superman. Don't care about plot, don't care about his backstory. Don't" mirrors a second question structure, implying a second reader's attitude toward the show's characters, likely continuing the "not interested in wiki" sentiment.

Final token "Don't" is mid-sentence in a parallel list ("If you know about characters, don't care about them. Don't..."), strongly expecting "care about the series" or "want to know anything about the wiki" or "care about the characters' details."


======================================================================

## i=1  ·  token_at_pos = ' hoping'

**context:** …ing Single Post From: Spoilers for the Week of February 11th|
|Lil||Feb 1 2013, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really, but hoping


**AV(h16) [before, for reference]:**

> Fanfic forum post tone with casual, informal register ("Watching the series... hoping") suggesting a list or mixed feelings about characters.

The phrase "Not sure about the other one (Sara's pregnancy) but hoping" implies a second item or incomplete sentiment, likely continuing a comma-separated list of desired outcomes or uncertain status for a character.

Final token "hoping" ends an incomplete clause ("But hoping... but hoping"), part of a casual qualifier structure ("Not sure about this one but... and hoping"), strongly expecting a noun phrase like "for a happy ending" or "they don't die" or "that the other series improves."


**frozen AV(h24) [after]:**

> TV fan forum post tone with casual/angsty attitude about Kelsie's pregnancy status, implying a Soap Opera discussion thread.

The phrase "So I'm seeing Kelsie disappear from the blogs again... hoping" suggests a list or mixed feelings about certain characters' fates, implying a specific character's storyline or fate is being referenced.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a parenthetical clause structure ("Either one or the other, but not both... hoping"), strongly expecting a noun phrase like "she'll resolve" or "for her to stay away" or "Sarah's storyline resolves."


**AV2(h24) [BoN-SFT'd, after]:**

> TV fan forum post tone with casual humor about Kate's pregnancy status, implying a blog or fan site update about the WWE Divas.

The phrase "So I'm seeing both Kaitlyn and Tori disappear from the show this week, hoping" suggests a list or specific scenario, likely naming a character's fate or storyline goal, implying a contradictory or uncertain wish.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a list of possibilities ("Either one of them is pregnant, but not mine, so hoping"), strongly expecting a noun phrase like "she'll resolve" or "for Sarah's storyline" or "they aren't completely gone."


======================================================================

## i=2  ·  token_at_pos = '"'

**context:** …13, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really, but hoping that we get some good "SAMANTHA GENE!!" Marlena Death-Stares out of it. And "newfound"


**AV(h16) [before, for reference]:**

> Fanfic forum post tone with quoted dialogue and sarcastic commentary about a TV show character's behavior, implying crude humor and mocking.

The phrase "And the whole 'recovered' and "newly" suggests a list of quoted phrases or words from the story, likely continuing a sarcastic accusation about the character's false promises or "recovered friendship" theme.

Final token "newly" ends mid-phrase (" "Re-found" and "fancied""), part of a quoted phrase sequence listing questionable descriptors ("And "re-found""), strongly expecting a noun like "friendship" or "fake discoveries" or "words like integrity" completing the ironic construction.


**frozen AV(h24) [after]:**

> TV show forum post tone with fan speculation about Brooke's pregnancy storyline and character dynamics, implying ongoing soap opera dialogue.

The phrase "And the whole 'hating on Jack' thing...and her newfound " is a list of conflicting emotions/plot points, suggesting a second phrase about the other women's reconciliation or newfound family bonds, likely referencing the pregnancy or past relationships.

Final token "newfound" is mid-phrase in "or "new-found" — part of a quoted list ("with 'new found'"), strongly expecting a noun like "love for Sam" or "family loyalty" or "friendship with the other women" or "reconciliation with her past."


**AV2(h24) [BoN-SFT'd, after]:**

> TV show forum post tone with fan fiction context about Kelly/Bo's relationship drama, implying soap opera dialogue and character dynamics.

The phrase "Like she's having second thoughts about leaving Scott AND new "found" " suggests a list or contradiction of past emotions, likely referencing the pregnancy storyline or other women's reconciliation with the family, implying a contradiction or positive development.

Final token "new-found" is mid-phrase in "with 'found' " emotions," part of an incomplete clause listing character goals ("her newfound reconciliation and/or "new-found"), strongly expecting "friendship with Sarah" or "love for the family" or "loyalty to the pregnancy" or similar.


======================================================================

## i=135  ·  token_at_pos = 'ski'

**context:** … in E Major (BWV 1042), and the Concerto for Two Violins in D Minor (BWV 1043) — reproduced from an authoritative edition.
Caprices and Etudes for Solo Violin by Nicoḷ Paganini, Henryk Wieniawski


**AV(h16) [before, for reference]:**

> Musical reference book catalog format with numbered sections and historical/technical descriptions, now transitioning to a separate listing of piano repertoire titles.

The phrase "Two Famous Violin Concertos by Paganini, Tchaikovsky, Wieniawski" suggests a music CD or sheet music collection title, likely a classical virtuoso piece compilation with composers named.

Final token "Wieniawski" ends mid-list ("Niccolò Paganini, Antonín Dvořák, Franz Liszt, Tchaikowskii"), strongly expecting continuation like "and other composers" or "and piano solos" or " [composer names]" completing the list of composers.


**frozen AV(h24) [after]:**

> Classic reference book format with authoritative music publishing description, featuring beautiful illustrated edition of Mendelssohn's greatest violin works.

The phrase "Two legendary violin concertos—Brahms's 12 Sonatas, 24 Caprices: The Essential Mendelssohn" mirrors a book title listing composers and works, now naming a second famous collection of challenging pieces.

Final token "Waxisiakski" ends mid-list ("Paganini, Spohr: Mendelssohn, Spohr's Brilliant Virtuoso Works"), strongly expecting "Two masterpieces" or "The complete works of two great violin repertoire" or similar closing phrase naming the pieces.


**AV2(h24) [BoN-SFT'd, after]:**

> Classic music-book format with authoritative American library edition, presenting a comprehensive series of piano masterpieces by Brahms.

The second item "Two Great Violin Classics: Mendelssohn's 24 Violin Sonatas, Newman's Essential Works" mirrors a book title listing, with "Two giants of the repertoire: Paganini, Wieniawski" suggesting a third element about their works.

Final token "Wieniawski" ends mid-list ("Paganini, Spiccato: Mendelssohn, Wieniawski"), strongly expecting "Two Brilliant Concertos" or "The finest collections of virtuoso pieces" completing the description.


======================================================================

## i=684  ·  token_at_pos = '2'

**context:** …uch money the company actually made? Brown is doing what he gets paid to do; err on the side of caution. This is a short sellers manipulation, I wouldn't be surprised if the stock recovers at least $2


**AV(h16) [before, for reference]:**

> Financial stock discussion format with a trader's tweet tone suggesting negative sentiment about a stock price, implying speculative or speculative trading context.

The phrase "I see a lot of negative commentary about the stock price today...and I'm not sure if this company is going to survive" implies an incomplete question or claim about a specific company's performance or financial situation, likely referencing the previously mentioned stock.

Final token "2" is mid-number in "stock $2" — part of an incomplete numeric expression ("Whether or not the stock price drops $2"), strongly expecting a continuation like "0%" or "5's stock" or "of the scandal" or "week's poor results."


**frozen AV(h24) [after]:**

> Stock trading/technical discussion post tone with investor sentiment questions and market commentary format. The post is listing factors around stock price and earnings.

The phrase "With crude oil falling ~$5-6 today...if the stock drops another $2" suggests a speculative statement about stock price recovery or share value, implying a numerical estimate or price target for the company's recent decline.

Final token "2" ends an incomplete numeric expression ("up $2"), part of a price speculation clause ("Would it mean $2...might see stock up $2"), strongly expecting "per share" or "0 in the next few weeks" or "a premium" to complete the dollar value context.


**AV2(h24) [BoN-SFT'd, after]:**

> Stock trading/technical discussion post tone with investor sentiment questions about market momentum and earnings, suggesting informal trading forum context.

The phrase "With the market dropping ~$5 a share its possible the stock could double back to $2" implies a numerical estimate or speculation about stock price movement or valuation impact, likely completing a second figure or range about the company's recent fund value.

Final token "2" is mid-number in an incomplete clause ("could possibly $2"), part of a speculative statement ("If earnings drop $50+ per share... expect it might lose $2"), strongly expecting "per share" or "0 in the next few days" or "more in value."
