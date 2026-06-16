# Gate G BoN-AV2 — full worked examples

Snapshot during the BoN-AV2 round (gap 16→24). For each position the AV is shown a single activation vector and writes a description; it captures register/tone but confabulates specific entities (it never sees the text).

Fields:
- **context** — the real left-context (for your reference; the AV never sees it)
- **AV(h16) [before]** — frozen AV's description of the layer-16 state
- **AV(h24) [after]** — frozen AV's description of the layer-24 state (the target)
- **8 BoN candidates** — samples from the frozen AV (temp 0.8) of h24
- **selected best** — the candidate the two-text AR2 scored highest for reconstructing h24 *given the before-text* (i.e. highest diff value)

> These candidates are pre-SFT (sampled from the *frozen* AV). The diff-focused **AV2** descriptions — the actual deliverable — are generated after the SFT step and will be added when the run finishes.


======================================================================

## position i=0  ·  token_at_pos = "'t"

**context:** …|Viewing Single Post From: Spoilers for the Week of February 11th|
|Lil||Feb 1 2013, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't


**AV(h16) [before]** (96 words):

> Canadian TV show forum post tone with sarcastic/poetic language ("Don't like this show. Don't like the new season."), suggesting a personal opinion or reaction format.

The phrase "Don't watch this show. Don't care about the plot. Don't" repeats a pattern, implying a second parallel negative declaration or list of unrelated things, likely continuing a dismissive stance.

Final token "Don't" is mid-repeated phrase ("Don't like this show. Don't care about characters. Don't"), strongly expecting a noun or verb continuation like "care about ratings" or "want to remember" or "need this" or "have any interest in the book."


**AV(h24) [after / target]** (101 words):

> Fanfic/forums post format with informal tone and character reference questions about TV show "SG1" — pattern establishes a reader's frustration with the show.

The sentence structure "I love the characters, don't care about the plot. Don't like the DCU, don't" mirrors a second question implying the reader is ignoring or dismissing the book's facts about the universe.

Final token "Don't" repeats the second clause ("If you don't care about the characters. Don't..."), strongly expecting "care about" or "want to know about" or "even care about the wiki" or "need to know anything else about the characters" — continuing the negative stance.


**selected best-of-8** (reward 0.8779 vs batch mean 0.8752):

> Fandom/forums post tone with informal humor ("I like the Doctor... Don't understand the show"), implying a character quiz or opinion thread about the TARDIS universe.

The pattern "I love the characters, don't care about backstory. Don't like them... " establishes a second clause parallel structure suggesting another viewer's ignorance or disinterest, continuing the complaint about canonical facts.

Final token "Don't" mirrors the opening question's repeated "If you don't care about plot, don't care about characters... Don't" structure, strongly expecting "care about the show" or "want to know anything about the characters" or "care about the WHO," completing the dismissal.


<details><summary>all 8 candidates</summary>


**[0]** Fanfic/forums post format with informal tone discussing TV show characters, establishing a pattern of complaints about the DC Comics universe.

The phrase "I love watching Superman... Don't care about his plotlines. Don't" mirrors a prior question structure ("Since I'm a sci-fi fan, don't care about DC characters..."), strongly implying another negative attitude about the wiki or character details.

Final token "Don't" is mid-sentence in the second clause "If you don't care about the wiki. Don't care about them... don't," continuing the parallel list of disinterest, expecting "care about the series" or "want to know his backstory" or similar.


**[1]** Fan fiction forum post format with informal tone, using a "TV show fan" tag to complain about the Doctor, implying an ongoing list of character attributes or facts.

The sentence structure "So I love the Heroes... don't care about plot, don't read it... Don't care about SG1. " mirrors a second similar claim, strongly implying the next tokens continue a dismissive attitude toward the show.

Final token "Don't" repeats the pattern "If you don't care about the universe, don't" — expects "care about the characters" or "want to know about the wiki" or "need to read/like the books" completing the contrasting clause.


**[2]** Fandom/forums post tone with informal humor ("I like the Doctor... Don't understand the show"), implying a character quiz or opinion thread about the TARDIS universe.

The pattern "I love the characters, don't care about backstory. Don't like them... " establishes a second clause parallel structure suggesting another viewer's ignorance or disinterest, continuing the complaint about canonical facts.

Final token "Don't" mirrors the opening question's repeated "If you don't care about plot, don't care about characters... Don't" structure, strongly expecting "care about the show" or "want to know anything about the characters" or "care about the WHO," completing the dismissal.


**[3]** Bulleted forum post tone with informal/insane humor about a TV show character (Buffy) showing dislike of books — pattern establishes reader seeking info about canon.

The phrase "I love the SG1 universe. Don't care about the characters, don't" mirrors an earlier clause structure, strongly implying a second question or dismissive attitude about the show's fans or characters.

Final token "Don't" repeats the pattern "If you don't care about plot, don't know much about them. Don't" — immediately expects "care about the wiki" or "want to know anything about the universe" or "even care about the numbers," completing the negative attitude claim.


**[4]** Female blogger/forum post tone with "Doctor Who" fan humor, establishing a pattern of mocking character's lack of knowledge about the show's universe.

The phrase "I hate the SG1 universe. Don't care about the characters, don't" mirrors a second question structure, implying a second common fan complaint about the book series, continuing the "not interested in facts" attitude.

Final token "Don't" repeats the negation pattern ("Don't care about the series. Don't..."), strongly expecting "care about" or "want to read the series" or "even care about the characters" or "need to remember them," completing the contrasting attitude toward the show's canon.


**[5]** Fanfic forum post tone with informal humor about TV shows, establishing a pattern of criticizing Star Trek characters and their canonical details.

The phrase "I love reading about this show. Don't care about sciency stuff... don't even like the characters. Don't" mirrors a question format with "Have you read the wiki about characters?," implying a second similar negative sentiment or dismissal about the fandom.

Final token "Don't" repeats the negative clause structure "If you don't know anything about the show, don't..." — strongly expects "care about" or "want to know about the universe" or "need the wiki" or "care about the other attributes."


**[6]** Fanfic/forums post format with informal tone using "TV show" to complain about plot details about Harry Potter's DC character.

The sentence structure "I love the Heroes show... Don't care about the sciency stuff. Don't like the characters, so don't follow them" establishes a pattern implying a second question about the book's fandom opinion or character preferences, continuing the "Not a fan" context.

Final token "Don't" mirrors the opening "If you don't know the backstory. Don't" pattern, strongly expecting "care about the books" or "want to know about the universe" or "need to know anything about them."


**[7]** Fanfiction forum post format with casual/irritated tone about TV show "Doctor Who" characters, establishing a pattern of mocking trivia questions about the series.

The phrase "I like the Doctor... Don't care about backstory. Don't like the characters. Don't" mirrors a previous clause structure suggesting the poster is dismissing the book series before reading it, implying another question or attitude about the show.

Final token "Don't" repeats the second negative clause ("If you don't know about the universe, don't..."), strongly expecting "care about the wiki" or "want to know about the characters" or "care about the canon" continuing the dismissive list.


</details>


======================================================================

## position i=1  ·  token_at_pos = ' hoping'

**context:** …ing Single Post From: Spoilers for the Week of February 11th|
|Lil||Feb 1 2013, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really, but hoping


**AV(h16) [before]** (98 words):

> Fanfic forum post tone with casual, informal register ("Watching the series... hoping") suggesting a list or mixed feelings about characters.

The phrase "Not sure about the other one (Sara's pregnancy) but hoping" implies a second item or incomplete sentiment, likely continuing a comma-separated list of desired outcomes or uncertain status for a character.

Final token "hoping" ends an incomplete clause ("But hoping... but hoping"), part of a casual qualifier structure ("Not sure about this one but... and hoping"), strongly expecting a noun phrase like "for a happy ending" or "they don't die" or "that the other series improves."


**AV(h24) [after / target]** (93 words):

> TV fan forum post tone with casual/angsty attitude about Kelsie's pregnancy status, implying a Soap Opera discussion thread.

The phrase "So I'm seeing Kelsie disappear from the blogs again... hoping" suggests a list or mixed feelings about certain characters' fates, implying a specific character's storyline or fate is being referenced.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a parenthetical clause structure ("Either one or the other, but not both... hoping"), strongly expecting a noun phrase like "she'll resolve" or "for her to stay away" or "Sarah's storyline resolves."


**selected best-of-8** (reward 0.8440 vs batch mean 0.8403):

> TV fan blog tone with casual/abbreviated dialogue format, referencing Scott's pregnant state and rumors about Tori and Gabby.

The phrase "So I'm waiting to see if Kaylor's pregnant or not... hoping" suggests a list or specific character scenario, implying a conflicting fan wish or soap opera storyline detail about Kelsi's fate.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a speculation list ("So either one of them is gone/active this week, but not both... hoping"), strongly expecting a specific character name or outcome like "she'll resolve her issues" or "for Taylor's storyline."


<details><summary>all 8 candidates</summary>


**[0]** TV show forum post tone with fan speculation about Bethany's pregnancy status, implying casual gossip and fan reaction posts about SOAP characters.

The phrase "So I'm seeing her being eliminated from the hospital this week... hoping" suggests a list or dual hope scenario, likely naming specific storyline goals or character fates, with "Carrie" implied as a separate hopeful subject.

Final token "hoping" ends an incomplete clause ("but, hoping... hoping"), part of a speculative list structure ("either one or the other, but not sure... hoping"), strongly expecting a noun phrase like "she'll resolve" or "for Kate's storyline" or "they'll stay away from."


**[1]** TV soap blog tone with fan forum post format, discussing Kelly's pregnancy status and other characters' fates.

The phrase "So I'm confused about Katarina being absent on the show today, but hoping" implies a list or mix of fan speculation items, likely naming a specific character's storyline or fate, suggesting a contradictory or hopeful sentiment about Becky's pregnancy.

Final token "hoping" ends an incomplete clause ("but, hoping... hoping"), part of a specific character list ("either one of them, though not sure, hoping"), strongly expecting a noun phrase like "she'll resolve" or "for her to finally disappear" or "both characters' storyline."


**[2]** TV fansite/post format with informal tone discussing Soaps actresses and their fates, implying a list or update about Kate's pregnancy status.

The phrase "So I'm kinda hoping for some Gabby/Rebecca death stuff this week... aside from wanting her gone, hoping" suggests a specific character scenario or fan speculation about another character's storyline direction.

Final token "hoping" ends mid-clause ("but, hoping... hoping"), part of an incomplete list or clause structure ("So far the rumor is pregnant with Mel, but not sure... hoping"), strongly expecting a noun like "she'll resolve" or "for her to finally disappear" or "the others' fate."


**[3]** TV mom blogger tone with fan forum post discussing Shield's pregnancy status and character drama, implying informal speculative or hopeful comments about someone's fate.

The phrase "So I'm not sure if I'm seeing Kara on her own show anytime soon, but hoping" suggests a list or contradictory status about two pregnant characters, implying a specific character's storyline or resolution.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a parenthetical clause listing desires ("So far I'm not sure about Summer's pregnancy, but hoping"), strongly expecting a noun like "she'll get resolved" or "for Mel's storyline" or "it's not her."


**[4]** TV fan blog tone with casual/abbreviated dialogue format, referencing Scott's pregnant state and rumors about Tori and Gabby.

The phrase "So I'm waiting to see if Kaylor's pregnant or not... hoping" suggests a list or specific character scenario, implying a conflicting fan wish or soap opera storyline detail about Kelsi's fate.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a speculation list ("So either one of them is gone/active this week, but not both... hoping"), strongly expecting a specific character name or outcome like "she'll resolve her issues" or "for Taylor's storyline."


**[5]** TV fandom/BBW blog tone with casual forum post format, implying fan speculation about Charlene's pregnancy status or character futures.

The phrase "So I'm watching BOTH Kelly and Alex's baby pregnancy... hoping" suggests a list or specific expectation, likely naming a character's fate or another soap story element, implying the writer is confused about the timing or outcome.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of an uncertainty qualifier statement ("So far it's either one of them, but not confirming, so hoping"), strongly expecting a noun phrase like "she'll resolve" or "for her to finally disappear" or "the other's storyline."


**[6]** TV show fan forum post tone with casual humor about Beth's pregnancy status, implying a gossip or speculation thread about the two OCs.

The phrase "So I'm seeing both Kelly and Sam being removed from the roster this week... hoping" suggests a list or specific wish, likely naming a character's storyline fate or character outcome, implying the author's personal preference or speculation about another character.

Final token "hoping" ends an incomplete clause ("but, hoping... hoping"), part of a conjunction structure listing character statuses ("Either one of them, not sure but hoping"), strongly expecting a noun phrase like "for Lindsay's storyline" or "she'll resolve the pregnancy" or similar.


**[7]** TV fandom forum post tone with casual, witty commentary about Kelly's pregnancy status and the show's upcoming storyline.

The phrase "So I'm hoping both Tori and Summer are dead already, but not sure...hmm, hoping" suggests an incomplete list or statement about one of the pregnant characters' fates, implying a character-specific wish or storyline direction about Kensi.

Final token "hoping" ends an incomplete clause ("but hoping... hoping"), part of a list of possible outcomes ("So far for Kelly's pregnancy, aside from her, hoping"), strongly expecting a specific named character or scenario like "she'll resolve" or "for both their fates to work out."


</details>


======================================================================

## position i=2  ·  token_at_pos = '"'

**context:** …13, 09:58 AM|
Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really, but hoping that we get some good "SAMANTHA GENE!!" Marlena Death-Stares out of it. And "newfound"


**AV(h16) [before]** (91 words):

> Fanfic forum post tone with quoted dialogue and sarcastic commentary about a TV show character's behavior, implying crude humor and mocking.

The phrase "And the whole 'recovered' and "newly" suggests a list of quoted phrases or words from the story, likely continuing a sarcastic accusation about the character's false promises or "recovered friendship" theme.

Final token "newly" ends mid-phrase (" "Re-found" and "fancied""), part of a quoted phrase sequence listing questionable descriptors ("And "re-found""), strongly expecting a noun like "friendship" or "fake discoveries" or "words like integrity" completing the ironic construction.


**AV(h24) [after / target]** (98 words):

> TV show forum post tone with fan speculation about Brooke's pregnancy storyline and character dynamics, implying ongoing soap opera dialogue.

The phrase "And the whole 'hating on Jack' thing...and her newfound " is a list of conflicting emotions/plot points, suggesting a second phrase about the other women's reconciliation or newfound family bonds, likely referencing the pregnancy or past relationships.

Final token "newfound" is mid-phrase in "or "new-found" — part of a quoted list ("with 'new found'"), strongly expecting a noun like "love for Sam" or "family loyalty" or "friendship with the other women" or "reconciliation with her past."


**selected best-of-8** (reward 0.8726 vs batch mean 0.8703):

> TV soap forum post tone with fan speculations about Kyle/Amber's relationship, referencing past episodes and soap storylines.

The phrase "Now that she's been 'discovering' her pregnancy and "new found" implies a contradictory or mixed emotion pattern, suggesting a list of character developments or a phrase about Samantha's reconciliation or newfound family bonds from the ex-wife storyline.

Final token "new-found" ends an incomplete noun phrase ("with "new found""), part of a contrasting clause ("either her 'reconciliation' and/or new-found"), strongly expecting "love for Mike" or "family ties" or "forgiveness from the other women" or similar emotional resolution concept.


<details><summary>all 8 candidates</summary>


**[0]** TV show forum post format with fan opinion/angst about Soap's pregnancy, establishing character dynamics around Brooke and Sam's reconciliation.

The phrase "But since she's been acting like "hating" her ex-husband AND newfound" suggests a list or contradiction of storyline themes, implying a third character arc or emotional resolution about family bonding or pregnancy revelation.

Final token "newfound" is mid-phrase in "or "new-found" implying a quoted list item or noun phrase continuation like "love for family" or "forgiveness with Taylor" or "friendship with other women," completing the contradiction about conflicting goals around the pregnancy reveal and the soap's reconciliatory arc.


**[1]** Fan fiction forum post tone with TV show references ("Sami/Samantha and Brooke") suggesting soap opera drama context around Kate's pregnancy and past hatred.

The phrase "But she's been acting all jealous of her ex's return...and new 'found' " implies a list or contradiction pattern about character arcs or emotional developments, likely a phrase about reconciling with the family.

Final token "new-found" ends an incomplete noun phrase ("with "new-found""), part of a contradiction clause ("either her 'reconciled' or new-found"), strongly expecting "love for the family" or "friendship with Beth" or "compassion for pregnant women" or similar.


**[2]** TV show forum post tone with fanfic character discussion about Brenda's pregnancy storyline and SG1 characters, implying ongoing soap opera conflicts.

The phrase "Now that Sam has her "new found love" and "discovering new-found" suggests a list or contradiction about Samantha's attitude toward Jack and other women, implying a plot resolution or character arc about reconciliation or family ties.

Final token "new-found" is mid-phrase in "or "new found" context, part of an incomplete list ("with 'new found' happiness and/or new-found"), strongly expecting "friendship with Jack" or "love for the family" or "loyalty to the past" or similar.


**[3]** TV soap forum post tone with fan speculations about Kyle/Amber's relationship, referencing past episodes and soap storylines.

The phrase "Now that she's been 'discovering' her pregnancy and "new found" implies a contradictory or mixed emotion pattern, suggesting a list of character developments or a phrase about Samantha's reconciliation or newfound family bonds from the ex-wife storyline.

Final token "new-found" ends an incomplete noun phrase ("with "new found""), part of a contrasting clause ("either her 'reconciliation' and/or new-found"), strongly expecting "love for Mike" or "family ties" or "forgiveness from the other women" or similar emotional resolution concept.


**[4]** TV soap fan forum post tone with character discussion and speculation about Brooke/Sarah's pregnancy drama, implying ongoing soap opera storyline.

The phrase "Like Kelly wanting to get over her hatred of Lila AND new 'found' " suggests a list of contradictory emotional states or plot developments, with "newfound family bond" implying a second "Family Reunion" or reconciliation theme.

Final token "new-found" is mid-phrase in "with 'new-found'" part of an incomplete list of character motivations ("either her 'reconciliation' and/or "new found"), strongly expecting a noun like "love for Sam" or "friendship with the other women" or "faithfulness."


**[5]** TV show forum post tone with fan fiction discussion about Erica Kane's pregnancy, implying ongoing soap opera drama context.

The phrase "And while she's been having fits over missing Luke and now "new found" suggests a contradiction or contradiction list pattern, implying two opposing themes ("reunion of ex-wives" or "conflict/love") about Brooke's character development and her newfound relationships.

Final token "new-found" is mid-phrase in an incomplete noun phrase ("with "new found""), part of a quoted list of "her newfound love/chemistry" clause, strongly expecting "friendship with Carter" or "loyalty to the family" or "forgiveness from past enemies."


**[6]** TV show forum post tone with fan speculation about Brooke's pregnancy storyline, referencing soap opera characters and their past struggles.

The phrase "Like she's having mixed feelings about Carter's return...and new "found" " implies a contradiction or dual sentiment, listing possible character arcs like "new found love" or "family bonds," suggesting a show continuity or reconciliation theme.

Final token "new-found" is mid-phrase in "with 'found' " noun phrase, part of a list of contradictory character behaviors ("her newfound love and/or "new found"), strongly expecting a noun like "friendship with Rachel" or "loyalty to her exes" or "compassion."


**[7]** TV soap fan forum post tone with character drama discussion about Lyla and Scottie's pregnancy situation involving ex-husband Sam.

The phrase "Like how Kelly's suddenly wanting to be with Sam again...and new 'found' " implies a list or contradiction of past storylines, suggesting a second theme about emotional reconciliation or pregnancy revelations for the WWE family.

Final token "new-found" is mid-phrase ("with "new-found""), part of a quoted list ("either her 'reconciliation' or new-found"), strongly expecting "love for Bethany" or "friendship with the other women" or "loyalty/faith" — continuing the contradictory emotional narrative context.


</details>
