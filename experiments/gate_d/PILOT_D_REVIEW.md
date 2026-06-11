# Gate D arm-H pilot — v4 prompt (post subagent review)

v2 -> v4 mechanical deltas over 100 pilot labels:
future-token leak ~40 -> 1; numeric weights 34 -> 1; 'The write' template
opener 100 -> 4; quoted-word groundedness 0.955 -> 0.969. Remaining: 40/100
contain one stray word from the (broad) ban list — handled by a
regen-on-violation pass at batch time.
Structural fixes: evidence context windows truncated at the position
(future leak now impossible); doc-start resting-state sources dropped
from bundles. Full issue list: pilot_d_review_subagent.md

## i=0 — target `' really'`
**context tail:** … Week of February 11th| |Lil||Feb 1 2013, 09:58 AM| Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really
**label:** The comma and "really" from the immediate context ("Sami, really") are pulled strongly across multiple reads, along with "care" and "about" from the parallel negation structure ("Don't care about Sami"). The write anchors the completion of the second negative clause by binding the emphatic adverb to the subject and the negated verb that precedes it, mirroring the rhythm of "Don't care about Chloe/Taniel/Jen-Jen" from the opening of the sentence.

## i=1500 — target `' can'`
**context tail:** … mental and fiscal resources. While there is no magic formula to starting up, Adeo’s 10 steps to launching a company can
**label:** The sequence "10 steps to launching a company" — pulled mainly from the middle of the text where Adeo's framework is introduced — anchors the grammatical completion at " can", linking the second mention of the 10-step structure back to its first. The word "steps" and the phrase "launching a company" dominate the reads across multiple attention traces, establishing that what follows will describe what this plan can accomplish. The write carries forward the core subject matter (the startup plan and its application) plus the verb form needed to transition into consequence or capability.

## i=3000 — target `' one'`
**context tail:** …governmental order, even though the images were taken from public streets. Was the government too hard on Google? As one
**label:** The question mark ending "Was the government too hard on Google?" and the phrase "As one" are pulled together as the strongest pattern, anchoring a transition into a new perspective after the rhetorical question. The word "one" itself carries forward from the local phrase while the question mark and "As" signal the shift in voice—from describing government action to inviting the reader into an opinion or counterargument. This write loads the grammatical scaffold for a clause beginning with "As one [noun]" that will present an alternative viewpoint.

## i=4500 — target `'.\n'`
**context tail:** …le new partnerships with US based production partners, design agencies and leaders in the field of additive technology. 
**label:** Punctuation marks—especially the closing ")." from "that soon!)." and the period from "Brooklyn Navy Yard."—dominate the reads, grounding the write in sentence-boundary structure. The attention also pulls "This" and "move" from the opening of the next sentence ("This move can enable"), along with scattered references to geographic and partnership content ("US based production partners", "Brooklyn", "New York") that contextualize the expansion narrative. Together these reads anchor the period as a major structural boundary while reinforcing the thematic continuity of locations and strategic moves in the passage.

## i=6000 — target `' capital'`
**context tail:** …f all charges,” Judge Vandana Tendulkar said after the trial of Samson D’Souza and Placido Carvalho in the state capital
**label:** The write pulls "Goa" from the opening narrative ("found on a Goa beach in 2008") and "court" from the earlier framing ("An Indian court on Friday cleared"), anchoring the legal proceeding in its geographic and institutional context. It also retrieves "state" and "I" (from "PANAJI" and "Judge Vandana Tendulkar said"), grounding the verdict in the state jurisdiction where the trial occurred. The local phrase "in the state capital" is carried forward to complete the locational specification of where Judge Tendulkar delivered her judgment.

## i=7500 — target `' ('`
**context tail:** … Standard), two leading standards which are commonly used in concert to verify combined carbon offsets and co-benefits (
**label:** The phrase "co-benefits" and its components ("co," "enef," "-b," "its") dominate the retrieved tokens, pulled repeatedly from the nearby phrase "carbon offsets and co-benefits." "Community" also registers from the earlier clause naming the "Climate, Community and Biodiversity Standard (CCB Standard)." This write locks onto the specific term that pairs with the standards being validated, reinforcing what MCDI expects those standards to verify alongside carbon offsets as the sentence reaches the opening parenthesis.

## i=9000 — target `' “'`
**context tail:** …a good loser,” when asked about the possibility of the November Presidential election results not being in his favour. “
**label:** The closing quotation mark pulls together the phrase "not a good loser," — the core claim Trump made — along with the preceding context "his favour. " that frames the conditional stakes of the election. Multiple reads of "loser," "good," and the comma scaffold the grammatical closure of the direct quote, while "that he is" anchors the attribution structure back to Trump's statement. This write completes the quoted material and the sentence structure that reports what Trump told Chris Wallace.

## i=10500 — target `' player'`
**context tail:** …aining health. In all game modes except osu!mania, there is an indicator moving along the health bar that shows a player
**label:** The phrase "health bar that shows" and its grammatical anchors ("that," "shows," "a") are pulled from the immediately preceding clause, along with the substantive term "health" from "moving along the health bar." The write also reads back to the sentence's opening context—"there is an indicator"—grounding the completion in the established setup. Together these retrievals reinforce the parallel structure completing "shows a player" and lock the local phrase into the sentence's ongoing description.

## i=12000 — target `' mechanical'`
**context tail:** … that it has been selected to provide the BP Decatur site with specialty turnaround management and turnaround mechanical
**label:** The phrase "turnaround management and turnaround" is pulled forward repeatedly across multiple reads, establishing the parallel structure that " mechanical" completes. "Specialty" and "with specialty turnaround management and" also arrive from earlier in the sentence, anchoring the descriptor that will follow. The write carries forward the noun phrase under construction ("and turnaround mechanical") while retrieving the organizational context—"BP Decatur site" and "selected to provide"—that frames what kind of service is being named.

## i=13500 — target `' That'`
**context tail:** …o 2% management fee levied on assets, hedge funds typically keep 20% of the profits generated each year as payment. That
**label:** The specific fee structure—"20% of the profits generated each year as payment"—anchors the write, with strong pulls on "profits," "of," and "payment" establishing the quantitative claim being completed. The broader context of performance fees appears alongside these pulls: "the performance fee" and the management structure ("Along with a 1% to 2% management fee") ground the sentence's topic. The write also catches the transitional marker "That" itself, completing the local phrase structure that opens the next clause.
