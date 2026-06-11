# Gate D arm-H pilot — 100 Haiku labels from attention evidence (v2 prompt)

Mechanical checks: 100/100 parsed; mean quoted-word groundedness 0.955;
banned mechanism-words 8/100 (v1 was 64/100 before the prompt fix).
Sampling: every 150th position (spread across the corpus).

## i=0 — target token `' really'`
**context tail:** … Week of February 11th| |Lil||Feb 1 2013, 09:58 AM| Don't care about Chloe/Taniel/Jen-Jen. Don't care about Sami, really
**top evidence:** 8%: "really"@0.6172, "|"@0.2197, ","@0.0669; 7%: "|"@0.2559, "care"@0.2402, "about"@0.1455; 6%: ","@0.4531, "|"@0.1514, "really"@0.1338
**label:** The write at " really" retrieves the immediately preceding context of "Sami," and the comma that follows it, which carries forward the local sentence structure and the object being evaluated. It also pulls in the repeated word "care" from earlier in the sentence ("Don't care about Sami"), reinforcing the pattern of negation and dismissal that frames this statement. Additionally, the write retrieves the document delimiter "|" with consistent attention across multiple retrieval paths, likely serving as a structural anchor or resting state that has minimal semantic contribution to the evaluation of Sami at this position.

## i=1500 — target token `' can'`
**context tail:** … mental and fiscal resources. While there is no magic formula to starting up, Adeo’s 10 steps to launching a company can
**top evidence:** 9%: "can"@0.6289, "As"@0.2041, "company"@0.0332; 7%: "steps"@0.3633, "can"@0.104, "As"@0.0947; 6%: "can"@0.208, "steps"@0.1426, "As"@0.1147
**label:** The write at " can" heavily retrieves the nearby phrase "launching a company can" (consistently weighted 0.2–0.6 across most retrievals), anchoring the local grammatical continuation. It also pulls in "10 steps to launching a" and "steps" from earlier in the sentence, reinforcing the structural parallel to the "10-step plan" mentioned at the start. Additionally, it retrieves "As many entrepreneurs know" from the opening, grounding the statement in the earlier context about entrepreneurs' experience with startup difficulty.

## i=3000 — target token `' one'`
**context tail:** …governmental order, even though the images were taken from public streets. Was the government too hard on Google? As one
**top evidence:** 13%: "one"@0.7695, "Because"@0.1006, "As"@0.0522; 7%: "?"@0.3652, "As"@0.2363, "."@0.0679; 5%: "?"@0.25, "Because"@0.2207, "one"@0.208
**label:** The write at " one" heavily retrieves the phrase "Google? As one forum member puts", establishing that this is introducing a forum member's perspective on the Google question posed earlier. It also pulls in "Because of a threat", connecting back to the security concern that framed the entire debate. Together, these retrievals set up the position to introduce an individual opinion—a forum member's view—that will respond to the question of whether the government was too hard on Google.

## i=4500 — target token `'.\n'`
**context tail:** …le new partnerships with US based production partners, design agencies and leaders in the field of additive technology. 
**top evidence:** 8%: "."@0.7461, "As"@0.1221, ")."@0.0508; 6%: ")."@0.4199, "."@0.1206, "."@0.0664; 5%: "This"@0.1973, ")."@0.1846, "."@0.1738
**label:** The write at this position retrieves multiple structural markers from earlier in the text—particularly the period ending "of additive technology." and the closing parenthesis from "that soon!)."—which anchor the transition between sentences. It also pulls in the opening words "As" and "This" from nearby sentences ("As part of our" and "This move can enable"), carrying forward the paragraph's structure and thematic continuity. These retrievals reinforce the sentence-boundary context and the logical flow connecting the announcement of new features to the discussion of enabling new partnerships.

## i=6000 — target token `' capital'`
**context tail:** …f all charges,” Judge Vandana Tendulkar said after the trial of Samson D’Souza and Placido Carvalho in the state capital
**top evidence:** 8%: "Goa"@0.3262, "P"@0.1494, "capital"@0.0908; 8%: "capital"@0.3125, "P"@0.2852, "court"@0.1084; 5%: "capital"@0.1904, "P"@0.1797, "state"@0.168
**label:** The write at " capital" retrieves "PANAJI" and the phrase "in the state capital Panaji" with consistently high weights across multiple retrieval paths, establishing that the word being completed refers to a specific geographic location. It also pulls in "Goa" from earlier in the text ("found on a Goa beach in"), reinforcing the Indian state context where this trial is taking place. Together these retrievals ground " capital" as the culmination of a location identifier—the state capital where Judge Vandana Tendulkar delivered her verdict.

## i=7500 — target token `' ('`
**context tail:** … Standard), two leading standards which are commonly used in concert to verify combined carbon offsets and co-benefits (
**top evidence:** 7%: "-b"@0.252, "co"@0.252, "enef"@0.1846; 7%: "enef"@0.4863, "co"@0.2158, "-b"@0.1484; 6%: "co"@0.3633, "enef"@0.2832, "-b"@0.1182
**label:** The write retrieves repeated references to "co-benefits" across multiple retrieval paths, pulling in the phrase "co-benefits (social" and the surrounding context of "carbon offsets and co-benefits" with substantial attention weights. It also draws from the nearby phrase "offsets and co-benefits (" immediately before the current position, along with scattered attention to "Community" and "Standards" from the broader discussion of verification standards. This consolidates the key concept being introduced—that co-benefits are a central feature of the standards being discussed—as the text opens a parenthetical explanation.

## i=9000 — target token `' “'`
**context tail:** …a good loser,” when asked about the possibility of the November Presidential election results not being in his favour. “
**top evidence:** 9%: "“"@0.7344, "In"@0.1738; 7%: "“"@0.3359, "In"@0.1162, "that"@0.0469; 6%: "“"@0.5195, "In"@0.2617, "."@0.1196
**label:** The write at this position retrieves the opening quotation mark and surrounding context from "his favour. "I am not"—the exact moment where Trump's direct quote begins—with substantial weight concentrated on this boundary between reported speech and the quoted statement itself. It also pulls in the phrase "not a good loser," from earlier in the quote, grounding the current position in Trump's actual claim. Together, these retrievals carry forward both the structural marker that a direct quotation is underway and the semantic content of what he claimed about himself, establishing continuity as the quoted statement continues.

## i=10500 — target token `' player'`
**context tail:** …aining health. In all game modes except osu!mania, there is an indicator moving along the health bar that shows a player
**top evidence:** 8%: "shows"@0.6602, "a"@0.1079, "that"@0.0894; 7%: "player"@0.7852, "For"@0.1279; 6%: "health"@0.375, "there"@0.2002, "health"@0.0574
**label:** The write at " player" retrieves the nearby phrase "shows a player's" (with high weight 0.78), establishing the immediate grammatical context where "player" appears as the object of "shows". It also pulls in the more distant phrase "health bar that shows" (weighted 0.23-0.37), reinforcing the subject and verb that frame this occurrence of "player". Additionally, the write retrieves the document's opening "For health as a" (consistently weighted across multiple retrievals at 0.11-0.20), grounding the current mention within the topic established at the start.

## i=12000 — target token `' mechanical'`
**context tail:** … that it has been selected to provide the BP Decatur site with specialty turnaround management and turnaround mechanical
**top evidence:** 8%: "mechanical"@0.8008, "K"@0.0845, "Mechanical"@0.0439; 7%: "turnaround"@0.4043, "and"@0.2451, "K"@0.1689; 7%: "turnaround"@0.3965, "K"@0.1465, "turnaround"@0.0781
**label:** The write at " mechanical" heavily retrieves the word "turnaround" from earlier in the text (appearing across multiple retrievals with weights up to 0.40), reinforcing that this position is describing "turnaround mechanical" services. It also pulls in "mechanical" itself from the nearby phrase "management and turnaround mechanical execution services," carrying forward the local semantic context. Additionally, the write retrieves "K" from "KBR Selected to" with consistent moderate weight across retrievals, anchoring the position within the announcement that KBR has been selected for these services.

## i=13500 — target token `' That'`
**context tail:** …o 2% management fee levied on assets, hedge funds typically keep 20% of the profits generated each year as payment. That
**top evidence:** 7%: "That"@0.6094, "Am"@0.2109, "."@0.0415; 7%: "."@0.2197, "of"@0.1943, "%"@0.0669; 6%: "fee"@0.1816, ","@0.0757, "as"@0.0732
**label:** The write at " That" retrieves the immediately preceding context—"payment" and the period—to carry forward the sentence structure, but its primary retrieval focuses on the substantive financial details introduced earlier: "20% of the profits generated each year" and "the performance fee" itself. This pulls in the specific fee structure being discussed ("20% of the profits") alongside the general concept of the "performance fee," establishing what "That" is about to refer to and explain in the next clause ("That fee structure creates").
