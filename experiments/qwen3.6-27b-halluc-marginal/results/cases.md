# Negative-marginal matryoshka items vs the standard NLA

Judge: `nex-agi/nex-n2-mini`. Marginal of item k = FVE(items 1..k) − FVE(items 1..k−1), own-critic. Damage = full − leave-one-out.

## Case 1: ci=244 rollout=1 item 2 — marginal -0.338

**Source tail** (…'?\n- How does irradiation affect foods?\n- How do you measure the amount of irradiation used?\n- How does irradiation affect disease-causing microbes?\n- Which foods can be irradiated?\n- Which foods have been approved for irradiation in the United States?\n- Which foods are being irradiated in the U.S.?\n')
**True continuation:** '- How can I tell if the food has been irradiated?\n- Are consumers ready to buy irradiated foods?\n- Would irradiation replace other food'

**Matryoshka explanation** (→ = the offending item):

-   [+0.068] "
- → [-0.338] Question list continues FAQ format about GMO prevalence
-   [+0.162] Next question "Where are GMOs grown now?" needs US answer quantity
-   [+0.150] Bullet list FAQ structure: third question "Which countries currently approve specific GMOs?" already answered, next topic: commercial production count or accession countries list answered
-   [+0.205] Document is older USDA official factual summary, nearly mid-section
-   [+0.002] FAQ headings pattern: "How many GMOs exist?", some answered, "What are currently grown in USA?" answered soon
-   [+0.009] Next bullet likely "Which GMO crops are available commercially?" or similar survey
-   [+0.075] Enumerative FAQ section: numbered questions predict "How many..." next: "How many are grown commercially?" already posed twice consecutively
-   [+0.040] 1998 page structure suggests US-specific adoption answer follows international approval question
-   [+0.033] Parallel question "Which crops are currently approved in other countries?" expects concrete numbers answer

Item verdict: **FABRICATED** | solo -0.247 | LOO damage -0.011

**Std match** (rollout 0, sent 3, marginal +0.038, solo -0.025, verdict CONTRADICTED): A further question about production statistics or specific commercial use is expected next, continuing the numbered FAQ sequence formatted as "15."
**Std match** (rollout 1, sent 2, marginal +0.366, solo +0.369, verdict CONTRADICTED): The list enumerates sequential questions from a government/regulatory document about Genetically Engineered Insulins, so the next entry continues this numbered FAQ list (number 9 and likely "Where are GE insulins currently being manufactured?" or similar).
**Std match** (rollout 2, sent 3, marginal +0.221, solo +0.617, verdict FABRICATED): Final token is an incomplete FAQ table of contents — "8. Where are GE crops used in research today?\n9. How many GE plants are commercially available?\n10. Where are GE plants produced?"
**Std match** (rollout 3, sent 6, marginal +0.640, solo +0.400, verdict CONTRADICTED): Final token "?" closes the third question ("Where are biotech products currently being used?"), and the list pattern strongly anticipates another question, likely "12. What biotech products are currently used in the U.S.?" or similar.

## Case 2: ci=218 rollout=2 item 1 — marginal -0.271

**Source tail** (…'ynaptic physiology that apply to both central and peripheral neurons. Although within the past 50 years we have vastly increased our understanding of the neurochemistry supporting transmitter synthesis, and the cytoarchitectural features of the nerve ending varicosities where these events take place')
**True continuation:** ', the cellular and molecular principles that govern transmitter storage and recycling are still elusive in many respects. We teach that transmitter storage and its reutilization are two separate'

**Matryoshka explanation** (→ = the offending item):

- → [-0.271] "place" microscopy structural knowledge despite historical advances
-   [+0.818] Contrast "although we have known X and also the ultrastructure of cellular compartments" continues mentioning imaging techniques or histology developments
-   [-0.002] Established scientific introduction framing a biophysical knowledge gap
-   [+0.062] Historical narrative "during past decades, both in our understanding of transport mechanisms AND in characterization of the detailed cellular architecture where these complex organelles reside" implies next: yet functional assay methodology lagged
-   [+0.004] Spanish biophysicist author tone academic opening of review article
-   [+0.002] "Despite progress both in quantification of synaptic vesicle cycle rates and in understanding, resolution of electron microscopy of the neuronal ultrastructure of these organs" implies "but molecular/physiological explanation remained elusive" before electrophysiology breakthroughs
-   [-0.004] "both in our best efforts to measure rates... and the ultrastructural details of the cellular compartment where these activities occur take place" expects "we historically lacked dynamic physiological correlation"
-   [-0.021] Semicolon after "place" suggests elaboration of classical techniques now complete, transitioning to newer discoveries or introduction thesis
-   [+0.008] 1970s Spanish neuroscience context implies foundational textbook chapter opening referencing vesicle morphology known via EM imaging
-   [+0.009] "Although over last decades, our quantitative estimates of uptake rates... and detailed architecture of neuron cell morphology" — ("and

Item verdict: **SUPPORTED** | solo -0.271 | LOO damage +0.037

**Std match** (rollout 0, sent 4, marginal +0.167, solo +0.329, verdict SUPPORTED): Final token "place" ends a subordinate "Although" clause mid-thought ("While we have thoroughly described in detail both the enzymatic steps... and the details of cellular microscopy..."), requiring continuation contrasting what was known with missing knowledge, likely "with advances in X" or "electron microscopy revealed..."
**Std match** (rollout 1, sent 3, marginal +0.210, solo +0.347, verdict CONTRADICTED): Final token "place" completes a subordinate clause begun with "Although we have long identified...," part of a compound ("While we identified both... and the structural details of the ultrastructural research tools we have employed to study the cellular anatomy of these processes")—next text will pivot to a contrast word like "the molecular mechanisms" or "electron microscopy."
**Std match** (rollout 2, sent 3, marginal +0.126, solo +0.313, verdict SUPPORTED): Final token "place" ends a multi-part subordinate clause mid-thought; "Although... both X... and the ultrastructural morphological details of these cells" awaits continuation with a contrastive main clause, likely referencing historical limitations.
**Std match** (rollout 3, sent 3, marginal +0.160, solo +0.368, verdict CONTRADICTED): Final token "place" closes a subordinate clause within a compound contrast ("While we have been able to characterize... both the metabolic properties... and, the structural aspects of the cytoarchitecture observed therein"), requiring continuation of the parallel "and" clause before pivoting — likely completing "we" with a reference to ultrastructural/cytological knowledge before transitioning to limitations.

## Case 3: ci=166 rollout=1 item 2 — marginal -0.222

**Source tail** (…'p improve the symptoms associated with these conditions.\nThere is currently no cure for Parkinson’s disease or any of the other conditions that can be treated with DBS, but the therapy is thought to be very effective in slowing down the progression of the disease or improving symptoms significantly.')
**True continuation:** '\nThere are a few potential side effects associated with DBS therapy, including headaches, mood swings, and sexual dysfunction. It is important to speak with your doctor'

**Matryoshka explanation** (→ = the offending item):

-   [+0.354] ." continues article explaining treatment benefits, continuation expected
- → [-0.222] AI-generated misleading medical article about knee surgery
-   [+0.040] Article structure "Currently, research suggests that X may be effective...". Positive claim then caveat expected
-   [+0.183] Paragraph explains mechanism and success rates, transitions to patient considerations
-   [-0.089] British English syntax still maintained "is a type of treatment..." third person generic
-   [+0.125] "and therefore many people with advanced arthritis may benefit from continued treatment." likely follows optimistic framing
-   [+0.013] Source paragraph mid-description of DBS therapy success, mentions effectiveness uncertainty
-   [-0.075] Coping with arthritis desperation article mixed with hallucinated AI content mismatch
-   [+0.117] "However, this treatment is still relatively new, and so more research is needed to fully understand its potential benefits and risks." typical filler conclusion
-   [-0.022] Benefits paragraph "Currently, scans are considered to be very successful at treating diabetic neuropathy, and many patients report significant improvement in their symptoms." ends. Next sentence continues with drawbacks/details of DBS success claim.

Item verdict: **FABRICATED** | solo -0.445 | LOO damage -0.036

**Std match:** none — the standard NLA does not assert this claim.

## Case 4: ci=185 rollout=2 item 3 — marginal -0.211

**Source tail** (…'e data-driven approach.\nPrevious publications in epistemology/sociology of science\n- Katona, Eszter, Németh, Renáta, Kmetty, Zoltán: Text analytics in social sciences – An example for NLP’s application (in Hung., submitted)\n- Bárdits, Anna, Németh, Renáta (2017): The rite of statistical significance')
**True continuation:** ' testing – contemporary critics; the rite in sociology. Szociológiai Szemle, 27:(1) pp. 119-'

**Matryoshka explanation** (→ = the offending item):

-   [+0.582] significance significance threshold cultural critique statistics paper citation "The cult of statistical significance"
-   [-0.005] Scientific sociology reference signals p-values misuse critique topic
- → [-0.211] Second citation "The cult of statistical significance" is famous open access article on statistical practice norms
-   [+0.038] "p-value" or hypothesis testing ritual likely follows
-   [+0.025] Book recommendation list continuing academic bibliography context
-   [-0.030] Two distinct references shown; second one title "The cult of statistical significance" expects continuation "p < 0.05" framing
-   [-0.036] Journal of Philosophy ofScience/STATISTICS discourse domain established
-   [+0.046] European Behavioural Social Sciences Network context suggests methodology critiques
-   [-0.022] "The cult of statistical significance" title published 2014 Perspectives on Science by Aldridge & Orsini
-   [+0.039] "Significance threshold" phrase implies critique of null hypothesis testing tradition

Item verdict: **CONTRADICTED** | solo -0.038 | LOO damage -0.122

**Std match** (rollout 0, sent 2, marginal +1.020, solo -0.178, verdict FABRICATED): Bibliographic list continuation: "Publications Recommendation: Methodological heterogeneity... / The cult of statistical significance" follows a pattern of listing publication titles by EuroSciPhil contributors, likely referencing authors and journal context on statistical practices.
**Std match** (rollout 3, sent 2, marginal +1.138, solo +0.244, verdict CONTRADICTED): List continuation momentum: publication highlights section is listing research outputs, each on a new bullet; "The cult of statistical significance" is clearly a book/chapter title in progress, requiring completion — e.g., "and p-values" or similar phrase critiquing statistical practices in science sociology.

## Case 5: ci=244 rollout=2 item 2 — marginal -0.174

**Source tail** (…'?\n- How does irradiation affect foods?\n- How do you measure the amount of irradiation used?\n- How does irradiation affect disease-causing microbes?\n- Which foods can be irradiated?\n- Which foods have been approved for irradiation in the United States?\n- Which foods are being irradiated in the U.S.?\n')
**True continuation:** '- How can I tell if the food has been irradiated?\n- Are consumers ready to buy irradiated foods?\n- Would irradiation replace other food'

**Matryoshka explanation** (→ = the offending item):

-   [+0.154] " questions list continues: current item asks which countries produce GMOs currently
- → [-0.174] FAQ format sequential questions about US agricultural GMO status
-   [-0.005] Next question likely asks market data or cultivation statistics answer
-   [+0.060] Third question "Which countries currently grow genetically engineered plants?" answered previous; "How many types of genetically engineered products are approved?" answered prior section awaits
-   [+0.261] FAQ from older regulatory document lists numbered questions continuing: next likely "Which foods actually contain GMOs today?" or equivalent
-   [+0.050] Document title structure: "commercially available" question needs answer section "What types are grown in US?" already covered; "How many genetically engineered products exist?" follows logically
-   [+0.084] Bullet navigation list pattern nearing end suggests "Which countries approve GMOs?" answered, "How many are approved?" answered, next "Are GMOs currently marketed?" parallels real-world consumption
-   [-0.012] Specific metric question "What percentage" or "current status" topic expected next
-   [+0.046] 1997-era USDA document genre conventions: factual inventory, Q&A list structure
-   [+0.010] Parallel list formatting: "Which countries have approved GE crops?", "What crops are currently grown in GE varieties in most countries?" next item is "Where are they grown?" or US-specific answer: "How much did US approve?" needs completion.

Item verdict: **CONTRADICTED** | solo -0.241 | LOO damage -0.012

**Std match** (rollout 2, sent 4, marginal -0.019, solo +0.076, verdict CONTRADICTED): The last question ends the current entry; the next line should be "11." followed by another question, likely about specific user companies or U.S./global commercial availability, mirroring sequential FAQ numbering about distribution/use evident from prior items.
**Std match** (rollout 3, sent 1, marginal -0.518, solo -0.518, verdict CONTRADICTED): FAQ/document structure with a numbered table of contents listing sequential questions and answers about genetically engineered crops in the US EPA context.

## Case 6: ci=218 rollout=0 item 1 — marginal -0.154

**Source tail** (…'ynaptic physiology that apply to both central and peripheral neurons. Although within the past 50 years we have vastly increased our understanding of the neurochemistry supporting transmitter synthesis, and the cytoarchitectural features of the nerve ending varicosities where these events take place')
**True continuation:** ', the cellular and molecular principles that govern transmitter storage and recycling are still elusive in many respects. We teach that transmitter storage and its reutilization are two separate'

**Matryoshka explanation** (→ = the offending item):

- → [-0.154] place microscopy techniques history understanding of cellular structure despite advances
-   [+0.607] "both in terms of molecular identity... and of knowledge... of the ultrastructure" sets up contrast "and although we have learned X, functional mechanisms remained elusive"
-   [+0.028] Clauses "and detailed structural understanding, and the ultrastructural anatomy of that organs" introduces historical imaging breakthroughs section incoming
-   [+0.013] Academic introduction format: Spanish biophysicist describing physiology research legacy
-   [+0.075] "despite our deep knowledge regarding ion transport molecules, and also the detailed ultrastructural characterization earned by development of electron microscopy in the morphology of cell biology T cells structure" suggests incomplete "and detailed cell anatomy studies"
-   [-0.001] Historical铺垫: "Through years, although we provided good understanding of kinds of channels involved in salt transport, and of the morphology of these epithelial cells" — both knowledge milestones achieved but functional studies lacking
-   [-0.007] "both the nature of ion channels" and "development of electron microscopy" implies structural/anatomical understanding paired with physiological gap
-   [-0.003] Intro clause "though relationship histologically ion biology, and detailed visualization of ultrastructural structure of these cells" continues with caveat "and ultrastructural morphology advancements, actual mechanisms remained unclear"
-   [+0.005] Latin/American Spanish academic writing style, mid 20th century cellular physiology framing expected
-   [-0.000] "not only

Item verdict: **FABRICATED** | solo -0.154 | LOO damage +0.046

**Std match** (rollout 0, sent 4, marginal +0.167, solo +0.329, verdict SUPPORTED): Final token "place" ends a subordinate "Although" clause mid-thought ("While we have thoroughly described in detail both the enzymatic steps... and the details of cellular microscopy..."), requiring continuation contrasting what was known with missing knowledge, likely "with advances in X" or "electron microscopy revealed..."
**Std match** (rollout 1, sent 3, marginal +0.210, solo +0.347, verdict CONTRADICTED): Final token "place" completes a subordinate clause begun with "Although we have long identified...," part of a compound ("While we identified both... and the structural details of the ultrastructural research tools we have employed to study the cellular anatomy of these processes")—next text will pivot to a contrast word like "the molecular mechanisms" or "electron microscopy."
**Std match** (rollout 2, sent 2, marginal +1.108, solo +0.442, verdict SUPPORTED): Narrative momentum: the sentence structure "Although we have, for decades, understood... extensively studied both the biological metabolic pathways underlying these functions and, with the electron microscope, the ultrastructural details of the cell compartment where they take place" sets up a contrast ("although"), promising a "but" clause highlighting a recent breakthrough or gap in knowledge, likely referencing freeze-fracture, morphology, or molecular advances.
**Std match** (rollout 3, sent 2, marginal +0.795, solo +0.318, verdict FABRICATED): Narrative momentum: the text contrasts past discoveries with gaps in understanding; "While, historically, we had a fairly good understanding of many cellular and biochemical functions of the egg cells, and, even the ultrastructural features of their cellular organization as revealed by electron microscopy" signals an incoming contrast ("but," "we lacked..." or similar pivot).

## Case 7: ci=133 rollout=2 item 2 — marginal -0.145

**Source tail** (…' predicated on an improper (or was lacking) foundation, basis or grounds.\n- The testimony was based on insufficient evidence, false assumptions or evidence not in the record.\n- The expert drew conclusions not supported by the evidence.\n- The testimony of the expert was impeached.\n- The testimony was')
**True continuation:** ' based on methods which were unscientific (not scientific, junk science).\n- The testimony would not assist the trier of fact.\n- The testimony was'

**Matryoshka explanation** (→ = the offending item):

-   [+0.560] was testimony objection categories incomplete list "or testimony was" redundant/inconsistent/unscientific repeating prior adjectives like irrelevant, improper, unreliable
- → [-0.145] Legal context enumerating objections hearsay inadmissibility arguments from opposing party
-   [+0.124] Parallel list structure "Or testimony was weak; or testimony was damaging" mirrors earlier items
-   [+0.034] Pattern suggests another adjective describing testimony quality defects
-   [+0.025] California Expert Witness fact sheet listing remaining rejected objection reasons
-   [+0.035] Final item "Or testimony was" requires predicate adjective like "speculative" or conclusionless
-   [+0.034] List continues with "Or the testimony was unsupported" variants cycling back to close
-   [+0.000] Document is administrative ruling context: compiled from court transcript citations
-   [+0.012] "Or testimony was irrelevant/corrupt/unsafe" near end list; maybe "unprofessional" or opposed
-   [+0.008] Final punctuation suggests list exhaustion; next item before period likely "overly subjective" synonym or newly introduced objection type like "egregiously misleading"

Item verdict: **FABRICATED** | solo -0.381 | LOO damage -0.011

**Std match** (rollout 0, sent 1, marginal -0.703, solo -0.703, verdict FABRICATED): Legal appellate memorandum establishing a comprehensive catalog of other courts' grounds for excluding expert testimony, systematically listing grounds excluding "insufficient basis" from prior cases.
**Std match** (rollout 1, sent 1, marginal -0.474, solo -0.474, verdict META): Legal appellate briefing format: case summary followed by exhaustive enumeration of appellate challenges to expert testimony exclusions, each introduced by "The testimony was," continuing the pattern.
**Std match** (rollout 3, sent 3, marginal +0.257, solo -0.148, verdict SUPPORTED): The enumerated list of excluded grounds is clearly ongoing, covering completeness, relevance, reliability, credibility, etc., and expects continued entries.

## Case 8: ci=232 rollout=2 item 2 — marginal -0.134

**Source tail** (…'on battle tactics of the 19th century Prussian army.\nBungay’s premise is that the leaders of almost all organizations can define reasonably good strategies. Where executives often fall down is leading their organization to execute their strategy. Bungay describes this problem as gaps and advises how')
**True continuation:** ' to close the gaps.\nHis assertion is that similar to military campaigns in war when a strategy encounters the real world then three types of gaps appear. He describes'

**Matryoshka explanation** (→ = the offending item):

-   [+0.494] how book explains solutions to identified management flaw causes
- → [-0.134] Vandermerwe explains problems and how to avoid them "identifies, and explains how"
-   [+0.187] Bullet continuing: chapter covering gap pitfalls "defines common... organizational management gaps and how" fixes
-   [-0.007] Book summary register mid-sentence continuation "identification problems identified, and explains how" to prevent/gain
-   [-0.009] Classic management consulting document genre, older author tour
-   [+0.013] Summary book explains strategic leadership concepts; subsection now prescribes remedies
-   [+0.018] "identifies a common failing condition of companies... and explains how" signals actionable methodology follows
-   [+0.006] Parallel list structure: "he identifies the gap problem and explains how [to remedy it]"
-   [+0.005] Specific chapter or recommendation will name avoiding mission-drift tactics
-   [+0.006] 1995 audio narrator description continues; remaining book critique content expected

Item verdict: **FABRICATED** | solo -0.106 | LOO damage +0.006

**Std match** (rollout 0, sent 3, marginal +0.028, solo +0.620, verdict FABRICATED): Final token: "how" is part of "He identifies a key problem gap in organizational execution can close by leadership failure. He identifies and explains how" — this immediately requires a continuation describing specific recommendations or remedies (e.g., "to avoid/overcome/correct these gaps"), completing the subordinate clause about Professor Z framing solutions to execution gaps.

## Case 9: ci=244 rollout=3 item 1 — marginal -0.116

**Source tail** (…'?\n- How does irradiation affect foods?\n- How do you measure the amount of irradiation used?\n- How does irradiation affect disease-causing microbes?\n- Which foods can be irradiated?\n- Which foods have been approved for irradiation in the United States?\n- Which foods are being irradiated in the U.S.?\n')
**True continuation:** '- How can I tell if the food has been irradiated?\n- Are consumers ready to buy irradiated foods?\n- Would irradiation replace other food'

**Matryoshka explanation** (→ = the offending item):

- → [-0.116] " question list continues answering how many countries use GMOs statistics
-   [+0.113] Next bullet question "Which countries currently use GMO movement in USA?" just answered, next is "Where are regulated/approved GMOs currently sold?" repeated
-   [+0.132] FAQ format document lists sequential heading questions about GMOs
-   [+0.138] Third question "Which organisms have been approved?" answered; "What are currently being grown?" next asks actual quantity/adopted usage
-   [+0.014] Expected next heading: "Which countries grow GMOs today?" pending answer section #9 as pattern
-   [+0.169] 1995 government web page lists remaining questions like "How many GMO events approved worldwide?", "What is U.S. current status?"
-   [+0.035] Next question in bulleted list: "Which are America's currently grown GMO crops?" likely follows "- What are approved GMO crops?"
-   [-0.004] Old USDA FAQ structure continues numbered Q&A unfolding
-   [+0.001] " How many types of GMO crops exist?
-   [-0.045] Where used?" section transitioning to another question "Are GMOs actually grown commercially?"

Item verdict: **FABRICATED** | solo -0.116 | LOO damage -0.071

**Std match** (rollout 3, sent 5, marginal -0.045, solo -0.294, verdict CONTRADICTED): A fourth or more "Where/How many" question is expected next.

## Case 10: ci=244 rollout=0 item 2 — marginal -0.115

**Source tail** (…'?\n- How does irradiation affect foods?\n- How do you measure the amount of irradiation used?\n- How does irradiation affect disease-causing microbes?\n- Which foods can be irradiated?\n- Which foods have been approved for irradiation in the United States?\n- Which foods are being irradiated in the U.S.?\n')
**True continuation:** '- How can I tell if the food has been irradiated?\n- Are consumers ready to buy irradiated foods?\n- Would irradiation replace other food'

**Matryoshka explanation** (→ = the offending item):

-   [+0.068] "
- → [-0.115] question list continues US GMO foods section FAQ structure
-   [+0.067] Next item asks quantity/current countries with commercial GMO approvals
-   [-0.032] "Which crops are genetically modified in different countries?" answered, next likely "How many GMO products are available in US market?"
-   [+0.087] FAQ format with numbered headers mirrors earlier parallel questions
-   [+0.098] Page navigation: third question "What is/X is now? Where?" follows: "Which GM foods are grown commercially?" third question asks US usage
-   [+0.150] Document is USDA or regulatory informational page from ~1998-2000 era
-   [+0.024] Next question "Are there major country approvals?" then "What are available in US?" suggests "How many GMO commercial products exist?" answer follows
-   [+0.005] Bullet Q&A pair pattern: question then answer section headings repeat
-   [+0.060] "Are GMO Foods approved commercially?" → "What countries have approvals?" → "Which crops are/GMOs grown commercially today?" → "What are currently sold in U.S.?"

Item verdict: **FABRICATED** | solo +0.068 | LOO damage -0.023

**Std match** (rollout 2, sent 4, marginal -0.019, solo +0.076, verdict CONTRADICTED): The last question ends the current entry; the next line should be "11." followed by another question, likely about specific user companies or U.S./global commercial availability, mirroring sequential FAQ numbering about distribution/use evident from prior items.
**Std match** (rollout 3, sent 1, marginal -0.518, solo -0.518, verdict CONTRADICTED): FAQ/document structure with a numbered table of contents listing sequential questions and answers about genetically engineered crops in the US EPA context.

## Case 11: ci=83 rollout=1 item 2 — marginal -0.113

**Source tail** (…'ting carrots will not make you fat.\nI’m pretty sure it was The South Beach Diet that was really responsible for popularizing the myth that carrots were “fattening” because the diet encouraged people to be conscious of the glycemic index of the carbohydrate-containing foods they ate.\nWhat is the Glyc')
**True continuation:** 'emic Index?\nThe glycemic index (GI) is a numerical system of measuring how quickly a carbohydrate containing food turns to glucose (blood sugar). The slower'

**Matryoshka explanation** (→ = the offending item):

-   [+0.404] 2 glycemic impact carbohydrate digestion number blood sugar
- → [-0.113] GI definition explanation "Gly... index measures how carbs spike" is next
-   [+0.147] Carrots GI section mid-explanation carbohydrate metabolism mechanics
-   [+0.059] Whole carrots vs juice comparison nutrition blog register
-   [+0.043] "Here's why GI score matters: The G" introduces glucose digit number carbs
-   [+0.018] Number Glycemic is truncated "2 Gly" likely becomes glucose absorption rate
-   [+0.030] Wheat berries low GI because fiber slows sugar entry bloodstream
-   [+0.009] GI topic introduced "Glycemic Index" measures how quickly carbs convert glucose
-   [+0.006] Nutrition info context vegan health advice requires plain accessible explanation
-   [+0.009] Carrots GI specifically addresses blood sugar spike timing mechanism

Item verdict: **SUPPORTED** | solo -0.189 | LOO damage -0.014

**Std match** (rollout 0, sent 5, marginal +0.007, solo -0.237, verdict FABRICATED): "GI = how fast potatoes spike blood glucose."
**Std match** (rollout 1, sent 4, marginal +0.022, solo +0.499, verdict FABRICATED): The immediate next token must complete "Glyc" as "emic" — describing how/slowly carbs affect blood sugar, explaining starch digestibility.
**Std match** (rollout 2, sent 3, marginal +0.813, solo +0.636, verdict FABRICATED): Final token "Glyc" is a truncated repetition of "GI = Glycemic Index" immediately following the heading "So, what is glycemic index? The GI stands for Glyc—" requiring completion of "emic," then explaining how GI measures blood sugar spike and carbohydrates are absorbed quickly despite potatoes' starch — the article must explain that carbohydrate digestion doesn't spike blood sugar in potatoes due to fiber/fat/fiber.
**Std match** (rollout 3, sent 3, marginal +0.748, solo +0.254, verdict SUPPORTED): Final token "Glyc" is a truncated word ("Glycemic Index: What is it?") mid-sentence: "The glycemic index (GI) measures..." — the incomplete word spells "[+emic]" continuing naturally.

## Case 12: ci=216 rollout=1 item 3 — marginal -0.109

**Source tail** (…'hts, general ideas, tricks and drive-by remarks on the topic of generating performant C++ code, comparable in time to QQmlComponent-based workflow (yes, it is actually non-trivial to generate faster code "out of the box" in the broad case - e.g. calqlatr demo).\nSo, the list (in no particular order):')
**True continuation:** '\n- The big problem of generated code vs QQmlComponent is the lack of QQmlObjectCreator (and its shared state) in the former case. There'

**Matryoshka explanation** (→ = the offending item):

-   [+0.354] : optimization topics list begins compiler/codegen game
-   [+0.101] Performance optimization wishlist for entity simulation library
- → [-0.109] "First thoughts:" introduces bullet list of C++ optimization ideas
-   [+0.048] Inlining, loop unrolling, data locality suggestions expected next
-   [+0.010] Context: user-facing generated code, inefficient math calls highlighted
-   [-0.008] "Some thoughts on the generated code" promises technical specifics
-   [-0.036] Hedged informal dev notes register, IRC team currently debugging
-   [+0.053] Multiple angles: branching, memory layout, avoiding function call overhead
-   [+0.042] World update interface overhead is central performance concern
-   [+0.050] "TODO:" signals first optimization point immediately follows :

Item verdict: **SUPPORTED** | solo -0.120 | LOO damage -0.079

**Std match** (rollout 0, sent 3, marginal -0.309, solo -0.386, verdict FABRICATED): The phrase "some initial thoughts:" explicitly signals a bullet-point list is about to begin, likely markdown `-` items.
**Std match** (rollout 1, sent 3, marginal -0.197, solo -0.260, verdict FABRICATED): The final token "tricks):\n" ends a list introducer clause ("Main question... code generation output should result in performance... With some basic C++ knowledge in mind. Initial thought list:"), acting as a section header demanding list items follow — likely starting with bullet points or numbered items covering topics like inline functions, loop unrolling, or codegen optimizations specific to this framework.
**Std match** (rollout 2, sent 2, marginal +0.170, solo -0.005, verdict SUPPORTED): List introduction signals an enumerated set of points: "Some ideas/ideas:" explicitly promises concrete items follow (e.g., inline functions, loop optimizations, data layout considerations, etc.).
**Std match** (rollout 3, sent 4, marginal +0.023, solo +0.197, verdict CONTRADICTED): Final token "Some initial motivating items:" — a colon ending an introductory clause, directly preceding a bulleted list; next tokens will be list items (e.g., "- [optimization technique]") covering things like loop unrolling, SIMD, memory layout, or constexpr usage.

## Case 13: ci=133 rollout=1 item 2 — marginal -0.107

**Source tail** (…' predicated on an improper (or was lacking) foundation, basis or grounds.\n- The testimony was based on insufficient evidence, false assumptions or evidence not in the record.\n- The expert drew conclusions not supported by the evidence.\n- The testimony of the expert was impeached.\n- The testimony was')
**True continuation:** ' based on methods which were unscientific (not scientific, junk science).\n- The testimony would not assist the trier of fact.\n- The testimony was'

**Matryoshka explanation** (→ = the offending item):

-   [+0.594] was testimony objection categories incomplete list continuation "The testimony was" repeated invalidity adjective
- → [-0.107] Legal evidentiary objection enumerations pattern continuing "or vague, irrelevant" theme
-   [+0.136] Parallel list structure: "or the testimony was unsupported, unreliable... or" expects synonym
-   [-0.039] California appellate court litigation context defining exclusion criteria
-   [+0.065] List of adjectives attacking opposing expert witnesses' credibility still ongoing
-   [+0.026] Earlier items: "testimony was inadmissible/unscientific/unhelpful/inconsistent; ...or testimony was biased" suggest duplicate adjective like confusing or speculative
-   [+0.016] Final item in long exhaustive quoted list of dismissal rationale descriptions
-   [-0.004] "or The testimony was" likely completed by spatiotemporal, procedural flaw adjective e.g. improper, unresponsive
-   [-0.001] Quoted block from appellant's brief making humorous/ironic exhaustive complaint list
-   [+0.014] "or the testimony was" often followed by "misleading" or "erroneous" mirroring prior synonyms

Item verdict: **META** | solo +0.023 | LOO damage -0.007

**Std match** (rollout 0, sent 2, marginal +1.367, solo +0.610, verdict FABRICATED): The author follows a clear enumeration pattern: "The testimony was unreliable, irrelevant, unscientific, speculative opinion, invalid, unsubstantiated, without foundation, incompetent, rendered inadmissible; the witness was not qualified...guesswork...biased...prejudiced; the testimony was credulous; the testimony was" — each item a short, parallel disqualifying descriptor.
**Std match** (rollout 1, sent 1, marginal -0.474, solo -0.474, verdict META): Legal appellate briefing format: case summary followed by exhaustive enumeration of appellate challenges to expert testimony exclusions, each introduced by "The testimony was," continuing the pattern.
**Std match** (rollout 2, sent 3, marginal +1.413, solo +0.482, verdict CONTRADICTED): Enumerated list pattern: the text has been listing struck testimony descriptions ("speculative," "unreliable," "given scant evidentiary or legal weight," "lacking qualifications," "credibility attacks") — each clause beginning "The testimony or expert was [adjective phrase]" — clearly continuing toward exhaustion.
**Std match** (rollout 3, sent 3, marginal +0.257, solo -0.148, verdict SUPPORTED): The enumerated list of excluded grounds is clearly ongoing, covering completeness, relevance, reliability, credibility, etc., and expects continued entries.

## Case 14: ci=19 rollout=0 item 1 — marginal -0.106

**Source tail** (…'gone\nto lift the tiny creatures\nfrom their plastic world\nand set them,\nconfused and stunned,\nupon the sand\nto turn in circles\nuntil we placed them right again\nsometimes patting their tails\nto encourage their voyage\nto a new life shocking in its largeness.\n“What is this\nlifting up and putting down?”\n')
**True continuation:** 'they must have thought,\n“and then this broad expanse\nthat lifts us, spins us,\nCourageously, they lifted their heads to swim'

**Matryoshka explanation** (→ = the offending item):

- → [-0.106] "
-   [+0.291] ?" poetic voice asks philosophical question mid-hike meditation
-   [+0.200] “What is this?” quoted dialogue fragment begins / attribution follows “ “Who asks?”
-   [+0.003] Byte-sized prose poetry style: retrospective mystical-ecological observation
-   [-0.004] Internal monologue shifts: lines of self-questioning about suffering ceremony
-   [-0.004] “
-   [+0.002] ‘What kind of pushing?’,
-   [+0.024] ” suggests another unnamed interlocutor or narrator voice responds
-   [+0.009] Nature walk narrative continues with human compassion metaphor “pushing”
-   [+0.000] Obsidian/contemplative American poet register (likely Mary Oliver adjacent sequence)

Item verdict: **SUPPORTED** | solo -0.106 | LOO damage +0.026

**Std match:** none — the standard NLA does not assert this claim.

## Case 15: ci=139 rollout=3 item 2 — marginal -0.098

**Source tail** (…'very human cell. And it’s a transformative textbook of medicine, with insights that will give health care providers immense new powers to treat, prevent and cure disease.”\nIt was evident in the minds of most scientists that were involved in this monumental task, that, once complete it would turn the')
**True continuation:** ' tide of disease prevention and treatment, and advance our knowledge of medicine to a level not witnessed within the last 100 years.\nFurthermore, as stated'

**Matryoshka explanation** (→ = the offending item):

-   [+0.607] the turn biomedical revolution genetics would turn medicine upside down
- → [-0.098] "would turn" signals paradigm shift in medicine promised
-   [+0.022] Human Genome Project context introduced, revolutionary future implications
-   [-0.001] Book genre: popular science/history presenting dramatic contrast narrative
-   [+0.154] "this genetic map ... would turn the" implies medicine/biology textbook inversion
-   [-0.001] 1990 founding momentum claims genomic knowledge fundamentally changes disease approach
-   [-0.003] "some predicted that" implies the genome discovery would revolutionize drug design or treatment
-   [+0.001] Gene-centric deterministic view critique is the book's thesis, setup here
-   [+0.014] "turn the" likely followed by a specific domain: medical field, biology textbooks
-   [+0.008] Rhetorical setup: grand technological achievement framing "the" + noun like "age" or traditional medicine upside down

Item verdict: **SUPPORTED** | solo +0.385 | LOO damage +0.016

**Std match** (rollout 0, sent 3, marginal +1.333, solo +0.762, verdict CONTRADICTED): Final token "turn the" is mid-sentence, part of a clause ("The promise of DNA mapping, which at the time appeared to explain all the answers about life, was expected to turn the"), requiring a noun phrase completing what HGP was expected to transform — likely "paradigm of medicine," "foundations of biology," or similar epochal language about transforming future healthcare.
**Std match** (rollout 1, sent 3, marginal +0.527, solo +0.735, verdict SUPPORTED): Final token "the" is mid-sentence, completing "It was anticipated that the sequencing of the human genome in 2003 would turn the" — immediately requires a figurative or revolutionary phrase like "page of medicine" or "entire field of biology on its head," describing promised transformative impact.
**Std match** (rollout 2, sent 4, marginal +1.206, solo +0.696, verdict SUPPORTED): Final token "turn the" is mid-sentence, part of "It was thought that such powerful base-line information would turn the" — grammatically requires a noun phrase completing this thought, likely referencing a paradigm shift in medicine or biology (e.g., "field of medicine upside down" or "entire approach to medicine/pathology on its head").
**Std match** (rollout 3, sent 3, marginal +0.197, solo +0.625, verdict SUPPORTED): Final token "turn the" opens an idiom or metaphor (e.g., "turn the page/field upside down"), requiring a noun phrase describing a transformation — almost certainly referencing medicine or biological understanding fundamentally changing our approach to healthcare; specifically a positive claim about human genetics revolutionizing medicine before a critique.

## Case 16: ci=142 rollout=1 item 2 — marginal -0.080

**Source tail** (…'aughton about more details on this 4 session residency that can be tailored specifically to meet the needs of your students at your school.\nMax. Capacity: 30\nTransportation Needed? No\nWho Arranges Transportation?\nIs Financial Assistance Offered? No\n“You are Special-You are Unique,” is a four session')
**True continuation:** ' program for 2nd – 8th grade students, held once a week in the classroom and is presented by teaching artist Hedda Matza-Haught'

**Matryoshka explanation** (→ = the offending item):

-   [+0.564] session four session educational art program structure described
- → [-0.080] "Four session" repeated curriculum introduces workshop format
-   [+0.056] Workshop description begins, NYC elementary school outreach context
-   [+0.146] Four-part curriculum structure mirrors "is a four session" intro topic
-   [+0.010] Sessions likely cover peace/migration themes across multiple visits
-   [-0.006] Program description announcement genre "This program is a four session" implies lesson count specified twice
-   [+0.014] Artist educator workshop offered schools, improvisation/arts integration
-   [-0.002] "Peace & The Immigrant Experience Project" is a "four session" unit, each meeting described next
-   [-0.005] Introductory outreach program likely lists session topics sequentially
-   [+0.000] "novel" signals hands-on interdisciplinary art curriculum for children continues "is a four session" with sessions, weeks, meetings noun

Item verdict: **SUPPORTED** | solo +0.353 | LOO damage +0.005

**Std match** (rollout 0, sent 2, marginal +1.359, solo +0.641, verdict CONTRADICTED): Program description momentum: "Holistic self expression is a four session" sets up an expected elaboration of the curriculum structure — already establishing it's a "four session program," mirroring the sidebar's "Art Workshops - Four Lessons" reference, so the following text must describe what those four sessions entail.
**Std match** (rollout 1, sent 4, marginal +0.019, solo +0.472, verdict FABRICATED): The repetition of "four sessions" from the title and "This Program is a four session" demands具体内容 about sessions 1–4 described sequentially.
**Std match** (rollout 2, sent 4, marginal +0.712, solo +0.577, verdict FABRICATED): Final token "session" is mid-compound: "Our Voices is a four session" — immediately requires a noun phrase completing the description of the curriculum (e.g., "workshop series" or transitioning into session details).
**Std match** (rollout 3, sent 5, marginal +0.008, solo +0.588, verdict FABRICATED): The phrase "The Art of Diversity is a four session" demands a plural noun ("workshop/art program") followed by curriculum specifics.

## Case 17: ci=162 rollout=3 item 3 — marginal -0.079

**Source tail** (…'servation convention, known as the \'Convention Concerning the Protection of the World Cultural and Natural Heritage\' commonly know as the "World Heritage Convention" began as a UNESCO agreement in 1972. It came into force with the sufficient level of ratifications in 1974 and achieved substance in 1')
**True continuation:** '978 with the inscription of the first four natural sites on the World Heritage List. By 1994 there were 137 signatories'

**Matryoshka explanation** (→ = the offending item):

-   [+0.403] 2 UNESCO criteria numbers Heritage global significance criteria
-   [+0.158] enumeration criteria for World Heritage sites exists
- → [-0.079] criteria framework "criteria" or numbered values UNESCO uses "10 criteria" typically
-   [+0.011] formal declaration context "to be nominated qualify under UNESCO"s defined standards
-   [+0.007] "World Heritage sites" section introduces heritage evaluation metrics
-   [+0.009] evaluative admissions usually cite qualities: biodiversity, cultural, natural
-   [-0.014] Caribbean/nature tourism context moving into specific listing criteria
-   [+0.021] UNESCO official text "A World Heritage Site is selected...based on criteria" standard UNESCO language
-   [+0.042] "1" begins numeric criteria reference, "criteria" or "i" expected next
-   [-0.003] Introductory explainer framing signals Wikipedia/UNESCO definition continuation

Item verdict: **SUPPORTED** | solo -0.095 | LOO damage -0.013

**Std match** (rollout 0, sent 3, marginal +0.719, solo +0.258, verdict CONTRADICTED): The text is building an argumentative case for World Heritage recognition, establishing global context; "178 member countries" signals the reader should expect specific UNESCO selection criteria (outstanding universal value, natural/cultural criteria I-XI).
**Std match** (rollout 1, sent 3, marginal +0.453, solo -0.132, verdict CONTRADICTED): Final fragment "1" is an OCR/typo repetition of "in 1" (likely "1972"), but more likely text is mid-word or mid-sentence — "criteria" just stated, continuing "Article 1" or UNESCO's numbered evaluation criteria.
**Std match** (rollout 2, sent 4, marginal +0.711, solo +0.651, verdict CONTRADICTED): Final token "1" is part of re-referencing "criteria numbers" — the word "within the world" earlier used "10" implicitly; here, a truncated reference to UNESCO's evaluation criteria means the next tokens will complete specific numeric criteria (e.g., "outstanding universal value" language or biodiversity/natural/cultural heritage criteria lists).
**Std match** (rollout 3, sent 3, marginal +0.924, solo +0.744, verdict CONTRADICTED): Final token "of 1" completes the phrase "World Heritage site" which ends a sentence beginning "established in 1972" — this repetition of "1972" is incomplete, strongly constrained to continue with UNESCO selection criteria text, commonly "11 criteria" or Roman numeral criteria or phrases like "outstanding universal value," referencing the Convention's qualifying standards for natural/cultural sites.

## Case 18: ci=96 rollout=0 item 2 — marginal -0.079

**Source tail** (…'o advance food security and nutrition objectives. Nepal was one of 19 focus countries chosen for the Presidential FTF initiative in 2010. The project has two primary and integrated components:\nA) Increase agricultural productivity\nB) Improve nutrition\nThe project will work in close coordination with')
**True continuation:** ' the GoN’s Ministry of Agricultural Development (MOAD) and Ministry of Health and Population (MOHP). Moreover, KISAN will engage with a wide'

**Matryoshka explanation** (→ = the offending item):

-   [+0.615] with government ministries Nepal agriculture development partnership structure USAID programs align government counterparts and local stakeholders
- → [-0.079] "USAID cooperation closely aligned with" implies ministries, provinces, DLS government implementation partners required
-   [+0.050] USAID program design convention lists institutional partnership coordination
-   [+0.044] The program description follows standard USAID feeding program structure which interacts with national agricultural policy frameworks
-   [+0.004] "Implementation will align closely with" signals ministry or gubernatorial partners, bottom-up decentralized approach Nepal context
-   [-0.005] IGAD rural development sector document often mentions government counterparts first, then NGOs/private sector
-   [+0.017] USAID Nepal Ag Program typically partners with local government implementing partners at national and provincial level
-   [+0.033] Enumeration "program conducts five pillars" requires public-sector counterpart structure "in coordination with" Ministry of Agriculture Nepal
-   [+0.017] Technical assistance project framing often specifies parallels to national strategies or Ford Foundation's grassroots NGO networks
-   [-0.015] Formal USAID project document genre momentum: after USAID introduction, partnership with partner governments articulated next "with" governments and communities stated in opening

Item verdict: **FABRICATED** | solo +0.422 | LOO damage -0.003

**Std match** (rollout 0, sent 2, marginal +1.163, solo +0.577, verdict SUPPORTED): The sentence structure "This project will be closely coordinated with" signals an imminent list of partner government ministries or implementing partners, continuing the pattern of crediting institutional collaboration — specifically Nepalese government partners or local agencies, likely contrasting with USAID's private/donor funding role.
**Std match** (rollout 1, sent 2, marginal +1.386, solo +0.711, verdict FABRICATED): Narrative momentum: the text has outlined problem context, program design, and outcomes; "USAID NIFA will work in close partnership with" signals an upcoming list of key governmental or institutional partners, particularly relevant to Nepal's agricultural extension systems and US operational localization.
**Std match** (rollout 2, sent 4, marginal -0.028, solo +0.632, verdict SUPPORTED): Final token "with" opens a prepositional phrase requiring a named stakeholder; after "The program will also maintain key synergies with," the next tokens will likely list Nepalese government partners or state-level/provincial ministries involved in Nepal's enabling environment or federal agricultural institutions.
**Std match** (rollout 3, sent 4, marginal +0.010, solo +0.523, verdict SUPPORTED): Final token "with" opens a prepositional phrase requiring noun objects — likely "Governments/State agricultural ministries," "the Nepalese government," or similar institutional partners (RTI's standard commitments cite State line ministries and ministry of agriculture).

## Case 19: ci=63 rollout=2 item 3 — marginal -0.072

**Source tail** (…'prüske country of origin, nationality, & ethnicity\n- Asal-prüske last name meaning & etymology\n- Asal-prüske spelling & pronunciation\nLatest photos on AncientFaces\nNo one from the asal-prüske community has shared photos. Here are new photos on AncientFaces:\nAsal-prüske Country of Origin, Nationality')
**True continuation:** ', & Ethnicity\nNo one has submitted information on asal-prüske country of origin, nationality, or ethnicity. Add to this section\nNo content has'

**Matryoshka explanation** (→ = the offending item):

-   [+0.255] nationality surname origin section header expects ethnicity/culture descriptors
-   [+0.228] "Origin, Nationality" phrase standard genealogy website template continues
- → [-0.072] Parallel title "NORRA Surname Origin & Ethnicity: Country, Ethnicity, Nationality" is header prompt
-   [-0.034] "What is surname NORRA? Country of Origin, Ethnicity, Nationality" implies next token nationality-related users
-   [+0.054] Standard template pages list "Origin, Nationality" heading often followed by "or Heritage"
-   [+0.009] Genealogy platform genre: formulas like Irish, Scottish, Italian or formal labels likely next
-   [-0.003] Prompt "Surname NORRA Country of Origin, Nationality" repeats question,". Next section likely geographic/cultural categorization
-   [-0.011] Website genre conventions: "Surname Origin & Nationality" headers typically enumerate: "What is X, Ethnicity, Nationality" suggests race/ancestry categories next
-   [-0.016] Structured FAQ genre: question form established, answer template about ethnicity/surnames
-   [-0.008] "Country of Origin, Nationality" phrase pairs often followed by "or Culture" completing the prompt title repetition

Item verdict: **FABRICATED** | solo +0.017 | LOO damage -0.047

**Std match** (rollout 0, sent 5, marginal +0.270, solo +0.470, verdict FABRICATED): Final token "Nationality" is mid-header, part of "Denoic Surname Country of Origin, Nationality," likely requiring a continuation such as "and Ethnicity" or a closing punctuation before body text about ancestral heritage begins.
**Std match** (rollout 1, sent 4, marginal +0.127, solo +0.255, verdict SUPPORTED): Final token — "Nationality" ends an incomplete section heading ("Clue Surname Country of Origin, Nationality"), which immediately requires continuation such as "of Clue?:" or closing punctuation, likely followed by another "or Ethnicity" equivalent phrase per the article pattern, e.g., "or Ethnic Group."
**Std match** (rollout 2, sent 3, marginal +0.723, solo +0.698, verdict CONTRADICTED): Final token "Nationality" is mid-phrase within a section heading "Tooley Surname Country of Origin, Nationality" — the section title from similar genealogy sites typically continues with "& Ethnicity" or "& Origin" before a colon or question mark.
**Std match** (rollout 3, sent 3, marginal +0.094, solo +0.582, verdict CONTRADICTED): Final token "Nationality" is mid-header, part of a section heading that began "Lodden Surname Country of Origin, Nationality" — immediately requires a comma or ampersand followed by another category like "& Ethnicity" or country slug completing the parallel list, then a colon header.

## Case 20: ci=13 rollout=1 item 2 — marginal -0.071

**Source tail** (…'alf (69) of the cases resulted from outbreaks of St. Louis encephalitis (SLE) in Arkansas and Texas. In addition, an epizootic of eastern equine encephalitis (EEE) extending from the Atlantic and Gulf coasts into the upper midwest caused sporadic human cases and a substantial loss of livestock. This')
**True continuation:** ' report summarizes the reported cases of arboviral encephalitis in the United States during 1991 and underscores the continuing need for arbo'

**Matryoshka explanation** (→ = the offending item):

-   [+0.589] This CDC MMWR epidemiological report beginning sentence description
- → [-0.071] "This" introduces weekly surveillance summary methodology or data presentation
-   [+0.055] 1987 measles outbreak statistics prompt narrative continuation or conclusion update
-   [+0.071] Classic CDC MMWR bulletin opening format, epidemiology introduction paragraph
-   [+0.021] "From January through August 1987... reported deaths." This suggests sentence continues "This report discusses..."
-   [+0.020] Historical MMWR format typically follows with bulletin purpose/epidemiologic implications
-   [-0.002] Two specific morbidity statistics just summarized introduce surveillance findings context
-   [+0.009] CDC MMWR genre conventions: epidemiologic note updates annual immunization trends/themes
-   [+0.004] Measles cases decline expected but outbreaks prompt discussion of vaccination coverage gaps
-   [-0.011] "8 measles deaths and 1,083 cases of measles were reported in the United States. This" expects "report

Item verdict: **SUPPORTED** | solo +0.226 | LOO damage +0.005

**Std match** (rollout 0, sent 5, marginal +0.023, solo +0.448, verdict FABRICATED): The preceding "complications...reduced measles resistance among persons born before 1957... This" demands a closing noun phrase describing the report itself, most likely "report summarizes..." or similar.
**Std match** (rollout 1, sent 3, marginal +0.920, solo +0.545, verdict SUPPORTED): "This" is the final token, functioning as a demonstrative pronoun opening a concluding or transitional summary sentence; it directly precedes a noun phrase or verb completing the MMWR "This [season/report/morbidity report] summary/pattern..." construction, likely continuing with "report" followed by a description of the CDC recommendation or purpose (e.g., "note describes").
**Std match** (rollout 2, sent 3, marginal +0.078, solo +0.526, verdict SUPPORTED): Final token "This" is a demonstrative pronoun beginning a concluding/editorial sentence, strongly expecting a noun phrase like "report summarizes..." or "year's data are reported..." or transitioning into typical CDC surveillance concluding boilerplate.
**Std match** (rollout 3, sent 2, marginal +0.433, solo +0.361, verdict FABRICATED): Narrative momentum: the text has introduced the public health problem (rates declining overall but increasing in outbreaks), and is now presenting background evidence — "data from 1983 procedures and vaccine effectiveness. This" signals an imminent continuation explaining the document's purpose or transition to surveillance findings/implications.
