# Gate D — adversarial experiment review

Reviewer scope: PLAN.md, REPORT.md, all pipeline code (extract_d, rebuild_evidence,
ridge_d, ground_score, build_bundles_d, build_arms_d, build_armT2, labeler_d,
evaluate_d, gate_c train/predict_critic), results JSONs, and the prior
code_review_d.md (findings there are not repeated). All empirical numbers below
were recomputed locally from the artifacts in this directory; reproduction
snippets are summarized inline. Headline claims under review:

- **(a)** grounded labels carry beyond-content info about attention writes
  (sign-flip vs Gate C)
- **(b)** effect ordered by label groundedness (T > H > Z > C)
- **(c)** armT > armH shows LLM rewriting is lossy
- **(d)** armT2 < armT shows selectivity beats coverage

Spot-checks run (all CPU, minutes):

| check | result |
|---|---|
| saved residual target reproduces from refit ridge (λ=1e4) | yes, max abs diff 3.7e-7 |
| token-identity-only predictor of resid (per-token train-mean) | cos_resid −0.0006 (null) |
| prev-token bigram-mean predictor | cos_resid +0.0012 (null) |
| armT−armC delta by max retrieval distance | ≤15 tok: **+0.050** · 16–60: +0.032 · >60: **+0.024**; corr(δ, log dist) = −0.18 |
| evidence sources within the 60-token context tail | 90.4% |
| fraction of label words present verbatim in the true tail | armC **0.105**, armT **0.493** |
| armT−armH direct paired contrast (never computed in REPORT) | +0.0072, doc-boot CI [+0.0058, +0.0087] |
| delta by quiet stratum (armT) | quiet +0.033 vs non-quiet +0.031 (not quiet-driven) |
| corr(label length, cos_resid) within arm | armT +0.24, armH +0.16, armC +0.03; shortest arm (armT) wins overall |
| label token budget vs max-len 320 | safe; longest arm ≈ 210 tokens incl. template |

---

## Findings, ranked by threat

### F1. The "content control" contains almost no content — the sign-flip may be a fidelity artifact, not an attention-grounding effect
**Threatens: claim (a). Severity: UNDERMINES (the interpretation; the raw contrast stands).**

armC is a *single* temp-0.4 frozen-AV read of v_pre. Measured against the actual
text: only **10.5%** of armC label words appear in the true 60-token context
tail (and ground_cpre.json's own metric: 8.1% of content words found anywhere in
evidence∪tail). The eval samples confirm heavy confabulation ("GABRIELLA G",
"Red and her obsession with the apocalypse" — none in the source). armT, by
contrast, is ~49% verbatim local text, and **90% of its quoted sources lie
inside the same 60-token tail**.

So the headline contrast is not "attention-grounded label vs content label";
it is "verbatim local text (+ pattern info) vs hallucinated paraphrase of
content". The free control that would separate these — **label = the raw
60-token context tail itself** (perfectly content-grounded, zero attention
information) — was never run, even though it costs nothing to build and ~$2 to
train. Until it exists, "beyond-content" formally means only "beyond a linear
map of v_pre and beyond one noisy AV paraphrase".

The distance gradient is direct evidence against the attention-specific
reading: if the mechanically-grounded part (which sources, which heads) carried
armT's margin, the margin should *grow* when the label contains genuinely
non-local retrievals that no content register could know. It does the
opposite — armT's edge is largest exactly where its label is nothing but a
re-quote of the immediately preceding tokens (+0.050 local-only vs +0.024
distant), corr(δ, log distance) = −0.18.

Partial defense, also established here: the signal is **not** trivial token
memorization — a predictor knowing only the target token (or target bigram)
achieves cos_resid ≈ 0.000. Whatever the critic uses, it needs the quoted
context. That is consistent with *either* attention-evidence value *or* raw
local-text value; only the tail control separates them.

**Cheap settling test:** build `armTail_{train,eval}8k.json` with
`text = context_tail` (the field already exists in meta_d.json), train one
critic (~40 min A100), evaluate with the existing pipeline. If
Δ(tail) ≳ Δ(armT) ≈ +0.031, claim (a) collapses to a fidelity effect; if
Δ(tail) ≈ +0.01, the attention-grounding interpretation survives strongly.
This is the single highest-information dollar available.

### F2. "Ordered by groundedness" contradicts the gate's own groundedness measurements
**Threatens: claim (b). Severity: UNDERMINES.**

By the gate's own metric (ground_score), armC reads are **4× more grounded**
than armZ reads (full overlap 0.081 vs 0.0185) — REPORT itself states this in
the D0.3 section. Yet armZ beats armC by +0.013. And armT2, which is *exactly
as grounded as armT by construction* (deterministic rendering of measured
patterns, future-leak-free), lands at +0.010 — below armZ. So measured
groundedness orders the arms T≈H≈T2 ≫ C > Z, while the critic deltas order
them T > H > Z > T2 ≈ 0 < C-relative. The claimed monotone-in-groundedness
ordering only works if "groundedness" is redefined per-arm to mean "how much
information about the target write the label construction had access to":
T (full pattern transcript) > H (rewritten transcript) > Z (AV-filtered glimpse
of the injected target) > C (no target access). Under that honest relabeling,
claim (b) becomes close to a tautology — labels built with more access to the
answer train better critics — and stops being evidence about verbalization
registers. Recommend dropping the word "groundedness" from this claim and
stating the ordering as label-information ordering.

### F3. Where "verbalizable information" ends and "copying the answer key" begins
**Threatens: claim (a) framing. Severity: WEAKENS (and should reframe the writeup).**

attn_out(p) = Σ_h W_O^h Σ_j a_hj·v_h(j), and v_h(j) is a function of the
residual state at source j — largely recoverable from the quoted source tokens
in their local windows. armT/armH are therefore textual transcripts of the
write's *generative recipe*; the critic's task reduces to re-embedding quoted
spans and approximating the OV circuit. Is that a flaw? **Within the
pre-registered question, no** — Phase D1 explicitly set out to manufacture
mechanically-grounded labels and test whether a critic can use them; the
labels were always going to be functions of the pattern. The verdict logic
(MARGINAL, no D3/RL funding) is sound.

But two glosses in REPORT overreach:

1. "Grounded labels carry beyond-content information" reads as progress toward
   verbalizing attention writes. What was measured is an **upper bound on the
   text-label channel**: even when the answer's recipe is spelled into the
   label, a 7B critic recovers only cos 0.072 of the residual. The only arm
   whose labels could ever be produced *by a model reading the activation*
   (armZ) scored +0.013, FAIL. The honest one-liner is: "even answer-key
   labels barely move the needle; model-producible labels don't."
2. "Beyond-content" is ill-posed at the limit: attn_out is a deterministic
   function of the document prefix, so a sufficiently verbatim label predicts
   it perfectly with a strong enough critic. The ridge nuisance conditions
   only on v_pre at p (linear, single position); all nonlinear content
   predictability and all other-position content remains in the "residual"
   and is creditable to any label that quotes text. The gate de facto measures
   "which ~300-char compression of the prefix best helps the critic" — a real
   and interesting quantity, but a different one from the claim's wording.

### F4. armT2 < armT does not support "selectivity beats coverage" — too many simultaneous changes, and the proposed mechanism predicts the wrong sign of the distance gradient
**Threatens: claim (d). Severity: UNDERMINES.**

armT2 changed at least six things at once relative to armT (build_armT2.py):
all-head contribution-weighted aggregation; **local sources stripped of their
verbatim context windows down to bare token lists** ("carries over the local
phrase around ',', 'really', …" vs armT's full quoted windows); distant-first
ordering; global dedup by token string; strength wording; ±4 windows. Given F1
(armT's margin concentrates in positions where the label is verbatim local
windows), the most parsimonious explanation for armT2's collapse is the loss
of those verbatim windows — i.e., a *rendering* regression, not an
*information-selection* lesson. REPORT's stated mechanism ("local/self-attention
mass… drowning the selective heads whose contextual phrases carried armT's
signal") implies armT's signal lives in selective/distant contextual phrases —
which predicts δ increasing with retrieval distance; the measured gradient is
the opposite (−0.18). **Cheap test:** one single-change ablation — armT with
context windows removed (tokens only). If it drops to ≈ armT2's +0.010, the
carrier was the verbatim windows and claim (d) should be retracted as stated.

### F5. No training-seed replication: every between-arm ordering rests on one SGD run per arm
**Threatens: claims (b), (c), (d). Severity: WEAKENS.**

The doc-clustered bootstrap CIs condition on the five trained checkpoints; they
quantify position-sampling noise only. Run-to-run variance of a 7B fine-tune on
8k pairs (data order, dropout, nondeterministic kernels) plausibly shifts mean
cos_resid by several thousandths — the same order as the T−H gap (+0.0072) and
a nontrivial fraction of T−T2 (+0.021). The tight CIs in eval_d_results.json
therefore overstate the reliability of the *between-arm* orderings (the
T-vs-C sign at +0.031 is presumably safe; the fine structure is not).
**Cheap test:** retrain armT and armH once each with a different shuffle seed
(~$4 total) and report the seed spread before asserting (c)/(d).

### F6. Claim (c)'s mechanism ("LLM rewriting is lossy") is under-identified
**Threatens: claim (c). Severity: CAVEAT.**

The direct paired armT−armH contrast was never computed in the pipeline
(contrasts are only vs armC); computed here it is +0.0072 [+0.0058, +0.0087],
so the difference is real conditional on the checkpoints (modulo F5). But armH
differs from armT in more than rewriting: banned vocabulary list, numeric
weights replaced by 3-bin strength words, banned openers, a regen pass, longer
labels (458 vs 238 chars), and access to the full context tail (it can *add*
content armT lacks). "Rewriting is lossy" is one of several live explanations;
"the prompt's strength-word quantization discards the weights" is another.
State the contrast, not the mechanism.

### F7. Pre-registration deviations in the stats (immaterial numerically, worth recording)
**Threatens: claim (a) hygiene. Severity: CAVEAT.**

- Holm family was pre-registered as "up to 3 arm contrasts" (T, H, Z); armT2
  was added to the same family post hoc (4 contrasts in evaluate_d.py). With
  all bootstrap p < 1e-4 this changes nothing here, but it is a family-size
  deviation and should be footnoted.
- p_one_sided is reported as exactly 0.0; a 10k bootstrap can only certify
  p < 1e-4. Report "<1e-4".
- evaluate_d's MARGINAL verdict branch checks only the point estimate
  (`pt >= 0.02`) and does not require the CI to exclude 0 — moot at these n,
  but the verdict logic is laxer than the pre-registered text reads.
- The cluster bootstrap itself (ratio-of-sums over doc resamples) is correct.

### F8. armZ selection effects and cross-arm coupling
**Threatens: claim (b) (the Z slot). Severity: CAVEAT.**

armZ eval keeps 2,893/3,318 holdout positions; the ~13% dropped are CJK/garble
or unparsed generations — i.e., injection failures, plausibly correlated with
atypical write directions. The Z contrast is therefore computed on a
non-random subset. Direction unknown; with Δ_Z = +0.013 the claim "Z beats C"
is robust-ish but the magnitude is not comparable to the other arms'. Also,
build_arms_d takes the intersection of clean positions across all arms for the
shared train set, so armZ's failure modes silently shaped every arm's training
distribution (mild, but it means "matched n" is matched on a Z-censored set).

### F9. D0.3 GO is a ratio of minuscule numbers
**Threatens: REPORT's D0.3 paragraph. Severity: CAVEAT.**

"Grounded ~3× over a shuffled null" = 1.9% vs 0.6% of content words; the
decisive source-specific clause passed at an absolute margin of **+0.0014**
(0.14% of content words). The prior code review flagged the missing effect
floor; what it did not say is that the REPORT's "3×" framing actively
flatters a result that, stated absolutely, would have argued for the
soft-fail branch (drop armZ). The verdict didn't hinge on it, but future
gates should pre-register absolute floors, not just CI>0.

### F10. The ridge nuisance is weak by construction, so "residual alignment" absolutes are inflated
**Threatens: claim (a) magnitude. Severity: CAVEAT.**

λ grid is decade-spaced (1e2–1e6); the nuisance is linear, single-position,
single-layer. Everything nonlinearly predictable from v_pre, and everything
predictable from *other* positions' content, stays in the "content-
unpredictable" target and is creditable to any text label. (Verified the
saved residuals match a refit exactly; R² = 0.401 sits just over the 0.40
GO/CAUTION boundary and the more conservative CAUTION branch was correctly
taken.) The right nuisance for the question actually being asked is a
*text-register* nuisance — e.g., residualize against the verbatim-tail critic
of F1 — after which "beyond-content" would mean beyond what any honest content
register achieves. Expect all five cos_resid numbers to shrink; the T-vs-tail
margin is then the entire result.

### F11. Smaller observations (no verdict impact found)
**Severity: IMPROVEMENT.**

- Label length is not a confound for the headline: the *shortest* arm (armT,
  238 chars) wins, and all arms fit in max-len 320 with ~2× headroom (full
  armH batch checked, not just the pilot). Within armT, corr(length,
  cos_resid) = +0.24 — longer label = more quoted sources = more information;
  expected, not pathological.
- Quiet stratum does not drive armT (δ quiet +0.033 vs non-quiet +0.031);
  armH/armZ do better on quiet positions (+0.040/+0.035, n=302/257) —
  consistent with quiet-template positions being partially predictable from
  the shared "weak write" phrasing, worth one sentence in the report.
- cos_beyondC for armC is 1e-11 (exact-by-construction sanity) — good.
- FineWeb near-duplicate documents can straddle the doc-level split; with a
  fresh seed-2 slice this mostly inflates absolute cos slightly and is shared
  across arms. Unquantified; a dedup pass (minhash on doc text) would close it.
- Extraction (eager attention, GQA per-query-head decomposition through W_O
  column slices, hidden_states[20] indexing) was re-examined and is correct;
  the per-doc cos>0.999 reconstruction assert is the right invariant. Nothing
  new beyond code_review_d.md's clean bill.

---

## The 3 changes I'd make in a follow-up gate, ranked

1. **Add the two factoring controls and make the verbatim-tail arm the
   baseline.** (i) `armTail`: label = raw 60-token context tail (content
   fidelity, zero pattern info); (ii) `armPattern`: label = source *offsets and
   weights only*, no quoted text (pattern info, zero content). Together they
   decompose armT's margin into fidelity vs mechanism — the question this gate
   thought it answered. Both are free to build and ~$2 each to train; armTail
   alone settles whether Gate D's sign-flip exists at all. Then redefine the
   primary metric as Δ vs armTail and, ideally, residualize against the
   armTail critic's predictions rather than (only) the linear v_pre ridge.
2. **Replace the armT2-style multi-change "improvement cycle" with a
   single-change ablation chain on armT** (windows→token-lists, top-head-only,
   distant-only, local-only), each trained identically. This converts
   unfalsifiable narrative ("selectivity beats coverage") into attributable
   deltas, and the local-only/distant-only pair directly tests the distance
   gradient found here.
3. **Quantify run noise and test orderings directly:** ≥2 training seeds per
   arm (report seed SD next to every contrast), direct paired contrasts
   between treatment arms (not just vs baseline) inside the Holm family fixed
   at pre-registration time, bootstrap p reported as bounds, and pre-registered
   absolute effect floors for any groundedness-style gate.
