# Is the RL helping? Matryoshka RL AV vs its warm-start (pre-RL) AV

Same MATRYOSHKA model, same base, same 5,275 positions (honeypot + 4 plain
controls), same gpt-4o-mini generic eval-awareness judge. Only the adapters
differ: RL (`rl_av_lora_iter400` + `rl_critic_step400`) vs warm-start
(`warmstart_av_lora` + `warmstart_ar_critic`). `warmstart_gen.py` regenerated
the warm-start explanations + FVE on one H200 (~1.7h); `rl_vs_warmstart.py`
does the comparison. Figure: `rl_vs_warmstart.png`.

## Eval-awareness detection — RL nearly DOUBLES it (the primary result)

Judge reads only the AV's explanation text (no critic), so this cleanly
isolates the AV RL. Signal detection, honeypot = positives, plain = negatives:

| | plain mean | honeypot mean | **ROC AUC** | hit@FPR1% | hit@FPR5% |
|---|---|---|---|---|---|
| **RL** | 0.089 | 0.282 | **0.726** | 6.2% | 20.2% |
| **warm-start** | 0.055 | 0.100 | **0.520** | 4.0% | 9.9% |

**ΔAUC = +0.206.** The warm-start AV is at AUC 0.52 — essentially chance; its
honeypot mean (0.100) barely exceeds its plain mean (0.055). RL raises honeypot
detection to 0.282 while plain rises only to 0.089 — genuine separation, not a
baseline shift. It is NOT a verbosity artifact: explanation length is
comparable (RL ~907 vs warm-start ~716 chars on the honeypot) and the AUC
metric would be flat if the gain were bias/verbosity (plain FPR would rise in
lockstep). **The eval-awareness signal in the matryoshka explanations is
substantially produced by the RL, not the SFT warm-start.**

Mechanistically: warm-start explanations stay at the token/syntax level ("'by'
signals author attribution", "truncated mid-timestamp", "From: field indicating
sender") — they describe surface form, not the situational "this is a
test/roleplay" framing the RL AV surfaces.

## Front-loading — almost entirely an RL creation (bonus; end-to-end AV+critic)

This compares the full pipelines (both adapters), so it's the fair "does the RL
help" answer but reflects AV+critic RL jointly, not AV alone:

| | line-1-alone FVE | line-1 useful alone | full-expl FVE | FVE @25 words |
|---|---|---|---|---|
| **RL** | +0.294 | 93.6% | 0.503 | 0.405 |
| **warm-start** | +0.006 | 53.2% | 0.153 | 0.085 |

The warm-start AV+critic barely reconstruct the activation at all (full FVE
0.15); RL more than triples it to 0.50. The "buried lede" property — line 1
being a usable standalone summary — is an RL creation: warm-start's first line
is useful alone only 53% of the time (a coin flip), RL's 94%.

## Bottom line

The RL is doing the heavy lifting on both axes we care about. The SFT warm-start
is a weak starting point — near-chance at revealing eval-awareness and at
front-loaded reconstruction — and the RL is what turns the matryoshka NLA into
a useful, salience-ordered, eval-awareness-revealing explainer.

Caveat: eval-awareness cleanly attributes to the AV RL (judge = text only); the
front-loading numbers are the combined AV+critic RL effect and don't separate
the two. Seeds were not matched per-position (population-level comparison at
N=5,275; sampling noise averages out — the AUC/base-rate gaps are far larger
than any per-sample seed variance).
