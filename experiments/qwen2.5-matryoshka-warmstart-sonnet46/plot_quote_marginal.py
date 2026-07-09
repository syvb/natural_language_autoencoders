"""Marginal FVE per token, in-quote vs out-of-quote.

Left: mean per-token ΔFVE vs token index (5-index rolling mean), split by class,
v3 RL pair. Right: summary — mean ΔFVE per token by class for v3 RL and v3
warm-start, annotated with each class's share of tokens vs share of total FVE.
Reads v3_faithfulness_results/quote_marginal_v3{rl,ws}.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "v3_faithfulness_results")
rl = json.load(open(f"{D}/quote_marginal_v3rl.json"))
ws = json.load(open(f"{D}/quote_marginal_v3ws.json"))
XCAP = 120
WIN = 5


def index_curves(rows):
    """mean ΔFVE at each token index, split by in-quote."""
    byq, byn = {}, {}
    for r in rows:
        for t, (d, q) in enumerate(zip(r["delta_fve"], r["in_quote"])):
            (byq if q else byn).setdefault(t, []).append(d)
    def mean_curve(by):
        ks = sorted(k for k in by if k < XCAP)
        return ks, [float(np.mean(by[k])) for k in ks]
    return mean_curve(byq), mean_curve(byn)


def smooth(y, w=WIN):
    return np.convolve(y, np.ones(w) / w, mode="same")


def summary(rows):
    q = [d for r in rows for d, iq in zip(r["delta_fve"], r["in_quote"]) if iq]
    n = [d for r in rows for d, iq in zip(r["delta_fve"], r["in_quote"]) if not iq]
    return (np.mean(q), np.mean(n), len(q) / (len(q) + len(n)),
            sum(q) / (sum(q) + sum(n)))


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.4, 1]})

(qk, qv), (nk, nv) = index_curves(rl)
a1.plot(qk, smooth(qv), color="#1f77b4", lw=2.2, label="in-quote tokens")
a1.plot(nk, smooth(nv), color="#d62728", lw=2.2, label="out-of-quote tokens")
a1.axhline(0, color="k", lw=.8, alpha=.5)
a1.set_xlim(0, XCAP)
a1.set_xlabel("token index in AV explanation")
a1.set_ylabel(f"mean marginal ΔFVE per token ({WIN}-index rolling mean)")
a1.set_title("v3 RL — marginal FVE per token, by quote membership")
a1.grid(alpha=.3); a1.legend(fontsize=10)

labels, qm, nm, notes = [], [], [], []
for name, rows in (("v3 RL", rl), ("v3 warm-start", ws)):
    mq, mn, tokshare, fveshare = summary(rows)
    labels.append(name); qm.append(mq); nm.append(mn)
    notes.append(f"{tokshare:.0%} of tokens → {fveshare:.0%} of FVE")
x = np.arange(len(labels))
a2.bar(x - 0.18, qm, 0.36, color="#1f77b4", label="in-quote")
a2.bar(x + 0.18, nm, 0.36, color="#d62728", label="out-of-quote")
for i, (a, b, note) in enumerate(zip(qm, nm, notes)):
    a2.text(i - 0.18, a, f"{a:.4f}", ha="center", va="bottom", fontsize=9)
    a2.text(i + 0.18, b, f"{b:.4f}", ha="center", va="bottom", fontsize=9)
    a2.text(i, max(a, b) * 1.18, f"quotes: {note}", ha="center", fontsize=9, color="#1f77b4")
a2.set_xticks(x); a2.set_xticklabels(labels)
a2.set_ylabel("mean marginal ΔFVE per token")
a2.set_title("Per-token value by class")
a2.grid(alpha=.3, axis="y"); a2.legend(fontsize=10)

fig.suptitle("Where the FVE comes from: marginal contribution of in-quote vs out-of-quote tokens", y=1.02, fontsize=13)
fig.tight_layout()
out = f"{D}/fig_quote_marginal.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
for name, rows in (("v3 RL", rl), ("v3 warm-start", ws)):
    mq, mn, tokshare, fveshare = summary(rows)
    print(f"{name}: in-quote mean ΔFVE {mq:.5f} vs out {mn:.5f} (ratio {mq/mn if mn else float('nan'):.1f}x); "
          f"quote tokens {tokshare:.1%} of tokens, {fveshare:.1%} of FVE")
