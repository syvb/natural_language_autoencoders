"""Figures for eval_paraphrase_order.py results.

Left: v3 RL pair — FVE vs prefix length (lines) under: model order, paraphrased,
greedy-oracle order, random order, reversed order, cross-sample floor.
Right: paraphrase retention — orig vs paraphrased curves for the RL pair and the
pre-RL warm-start pair (co-adaptation control: if RL taught a private code, the
RL pair's retention should be much lower than the warm-start pair's).

Reads v3_faithfulness_results/para_order_v3rl.json + para_order_v3ws.json.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "v3_faithfulness_results")
rl = json.load(open(f"{D}/para_order_v3rl.json"))["results"]
ws = json.load(open(f"{D}/para_order_v3ws.json"))["results"]
K = 10


def curve(res, cond):
    c = res[cond]
    ks = [k for k in range(1, K + 1) if str(k) in c or k in c]
    return ks, [c.get(str(k), c.get(k)) for k in ks]


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4))

STYLES = [("oracle", "greedy-oracle order (own lines)", "#2ca02c", "--", "*"),
          ("orig", "model order", "#9467bd", "-", "o"),
          ("para", "paraphrased (line-wise)", "#1f77b4", "-", "^"),
          ("random", "random order (mean of 3)", "#7f7f7f", ":", "s"),
          ("reversed", "reversed order", "#ff7f0e", ":", "v"),
          ("crossfloor", "lines from OTHER samples (floor)", "#d62728", "-.", "x")]
for cond, lab, col, ls, mk in STYLES:
    if cond not in rl:
        continue
    ks, vs = curve(rl, cond)
    full = rl[cond].get("full")
    suffix = f" (full={full:.3f})" if full is not None else ""
    a1.plot(ks, vs, marker=mk, ms=5, color=col, ls=ls, lw=2, label=f"{lab}{suffix}")
a1.axhline(0, color="k", lw=.6, alpha=.4)
a1.set_xlabel("prefix length (list items)"); a1.set_ylabel("round-trip FVE")
a1.set_title("v3 RL pair — ordering & paraphrase conditions")
a1.grid(alpha=.3); a1.legend(fontsize=8.5, loc="lower right")

for res, name, col in ((rl, "v3 RL", "#9467bd"), (ws, "v3 warm-start", "#ff7f0e")):
    for cond, ls, mk in (("orig", "-", "o"), ("para", "--", "^")):
        ks, vs = curve(res, cond)
        a2.plot(ks, vs, marker=mk, ms=5, color=col, ls=ls, lw=2,
                label=f"{name} · {'orig' if cond=='orig' else 'paraphrased'} (full={res[cond]['full']:.3f})")
a2.axhline(0, color="k", lw=.6, alpha=.4)
a2.set_xlabel("prefix length (list items)"); a2.set_ylabel("round-trip FVE")
rr = rl["para"]["full"] / rl["orig"]["full"]
wr = ws["para"]["full"] / ws["orig"]["full"]
a2.set_title(f"Paraphrase retention: RL {rr:.2f} vs warm-start {wr:.2f}\n(similar ⇒ semantics; RL≪WS ⇒ co-adapted code)")
a2.grid(alpha=.3); a2.legend(fontsize=8.5, loc="lower right")

fig.suptitle("v3 NLA faithfulness: paraphrase robustness + order optimality (100 held-out)", y=1.02, fontsize=13)
fig.tight_layout()
out = f"{D}/fig_paraphrase_order.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
print(f"paraphrase retention: RL {rr:.3f}  WS {wr:.3f}")
if "oracle" in rl:
    for k in (1, 3, 5):
        o = rl["oracle"].get(str(k), rl["oracle"].get(k))
        m = rl["orig"].get(str(k), rl["orig"].get(k))
        r = rl["random"].get(str(k), rl["random"].get(k))
        print(f"k={k}: oracle-gap captured = ({m:.3f}-{r:.3f})/({o:.3f}-{r:.3f}) = {(m-r)/(o-r+1e-9):.2f}")
