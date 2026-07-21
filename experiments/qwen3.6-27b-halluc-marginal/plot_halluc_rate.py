"""Hallucination RATE, matryoshka vs standard, under progressively stricter
definitions of "hallucination":
  corpus (loose)          — single judge vs the corpus continuation (CON/FAB)
  corpus (2-model strict)  — both nex + gpt-4o-mini confirm under strict prompt
  model-aware (new)        — single judge that ALSO sees the model's own generation
                             (an item that merely described the model's output is
                             no longer counted); appears once the re-judge is done.
Rates are item-level (denominator = all judged items of that model)."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
H = {"CONTRADICTED", "FABRICATED"}
MODELS = ["nex-agi/nex-n2-mini", "openai/gpt-4o-mini"]
faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
strict = json.load(open(HERE / "results" / "strict_halluc.json")) if (HERE / "results" / "strict_halluc.json").exists() else {}
withgen = json.load(open(HERE / "results" / "withgen_verdicts.json")) if (HERE / "results" / "withgen_verdicts.json").exists() else None


def keys(arm):
    return [k for k in faith if k.startswith(f"{arm}|") and faith[k] is not None]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0, c - h), min(1, c + h)


conds = [("corpus\n(loose)", "#e34948")]
conds.append(("corpus\n(2-model strict)", "#8a6d3b"))
if withgen is not None:
    conds.append(("model-aware\n(sees model output)", "#2a78d6"))

fig, ax = plt.subplots(figsize=(8.4, 5.4))
models = [("matryoshka", "mat"), ("standard", "std")]
x = np.arange(len(models))
w = 0.8 / len(conds)
print("hallucination rate (item-level):")
for ci, (cname, col) in enumerate(conds):
    rates, los, his, labels = [], [], [], []
    for _, arm in models:
        ks = keys(arm); n = len(ks)
        if ci == 0:
            khall = sum(faith[k] in H for k in ks)
        elif ci == 1:
            khall = sum(1 for k in ks if faith[k] in H and k in strict and all(strict[k].values()))
        else:
            khall = sum(1 for k in ks if faith[k] in H and withgen.get(k) in H)
        p, lo, hi = wilson(khall, n)
        rates.append(p); los.append(p - lo); his.append(hi - p); labels.append(f"{p*100:.0f}%")
        print(f"  {arm} {cname.replace(chr(10),' ')}: {khall}/{n} = {p*100:.1f}%")
    bars = ax.bar(x + (ci - (len(conds) - 1) / 2) * w, rates, w, yerr=[los, his],
                  capsize=3, color=col, label=cname.replace("\n", " "))
    for xi, (r, lab) in enumerate(zip(rates, labels)):
        ax.text(x[xi] + (ci - (len(conds) - 1) / 2) * w, r + his[xi] + 0.012, lab,
                ha="center", fontsize=9, color=col, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([m[0] for m in models], fontsize=12)
ax.set_ylabel("hallucination rate (fraction of items)", fontsize=11)
ax.set_ylim(0, 0.70)
ax.set_title("Hallucination rate: matryoshka vs standard NLA\n(under progressively stricter definitions)", fontsize=12.5)
ax.legend(fontsize=9.5, loc="upper center", title="definition", ncol=3)
ax.grid(axis="y", color="#ccc", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
foot = ("Item = line (matryoshka, ~10/expl) vs sentence (standard, ~4/expl); rates are per-item. "
        "Loose = single judge vs corpus continuation.\n2-model strict = nex + gpt-4o-mini both confirm a "
        "concrete fabrication (§3f). Error bars: Wilson 95%.")
if withgen is None:
    foot += "  [model-aware bar pending the GPU generation re-judge.]"
fig.subplots_adjust(bottom=0.20)
fig.text(0.02, 0.02, foot, fontsize=7.8, color="#777", ha="left", va="bottom", wrap=True)
fig.savefig(HERE / "results" / "fig_halluc_rate.png", dpi=150)
print("[saved] results/fig_halluc_rate.png")
