"""Plot the near-range persona sweep from results/persona_near_agg.csv (local CPU)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
rows = list(csv.DictReader(open(os.path.join(R, "persona_near_agg.csv"))))
traits = ["sycophancy", "sadness", "yellow", "tofu"]
col = {"sycophancy": "#1f77b4", "sadness": "#9467bd", "yellow": "#e8b800", "tofu": "#2ca02c"}


def series(tr, key):
    xs, ys = [], []
    for r in rows:
        if r["trait"] == tr:
            xs.append(int(r["strength"]))
            v = r[key]; ys.append(float(v) if v not in ("", "nan") else np.nan)
    return np.array(xs), np.array(ys)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for tr in traits:
    x, y = series(tr, "appearance_rate")
    a1.plot(x, y, marker="o", lw=2, color=col[tr], label=tr)
a1.set_xlabel("system-prompt strength (0=neutral … 5=extreme)")
a1.set_ylabel("fraction of samples mentioning the trait")
a1.set_title("Trait appearance rate vs prompt strength\n(near range: persona + short neutral fragment, N=24)")
a1.grid(alpha=.3); a1.legend(); a1.set_ylim(-0.02, 1.0)
for tr in traits:
    x, y = series(tr, "mean_rank")
    if np.isfinite(y).any():
        a2.plot(x, y, marker="o", lw=2, color=col[tr], label=tr)
a2.set_xlabel("system-prompt strength (0=neutral … 5=extreme)")
a2.set_ylabel("mean rank of trait line (1 = top of list)")
a2.set_title("When it appears, how high in the list?\n(lower = higher up)")
a2.grid(alpha=.3); a2.legend(); a2.invert_yaxis()
fig.suptitle("Persona-probe: does a system-prompt persona surface in the AV's reading of the residual stream?", y=1.02, fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(R, "persona_near.png"), dpi=140, bbox_inches="tight")
print("wrote", os.path.join(R, "persona_near.png"))
