import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    L, F = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            L.append(int(r[list(r)[0]])); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


Lt, Ft = load("sweep_out/token_fve.csv")
Lk, Fk = load("sweep_out/lines_fve.csv")
FULL = 0.705
# residual / unexplained-variance fraction (positive; the "reconstruction error")
Rt, Rk = 1 - Ft, 1 - Fk

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.loglog(Lt, Rt, lw=1.6, color="#1f77b4")
a1.axhline(1 - FULL, ls="--", color="gray", lw=1, label=f"full-len floor 1−FVE={1-FULL:.3f}")
a1.set_xlabel("AV truncation length (content tokens, log)")
a1.set_ylabel("unexplained variance  1 − FVE  (log)")
a1.set_title("Reconstruction error vs token-truncation length (N=100)")
a1.grid(which="both", alpha=.3); a1.legend()

a2.loglog(Lk, Rk, marker="o", ms=4, lw=1.6, color="#d62728")
a2.axhline(1 - FULL, ls="--", color="gray", lw=1, label=f"full-len floor 1−FVE={1-FULL:.3f}")
a2.set_xlabel("AV truncation length (lines, log)")
a2.set_ylabel("unexplained variance  1 − FVE  (log)")
a2.set_title("Reconstruction error vs line-truncation length (N=100)")
a2.grid(which="both", alpha=.3); a2.legend()

fig.suptitle("kl0.01/iter_0000200 — log-log reconstruction error (1−FVE) vs AV-explanation truncation", y=1.02)
fig.tight_layout()
fig.savefig("sweep_out/fve_vs_truncation_loglog.png", dpi=130, bbox_inches="tight")

# quick power-law fit on the rising (pre-plateau) token region: L in [2, 60]
m = (Lt >= 2) & (Lt <= 60)
slope, intercept = np.polyfit(np.log(Lt[m]), np.log(Rt[m]), 1)
print(f"token region L=2..60: 1-FVE ~ L^{slope:.3f}  (R-floor at {1-FULL:.3f})")
print("WROTE sweep_out/fve_vs_truncation_loglog.png")
