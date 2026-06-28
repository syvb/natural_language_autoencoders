import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    L, F = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = list(r)[0]
            L.append(int(r[k])); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


tok_rl, f_tok_rl = load("sweep_out/token_fve.csv")
tok_ct, f_tok_ct = load("sweep_out/token_fve_kitft.csv")
lin_rl, f_lin_rl = load("sweep_out/lines_fve.csv")
lin_ct, f_lin_ct = load("sweep_out/lines_fve_kitft.csv")
RL_C = "#1f77b4"; CT_C = "#888888"
FULL_RL, FULL_CT = 0.705, 0.727

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# --- row 0: linear FVE ---
a = axes[0, 0]
a.plot(tok_rl, f_tok_rl, color=RL_C, lw=1.8, label="RLed truncation (kl0.01/iter200)")
a.plot(tok_ct, f_tok_ct, color=CT_C, lw=1.8, ls="-", label="published kitft (control)")
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (content tokens)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs token-truncation length"); a.grid(alpha=.3); a.legend(loc="lower right")

a = axes[0, 1]
a.plot(lin_rl, f_lin_rl, marker="o", ms=4, color=RL_C, lw=1.8, label="RLed truncation")
a.plot(lin_ct, f_lin_ct, marker="s", ms=4, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (lines)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs line-truncation length"); a.grid(alpha=.3); a.legend(loc="lower right")

# --- row 1: log-log unexplained variance (1-FVE) ---
a = axes[1, 0]
a.loglog(tok_rl, 1 - f_tok_rl, color=RL_C, lw=1.8, label="RLed truncation")
a.loglog(tok_ct, 1 - f_tok_ct, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(1 - FULL_RL, color=RL_C, ls="--", lw=1, alpha=.7, label=f"RLed floor {1-FULL_RL:.3f}")
a.axhline(1 - FULL_CT, color=CT_C, ls="--", lw=1, alpha=.7, label=f"control floor {1-FULL_CT:.3f}")
a.set_xlabel("truncation length (content tokens, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (tokens)"); a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)

a = axes[1, 1]
a.loglog(lin_rl, 1 - f_lin_rl, marker="o", ms=4, color=RL_C, lw=1.8, label="RLed truncation")
a.loglog(lin_ct, 1 - f_lin_ct, marker="s", ms=4, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(1 - FULL_RL, color=RL_C, ls="--", lw=1, alpha=.7)
a.axhline(1 - FULL_CT, color=CT_C, ls="--", lw=1, alpha=.7)
a.set_xlabel("truncation length (lines, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (lines)"); a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)

fig.suptitle("Round-trip FVE vs AV-explanation truncation — RLed truncation model vs published kitft control (N=100, same docs)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig("sweep_out/fve_compare_rl_vs_control.png", dpi=130, bbox_inches="tight")


# tokens-to-threshold summary
def reach(L, F, thr):
    for li, fi in zip(L, F):
        if fi >= thr:
            return li
    return None


for name, L, F in [("RLed", tok_rl, f_tok_rl), ("control", tok_ct, f_tok_ct)]:
    print(f"{name}: tokens to FVE>=0: {reach(L,F,0.0)}  >=0.5: {reach(L,F,0.5)}  >=0.6: {reach(L,F,0.6)}  full~{F[-1]:.3f}")
print("WROTE sweep_out/fve_compare_rl_vs_control.png")
