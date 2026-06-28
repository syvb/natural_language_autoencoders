import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XCAP = 150


def load(path, cap=None):
    L, F = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = list(r)[0]
            li = int(r[k])
            if cap is not None and li > cap:
                continue
            L.append(li); F.append(float(r["fve"]))
    return np.array(L), np.array(F)


tok_rl, f_tok_rl = load("sweep_out/token_fve.csv", XCAP)
tok_ct, f_tok_ct = load("sweep_out/token_fve_kitft.csv", XCAP)
lin_rl, f_lin_rl = load("sweep_out/lines_fve.csv")
lin_ct, f_lin_ct = load("sweep_out/lines_fve_kitft.csv")
RL_C = "#1f77b4"; CT_C = "#888888"; RED = "#d62728"
FULL_RL, FULL_CT = 0.705, 0.727

# ---------- 1) RLed-only linear ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.plot(tok_rl, f_tok_rl, lw=1.6, color=RL_C)
a1.axhline(FULL_RL, ls="--", color="gray", lw=1, label=f"full-len FVE={FULL_RL:.3f}")
a1.set_xlabel("AV explanation truncation length (content tokens)"); a1.set_ylabel("round-trip FVE")
a1.set_title("FVE vs token-truncation length (N=100)"); a1.set_xlim(0, XCAP); a1.grid(alpha=.3); a1.legend()
a2.plot(lin_rl, f_lin_rl, marker="o", ms=4, lw=1.6, color=RED)
a2.axhline(FULL_RL, ls="--", color="gray", lw=1, label=f"full-len FVE={FULL_RL:.3f}")
a2.set_xlabel("AV explanation truncation length (lines)"); a2.set_ylabel("round-trip FVE")
a2.set_title("FVE vs line-truncation length (N=100)"); a2.grid(alpha=.3); a2.legend()
fig.suptitle("kl0.01/iter_0000200 — round-trip FVE vs AV-explanation truncation", y=1.02)
fig.tight_layout(); fig.savefig("sweep_out/fve_vs_truncation.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- 2) RLed-only log-log ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
a1.loglog(tok_rl, 1 - f_tok_rl, lw=1.6, color=RL_C)
a1.axhline(1 - FULL_RL, ls="--", color="gray", lw=1, label=f"full-len floor 1−FVE={1-FULL_RL:.3f}")
a1.set_xlabel("AV truncation length (content tokens, log)"); a1.set_ylabel("unexplained variance 1−FVE (log)")
a1.set_title("Reconstruction error vs token-truncation length (N=100)"); a1.set_xlim(right=XCAP)
a1.grid(which="both", alpha=.3); a1.legend()
a2.loglog(lin_rl, 1 - f_lin_rl, marker="o", ms=4, lw=1.6, color=RED)
a2.axhline(1 - FULL_RL, ls="--", color="gray", lw=1, label=f"full-len floor 1−FVE={1-FULL_RL:.3f}")
a2.set_xlabel("AV truncation length (lines, log)"); a2.set_ylabel("unexplained variance 1−FVE (log)")
a2.set_title("Reconstruction error vs line-truncation length (N=100)"); a2.grid(which="both", alpha=.3); a2.legend()
fig.suptitle("kl0.01/iter_0000200 — log-log reconstruction error (1−FVE) vs AV-explanation truncation", y=1.02)
fig.tight_layout(); fig.savefig("sweep_out/fve_vs_truncation_loglog.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---------- 3) comparison RLed vs control ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
a = axes[0, 0]
a.plot(tok_rl, f_tok_rl, color=RL_C, lw=1.8, label="RLed truncation (kl0.01/iter200)")
a.plot(tok_ct, f_tok_ct, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (content tokens)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs token-truncation length"); a.set_xlim(0, XCAP); a.grid(alpha=.3); a.legend(loc="lower right")
a = axes[0, 1]
a.plot(lin_rl, f_lin_rl, marker="o", ms=4, color=RL_C, lw=1.8, label="RLed truncation")
a.plot(lin_ct, f_lin_ct, marker="s", ms=4, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(0, color="k", lw=.6, alpha=.4); a.axhline(0.5, color="green", lw=.6, ls=":", alpha=.6)
a.set_xlabel("truncation length (lines)"); a.set_ylabel("round-trip FVE")
a.set_title("FVE vs line-truncation length"); a.grid(alpha=.3); a.legend(loc="lower right")
a = axes[1, 0]
a.loglog(tok_rl, 1 - f_tok_rl, color=RL_C, lw=1.8, label="RLed truncation")
a.loglog(tok_ct, 1 - f_tok_ct, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(1 - FULL_RL, color=RL_C, ls="--", lw=1, alpha=.7, label=f"RLed floor {1-FULL_RL:.3f}")
a.axhline(1 - FULL_CT, color=CT_C, ls="--", lw=1, alpha=.7, label=f"control floor {1-FULL_CT:.3f}")
a.set_xlabel("truncation length (content tokens, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (tokens)"); a.set_xlim(right=XCAP)
a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)
a = axes[1, 1]
a.loglog(lin_rl, 1 - f_lin_rl, marker="o", ms=4, color=RL_C, lw=1.8, label="RLed truncation")
a.loglog(lin_ct, 1 - f_lin_ct, marker="s", ms=4, color=CT_C, lw=1.8, label="published kitft (control)")
a.axhline(1 - FULL_RL, color=RL_C, ls="--", lw=1, alpha=.7)
a.axhline(1 - FULL_CT, color=CT_C, ls="--", lw=1, alpha=.7)
a.set_xlabel("truncation length (lines, log)"); a.set_ylabel("unexplained variance 1−FVE (log)")
a.set_title("log-log reconstruction error (lines)"); a.grid(which="both", alpha=.3); a.legend(loc="lower left", fontsize=8)
fig.suptitle("Round-trip FVE vs AV-explanation truncation — RLed truncation model vs published kitft control (N=100, same docs)", y=1.0, fontsize=13)
fig.tight_layout(); fig.savefig("sweep_out/fve_compare_rl_vs_control.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("WROTE 3 capped figures (token x-axis -> 150)")
