"""EXACT marginal FVE by token decile, split by whether the decile chunk was
judged a hallucination. Uses decile_scores_{arm}.json — the critic scored
directly on cumulative token-decile prefixes on a GPU (score_deciles.py):
  pfx_decile[d] = FVE(first d+1 deciles);  marginal[d] = pfx[d] - pfx[d-1].
Aligned to the chunk_judge.py deciles, so decile d's marginal pairs with decile
d's hallucination verdict. Cluster-bootstrap 95% CI over the 250 contexts.
"""
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
H = {"CONTRADICTED", "FABRICATED"}
NCHUNK = 10
GREEN, RED = "#2f9c69", "#8a1c13"
cv = {tuple(k.split("|")[:1] + [int(x) for x in k.split("|")[1:]]): v
      for k, v in json.load(open(HERE / "results" / "chunk_verdicts.json")).items()}


def marg(pfx):
    return [pfx[0]] + [pfx[d] - pfx[d - 1] for d in range(1, NCHUNK)]


for arm, title in [("mat", "matryoshka (lines)"), ("std", "standard (sentences)")]:
    ds = json.load(open(HERE / "results" / f"decile_scores_{arm}.json"))["entries"]
    by_ci = {}
    for e in ds:
        ci = e["ci"]
        for ri, pfx in enumerate(e["pfx_decile"]):
            if pfx is None:
                continue
            m = marg(pfx)
            for d in range(NCHUNK):
                v = cv.get((arm, ci, ri, d))
                if v is None:
                    continue
                by_ci.setdefault(ci, []).append((d, m[d], v in H))
    cis = list(by_ci)

    def means(sample):
        sh = [[] for _ in range(NCHUNK)]; sn = [[] for _ in range(NCHUNK)]
        for c in sample:
            for d, m, h in by_ci[c]:
                (sh if h else sn)[d].append(m)
        return (np.array([np.mean(sh[d]) if sh[d] else np.nan for d in range(NCHUNK)]),
                np.array([np.mean(sn[d]) if sn[d] else np.nan for d in range(NCHUNK)]))

    rng = random.Random(0)
    ph, pn = means(cis)
    boots = [means(rng.choices(cis, k=len(cis))) for _ in range(1500)]
    bh = np.array([b[0] for b in boots]); bn = np.array([b[1] for b in boots])
    hlo, hhi = np.nanpercentile(bh, 2.5, 0), np.nanpercentile(bh, 97.5, 0)
    nlo, nhi = np.nanpercentile(bn, 2.5, 0), np.nanpercentile(bn, 97.5, 0)
    x = np.arange(1, NCHUNK + 1)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.errorbar(x, pn, yerr=[pn - nlo, nhi - pn], fmt="o-", color=GREEN, ms=6, lw=2, capsize=3,
                label="chunk NOT hallucinated")
    ax.errorbar(x, ph, yerr=[ph - hlo, hhi - ph], fmt="o-", color=RED, ms=6, lw=2, capsize=3,
                label="chunk hallucinated (CON/FAB)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("explanation token decile (1 = first 10% → 10 = last 10%)", fontsize=11)
    ax.set_ylabel("marginal FVE of the decile", fontsize=11)
    ax.set_title(f"{title} — decile marginal FVE: hallucinated vs not (EXACT)\n"
                 "critic scored directly on cumulative token-decile prefixes (GPU)", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.02,
             "Exact: marginal[d] = FVE(first d+1 token deciles) − FVE(first d deciles), own critic, real Qwen "
             "tokenizer (deciles aligned to §3i judging).\nHallucination = decile chunk judged CON/FAB. Error bars: "
             "cluster bootstrap over the 250 contexts (95%).",
             fontsize=7.6, color="#777", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_decile_marg_{arm}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] fig_decile_marg_{arm}.png")
    print(f"  {arm} marg not: " + " ".join(f"{v:.3f}" for v in pn))
    print(f"  {arm} marg hal: " + " ".join(f"{v:.3f}" for v in ph))
