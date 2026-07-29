"""Marginal FVE by token decile, split by whether the decile chunk was judged a
hallucination. The critic scored marginal FVE at the LINE level (subset_scores);
here each line's real marginal is redistributed onto the 10 token deciles in
proportion to token overlap (real Qwen tokenizer, boundaries aligned to the same
full-text token sequence chunk_judge.py used). Approximate (assumes within-line
uniformity); the exact version would re-score the critic on decile prefixes (GPU).
"""
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
SE = HERE.parent / "qwen3.6-27b-suffix-eval"
H = {"CONTRADICTED", "FABRICATED"}
NCHUNK = 10
GREEN, RED = "#2f9c69", "#8a1c13"
cv = {tuple(k.split("|")[:1] + [int(x) for x in k.split("|")[1:]]): v
      for k, v in json.load(open(HERE / "results" / "chunk_verdicts.json")).items()}


def get_tok(repo, sub):
    root = snapshot_download(repo, allow_patterns=[f"{sub}/*"],
                             token=open(os.path.expanduser("~/.hf_token")).read().strip())
    return AutoTokenizer.from_pretrained(f"{root}/{sub}")


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def decile_marginals(lines, mvals, tok):
    """Redistribute each line's marginal onto 10 token deciles by token overlap."""
    cum = [len(tok.encode("\n".join(lines[:k]), add_special_tokens=False)) for k in range(len(lines) + 1)]
    T = cum[-1]
    if T < NCHUNK:
        return None
    db = [round(d * T / NCHUNK) for d in range(NCHUNK + 1)]
    dm = np.zeros(NCHUNK)
    for k in range(len(lines)):
        a, b = cum[k], cum[k + 1]
        if b <= a:
            continue
        for d in range(NCHUNK):
            ov = max(0, min(b, db[d + 1]) - max(a, db[d]))
            if ov:
                dm[d] += mvals[k] * ov / (b - a)
    return dm


ARMS = [("mat", "ceselder/nla-qwen36-27b-matryoshka", "warmstart_av_lora", "subset_scores_mat.json", "matryoshka (lines)"),
        ("std", "ceselder/qwen3.6-27b-nla-L42", "av_sft_lora", "subset_scores_std.json", "standard (sentences)")]

for arm, repo, sub, fn, title in ARMS:
    tok = get_tok(repo, sub)
    # per context: list of (decile, marginal, is_halluc)
    by_ci = {}
    for e in json.load(open(HERE / "results" / fn))["entries"]:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            dm = decile_marginals(rec["units"], marg(rec), tok)
            if dm is None:
                continue
            for d in range(NCHUNK):
                v = cv.get((arm, e["ci"], ri, d))
                if v is None:
                    continue
                by_ci.setdefault(e["ci"], []).append((d, dm[d], v in H))
    cis = list(by_ci)

    def means(sample_cis):
        sh = [[] for _ in range(NCHUNK)]; sn = [[] for _ in range(NCHUNK)]
        for c in sample_cis:
            for d, m, h in by_ci[c]:
                (sh if h else sn)[d].append(m)
        mh = np.array([np.mean(sh[d]) if sh[d] else np.nan for d in range(NCHUNK)])
        mn = np.array([np.mean(sn[d]) if sn[d] else np.nan for d in range(NCHUNK)])
        return mh, mn

    rng = random.Random(0)
    ph, pn = means(cis)
    boots = [means(rng.choices(cis, k=len(cis))) for _ in range(1500)]
    bh = np.array([b[0] for b in boots]); bn = np.array([b[1] for b in boots])
    hlo, hhi = np.nanpercentile(bh, 2.5, axis=0), np.nanpercentile(bh, 97.5, axis=0)
    nlo, nhi = np.nanpercentile(bn, 2.5, axis=0), np.nanpercentile(bn, 97.5, axis=0)
    x = np.arange(1, NCHUNK + 1)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.errorbar(x - 0.06, pn, yerr=[pn - nlo, nhi - pn], fmt="o-", color=GREEN, ms=6, lw=2, capsize=3,
                label="chunk NOT hallucinated")
    ax.errorbar(x + 0.06, ph, yerr=[ph - hlo, hhi - ph], fmt="o-", color=RED, ms=6, lw=2, capsize=3,
                label="chunk hallucinated (CON/FAB)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("explanation token decile (1 = first 10% → 10 = last 10%)", fontsize=11)
    ax.set_ylabel("marginal FVE of the decile (token-weighted)", fontsize=11)
    ax.set_title(f"{title} — decile marginal FVE: hallucinated vs not\n"
                 "(line-level critic marginals redistributed onto token deciles)", fontsize=11.5)
    ax.legend(fontsize=9.5)
    ax.grid(color="#ccc", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.02, -0.02,
             "Approximate: each line's real critic marginal FVE is split across token deciles by overlap (real Qwen tokenizer). "
             "Hallucination = decile chunk judged CON/FAB (§3i).\nError bars: cluster bootstrap over the 250 contexts (95%). "
             "Curves track closely; any gap is small and confined to the high-marginal early deciles (consistent with §3e–§3h).",
             fontsize=7.6, color="#777", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(HERE / "results" / f"fig_decile_marg_{arm}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] fig_decile_marg_{arm}.png")
    print(f"  {arm} decile marg (not): " + " ".join(f"{v:.3f}" for v in pn))
    print(f"  {arm} decile marg (hal): " + " ".join(f"{v:.3f}" for v in ph))
