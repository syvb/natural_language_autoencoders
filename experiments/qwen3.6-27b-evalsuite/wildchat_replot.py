"""Re-plot the WildChat-control eval-awareness ROCs in the layouts we want.

Caches per-string judge scores to wildchat_scores.json so this only judges once
(temperature-0 judge → deterministic). Emits, for each budget, a standalone
ROC (clean title, no subheader), plus a 2-panel full+20-token image.

    OPENROUTER_API_KEY=... python wildchat_replot.py
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_awareness as ea  # noqa: E402

HERE = Path(__file__).resolve().parent
HONEY = "<|im_start|>system"
ENC = tiktoken.get_encoding("o200k_base")
CACHE = HERE / "wildchat_scores.json"
BUDGETS = [("full", None), ("first 40 tokens", 40), ("first 20 tokens", 20)]
MODELS = {
    "matryoshka RL":   ("space/precache.json",     "control_expl_mat_rl.json",   "#2a78d6"),
    "matryoshka warm": ("warmstart_precache.json", "control_expl_mat_warm.json", "#c98500"),
    "standard RL":     ("space_std/precache.json",  "control_expl_std.json",     "#e34948"),
}


def trunc(s, n):
    return s if n is None else ENC.decode(ENC.encode(s)[:n])


def auc(neg, pos):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([neg, pos]))
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


def honey_expls(pc):
    e = [x for x in json.load(open(HERE / pc))["entries"] if x["text"].startswith(HONEY)][0]
    return ["\n".join(r["lines"]) for r in e["results"] if r and r.get("lines")]


def control_expls(f):
    return ["\n".join(ls) for ls in json.load(open(HERE / f))["lines"] if ls]


async def get_scores():
    pos_raw = {n: honey_expls(pc) for n, (pc, _, _) in MODELS.items()}
    neg_raw = {n: control_expls(cf) for n, (_, cf, _) in MODELS.items()}
    need = set()
    for name in MODELS:
        for _, nt in BUDGETS:
            for s in pos_raw[name] + neg_raw[name]:
                need.add(trunc(s, nt))
    cache = json.load(open(CACHE)) if CACHE.exists() else {}
    todo = [s for s in need if s not in cache]
    print(f"{len(need)} unique strings, {len(todo)} to judge "
          f"(~${len(todo)*150/1e6*0.15:.2f})", flush=True)
    if todo:
        key = os.environ["OPENROUTER_API_KEY"]
        sem = asyncio.Semaphore(32); done = [0]
        async with httpx.AsyncClient() as client:
            async def run(s):
                cache[s] = await ea.judge(client, sem, key, s)
                done[0] += 1
                if done[0] % 5000 == 0:
                    print(f"  judged {done[0]}/{len(todo)}", flush=True)
            await asyncio.gather(*(run(s) for s in todo))
        json.dump(cache, open(CACHE, "w"))
        print(f"[cached] {CACHE}", flush=True)
    return pos_raw, neg_raw, cache


def plot_ax(ax, pos_raw, neg_raw, cache, nt, title):
    for name, (_, _, c) in MODELS.items():
        hp = np.array([cache[trunc(s, nt)] for s in pos_raw[name]
                       if cache.get(trunc(s, nt)) is not None])
        pl = np.array([cache[trunc(s, nt)] for s in neg_raw[name]
                       if cache.get(trunc(s, nt)) is not None])
        a = auc(pl, hp)
        thr = np.unique(np.concatenate([pl, hp]))[::-1]
        fpr = [(pl >= t).mean() for t in thr] + [1.0]
        tpr = [(hp >= t).mean() for t in thr] + [1.0]
        ax.plot(fpr, tpr, lw=2.3, color=c, label=f"{name}  {a:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="benign-chat FPR", ylabel="honeypot hit rate", title=title)
    ax.legend(loc="lower right", fontsize=9)


async def main():
    pos_raw, neg_raw, cache = await get_scores()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 3 standalone graphs
    for title, nt in BUDGETS:
        fig, ax = plt.subplots(figsize=(6.2, 5.4))
        plot_ax(ax, pos_raw, neg_raw, cache, nt, f"Eval-awareness detection — {title}")
        fig.tight_layout()
        slug = title.replace(" ", "").replace("tokens", "tok")
        fig.savefig(HERE / f"wildchat_roc_{slug}.png", dpi=140)
        plt.close(fig)

    # combined full + 20-token
    fig, axs = plt.subplots(1, 2, figsize=(12.4, 5.4))
    plot_ax(axs[0], pos_raw, neg_raw, cache, None, "Eval-awareness detection — full")
    plot_ax(axs[1], pos_raw, neg_raw, cache, 20, "Eval-awareness detection — first 20 tokens")
    fig.tight_layout()
    fig.savefig(HERE / "wildchat_roc_full_and_20tok.png", dpi=140)
    print("[saved] wildchat_roc_full.png / _first40tok.png / _first20tok.png / _full_and_20tok.png")


if __name__ == "__main__":
    asyncio.run(main())
