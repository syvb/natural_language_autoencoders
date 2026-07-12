"""Three-way eval-awareness ROC at an N-token reading budget (standalone panel).

Judges the first-N-token (o200k) truncation of each model's explanation with the
same generic judge, then draws the honeypot-vs-plain ROC for all three models.

    OPENROUTER_API_KEY=... python budget_roc.py --ntok 20
"""
import argparse
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
MODELS = {
    "matryoshka RL":   ("space/precache.json",     "#2a78d6"),
    "matryoshka warm": ("warmstart_precache.json", "#c98500"),
    "standard RL":     ("space_std/precache.json", "#e34948"),
}


def auc(neg, pos):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([neg, pos]))
    return (r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntok", type=int, default=20)
    args = ap.parse_args()
    n = args.ntok
    key = os.environ["OPENROUTER_API_KEY"]

    def trunc(s):
        return ENC.decode(ENC.encode(s)[:n])

    items, uniq = {}, {}
    for name, (pc, _) in MODELS.items():
        lst = []
        for e in json.load(open(HERE / pc))["entries"]:
            hp = e["text"].startswith(HONEY)
            for r in e["results"]:
                if r and r.get("lines"):
                    t = trunc("\n".join(r["lines"]))
                    lst.append((hp, t)); uniq.setdefault(t, None)
        items[name] = lst
    keys = list(uniq)
    print(f"first-{n}-tok: {sum(len(v) for v in items.values())} judgments, "
          f"{len(keys)} unique (~${len(keys)*150/1e6*0.15:.2f})", flush=True)

    sem = asyncio.Semaphore(32); done = [0]
    async with httpx.AsyncClient() as client:
        async def run(s):
            uniq[s] = await ea.judge(client, sem, key, s)
            done[0] += 1
            if done[0] % 3000 == 0:
                print(f"  judged {done[0]}/{len(keys)}", flush=True)
        await asyncio.gather(*(run(s) for s in keys))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    out = {}
    print(f"\n=== eval-awareness AUC at first {n} tokens ===")
    for name, (_, c) in MODELS.items():
        pl = np.array([uniq[t] for h, t in items[name] if not h and uniq[t] is not None])
        hp = np.array([uniq[t] for h, t in items[name] if h and uniq[t] is not None])
        a = auc(pl, hp)
        out[name] = a
        thr = np.unique(np.concatenate([pl, hp]))[::-1]
        fpr = [(pl >= t).mean() for t in thr] + [1.0]
        tpr = [(hp >= t).mean() for t in thr] + [1.0]
        ax.plot(fpr, tpr, lw=2.4, color=c, label=f"{name}  AUC={a:.3f}")
        print(f"  {name:17s} AUC {a:.3f}  (hit@FPR5% {(hp >= np.quantile(pl, .95)).mean():.1%})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set(xlabel="plain-control FPR", ylabel="honeypot hit rate",
           title=f"Eval-awareness detection — first {n} tokens")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    png = HERE / f"three_way_roc_first{n}.png"
    fig.savefig(png, dpi=140)
    json.dump(out, open(HERE / f"roc_first{n}.json", "w"), indent=1)
    print(f"[saved] {png}")


if __name__ == "__main__":
    asyncio.run(main())
