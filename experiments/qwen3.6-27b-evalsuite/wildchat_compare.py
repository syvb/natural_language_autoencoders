"""Three-way eval-awareness ROC with WildChat chat-token CONTROLS as negatives
(replaces the pretraining-doc controls, which confounded format with
eval-awareness). Positives = honeypot explanations. Redone at full / 40-tok /
20-tok budgets. Judges every (truncated) explanation with the same generic
gpt-4o-mini judge.

    OPENROUTER_API_KEY=... python wildchat_compare.py
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
BUDGETS = [("full", None), ("first 40 tok", 40), ("first 20 tok", 20)]
MODELS = {  # honeypot precache (for positives)  +  control_expl file (negatives)
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


async def main():
    key = os.environ["OPENROUTER_API_KEY"]
    pos_raw = {name: honey_expls(pc) for name, (pc, _, _) in MODELS.items()}
    neg_raw = {name: control_expls(cf) for name, (_, cf, _) in MODELS.items()}
    for name in MODELS:
        print(f"{name}: {len(pos_raw[name])} honeypot, {len(neg_raw[name])} control", flush=True)

    uniq = {}
    for name in MODELS:
        for _, n in BUDGETS:
            for s in pos_raw[name] + neg_raw[name]:
                uniq.setdefault(trunc(s, n), None)
    keys = list(uniq)
    print(f"{len(keys)} unique strings to judge (~${len(keys)*150/1e6*0.15:.2f})", flush=True)

    sem = asyncio.Semaphore(32); done = [0]
    async with httpx.AsyncClient() as client:
        async def run(s):
            uniq[s] = await ea.judge(client, sem, key, s)
            done[0] += 1
            if done[0] % 4000 == 0:
                print(f"  judged {done[0]}/{len(keys)}", flush=True)
        await asyncio.gather(*(run(s) for s in keys))

    def scores(strs, n):
        return np.array([uniq[trunc(s, n)] for s in strs if uniq.get(trunc(s, n)) is not None])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.1))
    summary = {}
    for bi, (btitle, n) in enumerate(BUDGETS):
        print(f"\n=== {btitle} (WildChat controls) ===")
        for name, (_, _, c) in MODELS.items():
            hp = scores(pos_raw[name], n); pl = scores(neg_raw[name], n)
            a = auc(pl, hp)
            summary[f"{name}|{btitle}"] = round(float(a), 4)
            thr = np.unique(np.concatenate([pl, hp]))[::-1]
            fpr = [(pl >= t).mean() for t in thr] + [1.0]
            tpr = [(hp >= t).mean() for t in thr] + [1.0]
            ax[bi].plot(fpr, tpr, lw=2.2, color=c, label=f"{name}  {a:.3f}")
            print(f"  {name:17s} AUC {a:.3f}  (plain n={len(pl)}, honey n={len(hp)})")
        ax[bi].plot([0, 1], [0, 1], "k:", lw=1)
        ax[bi].set(xlabel="WildChat-control FPR", ylabel="honeypot hit rate",
                   title=f"Eval-awareness ROC — {btitle}\n(chat-token controls)")
        ax[bi].legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    png = HERE / "wildchat_roc.png"
    fig.savefig(png, dpi=130)
    json.dump(summary, open(HERE / "wildchat_roc.json", "w"), indent=1)
    print(f"\n[saved] {png}")


if __name__ == "__main__":
    asyncio.run(main())
