"""Do hallucinated line items have LOW marginal FVE (relative to the average
line-item marginal)?  For the matryoshka (lines) and the standard NLA (sentences).

Extends the faithfulness sweep to EVERY matryoshka line (the original study
judged only top-3); the standard model's sentences were already all judged.
Then, per model, compares the marginal FVE of items by verdict — pooled AND
position-controlled, because marginal FVE is strongly position-dependent and
hallucination rate varies with position (the confound that faked the earlier
'std detector' result).

    python marginal_by_halluc.py            # judge remaining mat lines + stats + fig
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze  # reuse FAITH_PROMPT, ask/run_jobs, the judge cache, SE, HERE

HERE = analyze.HERE
SE = analyze.SE
HALL = {"CONTRADICTED", "FABRICATED"}
BLUE, RED, GREY = "#2a78d6", "#e34948", "#888888"


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
    mat = json.load(open(HERE / "results" / "subset_scores_mat.json"))["entries"]
    std = json.load(open(HERE / "results" / "subset_scores_std.json"))["entries"]

    # ── extend faithfulness to ALL matryoshka lines ──────────────────────────
    todo, meta = [], []
    for e in mat:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            for k, u in enumerate(rec["units"]):
                key = f"mat|{e['ci']}|{ri}|{k}"
                if key not in faith:
                    todo.append(analyze.FAITH_PROMPT.format(
                        passage=man[e["ci"]]["prefix_text"],
                        continuation=man[e["ci"]]["answer_text"], note=u))
                    meta.append(key)
    if todo:
        print(f"[judge] {len(todo)} un-judged mat lines…", flush=True)
        verds = analyze.run_jobs(todo, analyze.FAITH_RE, "mat-all")
        for key, v in zip(meta, verds):
            faith[key] = v
        json.dump(faith, open(HERE / "results" / "judged_faithfulness.json", "w"))
    else:
        print("[judge] all mat lines already judged", flush=True)

    # ── gather (position, marginal, verdict) per item, per model ─────────────
    def rows(entries, arm):
        out = []
        for e in entries:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                m = marg(rec)
                for k in range(len(m)):
                    v = faith.get(f"{arm}|{e['ci']}|{ri}|{k}")
                    if v is not None:
                        out.append((k, m[k], v))
        return out

    data = {"matryoshka (lines)": (rows(mat, "mat"), BLUE),
            "standard (sentences)": (rows(std, "std"), RED)}

    # ── stats ────────────────────────────────────────────────────────────────
    from scipy.stats import mannwhitneyu
    summary = {}
    print("\n" + "=" * 74)
    for name, (rws, _) in data.items():
        ks = np.array([r[0] for r in rws])
        mg = np.array([r[1] for r in rws])
        vd = np.array([r[2] for r in rws])
        is_h = np.isin(vd, list(HALL))
        is_s = vd == "SUPPORTED"
        gavg = mg.mean()
        # position-detrended marginal: subtract the mean marginal at each position
        detr = mg.copy().astype(float)
        for k in np.unique(ks):
            sel = ks == k
            detr[sel] = mg[sel] - mg[sel].mean()
        u_raw = mannwhitneyu(mg[is_h], mg[~is_h], alternative="less")
        u_det = mannwhitneyu(detr[is_h], detr[~is_h], alternative="less")
        print(f"\n{name}  (n={len(rws)}, global mean marginal = {gavg:+.4f})")
        print(f"  per-verdict mean marginal:")
        for cat in ["FABRICATED", "CONTRADICTED", "META", "SUPPORTED"]:
            sel = vd == cat
            if sel.any():
                print(f"    {cat:12s} n={sel.sum():4d}  mean {mg[sel].mean():+.4f}  "
                      f"(detrended {detr[sel].mean():+.4f})")
        print(f"  hallucinated  n={is_h.sum():4d}  mean marginal {mg[is_h].mean():+.4f}  "
              f"(Δ vs global {mg[is_h].mean()-gavg:+.4f})")
        print(f"  faithful(SUP) n={is_s.sum():4d}  mean marginal {mg[is_s].mean():+.4f}")
        print(f"  Mann-Whitney halluc<rest:  pooled p={u_raw.pvalue:.2e}  "
              f"position-detrended p={u_det.pvalue:.2e}")
        summary[name] = dict(
            n=len(rws), global_mean=float(gavg),
            halluc_mean=float(mg[is_h].mean()), halluc_n=int(is_h.sum()),
            faithful_mean=float(mg[is_s].mean()), faithful_n=int(is_s.sum()),
            halluc_detr=float(detr[is_h].mean()), faithful_detr=float(detr[is_s].mean()),
            p_pooled=float(u_raw.pvalue), p_detrended=float(u_det.pvalue),
            per_verdict={c: [float(mg[vd == c].mean()), int((vd == c).sum()),
                             float(detr[vd == c].mean())]
                         for c in ["FABRICATED", "CONTRADICTED", "META", "SUPPORTED"]
                         if (vd == c).any()})
    json.dump(summary, open(HERE / "results" / "marginal_by_halluc.json", "w"), indent=1)
    print("\n[saved] results/marginal_by_halluc.json")

    # ── figure: 2 models × (pooled by verdict | position-controlled) ─────────
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))
    cats = ["FABRICATED", "CONTRADICTED", "META", "SUPPORTED"]
    catcol = ["#b4520a", "#d98a3a", "#8896a3", "#2f9c69"]
    for row, (name, (rws, col)) in enumerate(data.items()):
        ks = np.array([r[0] for r in rws]); mg = np.array([r[1] for r in rws])
        vd = np.array([r[2] for r in rws]); gavg = mg.mean()
        is_h = np.isin(vd, list(HALL))
        # left: mean marginal by verdict, with global-average reference line
        axL = axes[row][0]
        vals = [mg[vd == c].mean() if (vd == c).any() else np.nan for c in cats]
        ns = [(vd == c).sum() for c in cats]
        bars = axL.bar(range(4), vals, color=catcol, width=0.72)
        lo = min(0, np.nanmin(vals)); hi = np.nanmax(vals)
        axL.set_ylim(lo - 0.02 * (hi - lo), hi + 0.16 * (hi - lo))
        axL.axhline(gavg, color=GREY, ls="--", lw=1.4,
                    label=f"average line-item marginal ({gavg:+.3f})")
        vmax = np.nanmax(vals)
        for i, (v, n) in enumerate(zip(vals, ns)):
            if v == vmax:   # tallest bar: label inside (white) to clear the legend
                axL.text(i, v - 0.006 * (hi - lo) / 0.1, f"{v:+.3f}", ha="center",
                         fontsize=8.5, color="white", fontweight="bold", va="top")
            else:
                axL.text(i, v + (0.006 if v >= 0 else -0.006), f"{v:+.3f}", ha="center",
                         fontsize=8.5, va="bottom" if v >= 0 else "top")
            axL.text(i, 0.001, f"n={n}", ha="center", fontsize=7, color="#333",
                     rotation=90, va="bottom")
        axL.set_xticks(range(4)); axL.set_xticklabels(cats, fontsize=8.5, rotation=12)
        axL.set_ylabel("mean marginal FVE", fontsize=10)
        axL.set_title(f"{name} — marginal by faithfulness", fontsize=11, color=col)
        axL.axhline(0, color="#444", lw=0.8); axL.legend(fontsize=8.5, loc="upper right")
        axL.grid(axis="y", color="#ccc", alpha=0.3)
        axL.spines[["top", "right"]].set_visible(False)
        # right: mean marginal vs position, halluc vs faithful
        axR = axes[row][1]
        kmax = int(np.percentile(ks, 99))
        xs = list(range(kmax + 1))
        hy = [mg[(ks == k) & is_h].mean() if ((ks == k) & is_h).any() else np.nan for k in xs]
        fy = [mg[(ks == k) & ~is_h].mean() if ((ks == k) & ~is_h).any() else np.nan for k in xs]
        axR.plot(xs, fy, "-o", color="#2f9c69", ms=4, lw=1.8, label="faithful / META")
        axR.plot(xs, hy, "-o", color="#b4520a", ms=4, lw=1.8, label="hallucinated")
        axR.axhline(0, color="#444", lw=0.8)
        axR.set_xlabel("item position (line / sentence index)", fontsize=10)
        axR.set_ylabel("mean marginal FVE", fontsize=10)
        axR.set_title(f"{name} — marginal by position", fontsize=11, color=col)
        axR.legend(fontsize=8.5, loc="upper right")
        axR.grid(color="#ccc", alpha=0.3)
        axR.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Do hallucinated items have low marginal FVE? Pooled (left) vs position-controlled (right)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(HERE / "results" / "fig_marginal_by_halluc.png", dpi=150, bbox_inches="tight")
    print("[saved] results/fig_marginal_by_halluc.png")


if __name__ == "__main__":
    main()
