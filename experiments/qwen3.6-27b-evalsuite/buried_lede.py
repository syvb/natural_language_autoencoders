"""The buried lede: matryoshka leads with the answer; the standard NLA opens
with throat-clearing and hides the informative sentence in the middle.

Uses only precomputed data (space/loo.json, space_std/loo.json — the solo/pfx
arrays). No GPU. Produces buried_lede.png + prints the head-to-head stats.

Core measurements per described moment (5,274 positions, both models):
  - solo FVE of the FIRST unit read (line 1 / sentence 1)
  - index of the single most-informative unit (0 = it's first)
  - how many units, in the model's own order, before the running reconstruction
    beats knowing-nothing (FVE > 0)
"""
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SENT = re.compile(r"(?<=[.!?])\s+")


def collect():
    out = {}
    ml = json.load(open(HERE / "space" / "loo.json"))["entries"]
    mpre = json.load(open(HERE / "space" / "precache.json"))["entries"]
    sl = json.load(open(HERE / "space_std" / "loo.json"))["entries"]
    spre = json.load(open(HERE / "space_std" / "precache.json"))["entries"]

    def gather(loo_entries, pre_entries, cum_from_pre):
        first_solo, best_idx, first_pos, first_units = [], [], [], []
        for le, pe in zip(loo_entries, pre_entries):
            cum = (r["fve"] for r in [])  # placeholder
            for i, (f, s) in enumerate(zip(le["full"], le["solo"])):
                if f is None or s is None or not s:
                    continue
                first_solo.append(s[0])
                best_idx.append(int(np.argmax(s)))
                if cum_from_pre:  # matryoshka: cumulative fve lives in precache
                    fve = pe["results"][i]["fve"]
                else:             # standard: cumulative is loo's pfx
                    fve = le["pfx"][i]
                first_pos.append(next((k for k, v in enumerate(fve) if v > 0), len(fve)))
                # first unit text (for the preamble gallery)
                if cum_from_pre:
                    first_units.append(pe["results"][i]["lines"][0])
                else:
                    first_units.append(SENT.split(le["units"][i][0])[0]
                                       if le["units"][i] else "")
        return (np.array(first_solo), np.array(best_idx),
                np.array(first_pos), first_units)

    out["mat"] = gather(ml, mpre, True)
    out["std"] = gather(sl, spre, False)
    return out


def main():
    d = collect()
    for name, key in [("matryoshka (line units)", "mat"), ("standard (sentence units)", "std")]:
        fs, bi, fp, _ = d[key]
        print(f"\n[{name}] n={len(fs)}")
        print(f"  first unit alone: mean solo-FVE {fs.mean():+.3f}, frac positive {np.mean(fs > 0):.1%}")
        print(f"  most-informative unit index: mean {bi.mean():.2f}, frac it's FIRST {np.mean(bi == 0):.1%}")
        print(f"  units read (own order) to reach FVE>0: mean {fp.mean():.2f}, "
              f"frac needing ≥2 {np.mean(fp >= 1):.1%}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, RED = "#2a78d6", "#e34948"
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # panel 1: solo-FVE of the first unit
    bins = np.linspace(-1.1, 0.9, 61)
    ax[0].hist(d["mat"][0], bins=bins, density=True, color=BLUE, alpha=.8,
               label=f"matryoshka line 1\n(mean {d['mat'][0].mean():+.2f}, {np.mean(d['mat'][0] > 0):.0%} positive)")
    ax[0].hist(d["std"][0], bins=bins, density=True, color=RED, alpha=.65,
               label=f"standard sentence 1\n(mean {d['std'][0].mean():+.2f}, {np.mean(d['std'][0] > 0):.0%} positive)")
    ax[0].axvline(0, color="k", lw=.8, ls=":")
    ax[0].set(xlabel="recovery from the FIRST unit alone (FVE)", ylabel="density",
              title="Reading only the opening line/sentence\n(left of dotted = worse than reading nothing)")
    ax[0].legend(fontsize=8.5, loc="upper left")

    # panel 2: where is the informative unit?
    mx = 8
    for key, c, lab, off in [("mat", BLUE, "matryoshka", -0.19), ("std", RED, "standard", 0.19)]:
        bi = np.clip(d[key][1], 0, mx)
        vals, _ = np.histogram(bi, bins=np.arange(mx + 2))
        ax[1].bar(np.arange(mx + 1) + off, vals / vals.sum(), 0.38, color=c, label=lab)
    ax[1].set(xlabel="index of the single most-informative unit (0 = it's first)",
              ylabel="fraction of moments",
              title="Where is the answer buried?\nmatryoshka puts it first; the standard NLA scatters it")
    ax[1].legend(fontsize=9)

    # panel 3: units-to-positive, cumulative
    for key, c, lab in [("mat", BLUE, "matryoshka"), ("std", RED, "standard")]:
        fp = d[key][2]
        xs = np.arange(0, 7)
        ys = [np.mean(fp <= k) for k in xs]
        ax[2].plot(xs, ys, "o-", color=c, lw=2.4, ms=6, label=lab)
    ax[2].set(xlabel="units read, in the model's own order",
              ylabel="fraction of moments with useful (FVE>0) reconstruction",
              title="How much must you read before it's useful?\nmatryoshka: 1 line. standard: ~3 sentences",
              ylim=(0, 1.02))
    ax[2].legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    png = HERE / "buried_lede.png"
    fig.savefig(png, dpi=130)
    print(f"\n[saved] {png}")

    # a gallery of standard first sentences (the preamble) + matryoshka first lines
    print("\n--- standard opening sentences (throat-clearing) vs matryoshka opening lines ---")
    for i in range(0, 5275, 900):
        if i < len(d["std"][3]):
            print(f"  STD:  {d['std'][3][i][:88]}")
            print(f"  MAT:  {d['mat'][3][i][:88]}\n")


if __name__ == "__main__":
    main()
