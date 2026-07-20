"""Analyze + plot the couplet-generalization run (poetry_gen.py).

Metrics per couplet x model:
  mention  : fraction of line-break explanations naming the plan word (n=18)
  strict   : edit_all completions ending in the partner's plan word
  family   : ... or any word sharing the final 2 chars with the partner's
             plan or anchor (rhyme-family heuristic, e.g. fog/bog for log)
  plan_kill: plan-word rate in edit_all (should be 0) vs nopatch
Writes pg_summary.json and fig_poetry_generalize.png.

Usage: python pg_analyze.py <dir with pg_meta.json etc.>
"""
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def last_words(gens):
    out = []
    for g in gens:
        line = g.split("\n")[0].strip()
        w = re.sub(r"[^a-z]", "", line.split()[-1].lower()) if line.split() else ""
        out.append(w)
    return out


D = sys.argv[1] if len(sys.argv) > 1 else "results/poetry"
meta = json.load(open(f"{D}/pg_meta.json"))
steer = json.load(open(f"{D}/pg_steer.json"))
av = {m: json.load(open(f"{D}/pg_av_{m}.json"))["tags"] for m in ("mat", "std")}
sel = {r["name"]: r for r in meta["selected"]}

rows = []
for name, R in steer.items():
    r = sel[name]
    partner = sel[r["partner"]]
    fam = {partner["plan"][-2:], partner["anchor"][-2:]}
    row = {"name": name, "plan": r["plan"], "plan_n": r["plan_n"],
           "edit_to": f"{partner['plan']}/{partner['anchor']}", "m": {}}
    for m in ("mat", "std"):
        ws = last_words(R["arms"][f"mrec_{m}_edit_all"])
        strict = sum(w in r["targets"] for w in ws) / len(ws)
        family = sum(len(w) >= 3 and (w in r["targets"] or w[-2:] in fam)
                     for w in ws) / len(ws)
        kill = sum(w == r["plan"] for w in ws) / len(ws)
        nl = sum(w in r["targets"] for w in last_words(R["arms"][f"mrec_{m}_edit_nl"]))
        orig_plan = sum(w == r["plan"] for w in
                        last_words(R["arms"][f"mrec_{m}_orig_all"])) / 25
        a = av[m][name]
        row["m"][m] = dict(mention=a["nl_mention"] / a["nl_n"],
                           strict=strict, family=family, plan_in_edit=kill,
                           nl_strict=nl / 25, orig_plan=orig_plan)
    nop = sum(w in r["targets"] for w in last_words(R["arms"]["nopatch"])) / 25
    row["nopatch_target"] = nop
    rows.append(row)
rows.sort(key=lambda x: -x["plan_n"])

print(f"{'couplet':<8} {'plan':<6} {'base':>5} {'edit->':<12} "
      f"{'mat mention':>11} {'std mention':>11} {'mat strict':>10} {'std strict':>10} "
      f"{'mat family':>10} {'std family':>10}")
for row in rows:
    print(f"{row['name']:<8} {row['plan']:<6} {row['plan_n']:>3}/25 {row['edit_to']:<12} "
          f"{row['m']['mat']['mention']:>11.0%} {row['m']['std']['mention']:>11.0%} "
          f"{row['m']['mat']['strict']:>10.0%} {row['m']['std']['strict']:>10.0%} "
          f"{row['m']['mat']['family']:>10.0%} {row['m']['std']['family']:>10.0%}")
for m in ("mat", "std"):
    tm = sum(r["m"][m]["mention"] for r in rows) / len(rows)
    ts = sum(r["m"][m]["strict"] for r in rows) / len(rows)
    tf = sum(r["m"][m]["family"] for r in rows) / len(rows)
    tk = sum(r["m"][m]["plan_in_edit"] for r in rows) / len(rows)
    tn = sum(r["m"][m]["nl_strict"] for r in rows) / len(rows)
    print(f"[{m}] mean mention {tm:.0%} | edit_all strict {ts:.0%} family {tf:.0%} "
          f"| plan surviving edit {tk:.0%} | single-token strict {tn:.0%}")
json.dump(rows, open(f"{D}/pg_summary.json", "w"), indent=1)

# ── figure ───────────────────────────────────────────────────────────────────
C = {"mat": "#2a78d6", "std": "#eb6834"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
NAME = {"mat": "matryoshka", "std": "standard"}
labels = [f"{r['name']}→{sel[r['name']]['partner']}\n“{r['plan']}” {r['plan_n']}/25"
          for r in rows]
X = range(len(rows))

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), facecolor=SURF)
panels = [("mention", "explanation names the plan word at the line break (n=18)"),
          ("strict", "edit steering: completions ending in the new target word (n=25)")]
for ax, (key, title) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for j, m in enumerate(("mat", "std")):
        xs = [x + (j - 0.5) * 0.36 for x in X]
        ys = [r["m"][m][key] for r in rows]
        ax.bar(xs, ys, width=0.33, color=C[m], zorder=3,
               label=NAME[m] if key == "mention" else None)
        if key == "strict":  # family-loose extension
            for x, r, y in zip(xs, rows, ys):
                f = r["m"][m]["family"]
                if f > y:
                    ax.bar(x, f - y, width=0.33, bottom=y, color=C[m],
                           alpha=0.32, zorder=2)
    if key == "strict":
        for x, r in zip(X, rows):
            ax.plot([x - 0.75 * 0.36, x + 0.75 * 0.36], [0, 0], color=INK2)
        ax.text(len(rows) - 0.5, 0.9,
                "single-token arm: 0% on every couplet\nlight bars = same rhyme family",
                ha="right", fontsize=8.5, color=INK2)
    ax.set_xticks(list(X))
    ax.set_xticklabels(labels, fontsize=7.5, color=INK)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title(title, fontsize=10.5, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.tick_params(colors=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")
fig.suptitle("Generalization over 6 spontaneous couplets: whole-line explanation-edit steering "
             "moves every rhyme; plan verbalization is couplet-dependent",
             x=0.02, ha="left", fontsize=12, color=INK)
fig.text(0.02, 0.012,
         "couplets sorted by baseline plan concentration · edit swaps (plan, anchor) to the next "
         "couplet's pair · original plan word survives the edit patch in 0/300 completions per model",
         color=INK2, fontsize=8)
fig.tight_layout(rect=(0, 0.035, 1, 0.9))
fig.savefig(f"{D}/fig_poetry_generalize.png", dpi=170, facecolor=SURF)
print(f"saved {D}/fig_poetry_generalize.png")
