"""Score + plot the rhyme-steering result (local, CPU).

Reads results/steered_cont.json (13 conditions x 10 T=1 continuations), takes
the couplet's second line (first continuation line containing letters), rhyme-
classifies its final word (head-family /ɛd/ vs moon-family /uːn/), and renders
fig_rhyme_steer.png. Writes results/rhyme_scores.json.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
HEAD = {"bed", "spread", "dead", "said", "red", "overhead", "head", "instead",
        "fed", "shed", "led", "bread", "stead", "tread", "wed", "ahead", "widespread"}
MOON = {"moon", "soon", "june", "tune", "noon", "dune", "spoon", "croon",
        "strewn", "swoon", "boon", "loon", "attune", "balloon", "lagoon", "afternoon"}


def line2(cont):
    for ln in cont.strip().splitlines():
        if re.search(r"[A-Za-z]", ln):
            return ln.strip()
    return ""


def fw(cont):
    words = re.findall(r"[A-Za-z']+", line2(cont))
    return words[-1].lower() if words else "?"


d = json.load(open(HERE / "results" / "steered_cont.json"))
rows = []   # (display label, group, nH, nM, n)
ORDER = [
    ("baseline", "no patch", "ctrl"),
    ("self_vhead", "self-patch v_head", "ctrl"),
    ("donor_moon", "REAL moon activation", "ctrl"),
    ("mat:orig", "round-trip (unedited)", "mat"),
    ("mat:edit_rhyme_moon", "rhyme-expectation bit only", "mat"),
    ("mat:edit_lead_moon", "leading salience item only", "mat"),
    ("mat:edit_quotes_moon", "quoted line-1 text only", "mat"),
    ("mat:edit_full_moon", "FULL head→moon edit", "mat"),
    ("std:orig", "round-trip (unedited)", "std"),
    ("std:edit_rhyme_moon", "rhyme-expectation bit only", "std"),
    ("std:edit_finaltok_moon", "'final token' clause only", "std"),
    ("std:edit_quotes_moon", "quoted line-1 text only", "std"),
    ("std:edit_full_moon", "FULL head→moon edit", "std"),
]
scores = {}
for key, label, grp in ORDER:
    if key not in d["continuations"]:
        continue
    conts = d["continuations"][key]
    ws = [fw(c) for c in conts]
    nH = sum(w in HEAD for w in ws)
    nM = sum(w in MOON for w in ws)
    scores[key] = {"label": label, "H": nH, "M": nM, "n": len(ws),
                   "final_words": ws, "lines": [line2(c) for c in conts]}
    rows.append((label, grp, nH, nM, len(ws)))
json.dump(scores, open(HERE / "results" / "rhyme_scores.json", "w"), indent=1)

GH, GM = "#5b6773", "#2a78d6"
fig, ax = plt.subplots(figsize=(9.4, 6.4))
ys = range(len(rows))[::-1]
for y, (label, grp, nH, nM, n) in zip(ys, rows):
    ax.barh(y, nH, color=GH, height=0.62)
    ax.barh(y, nM, left=nH, color=GM, height=0.62)
    if nH:
        ax.text(nH / 2, y, str(nH), va="center", ha="center", color="w", fontsize=9, fontweight="bold")
    if nM:
        ax.text(nH + nM / 2, y, str(nM), va="center", ha="center", color="w", fontsize=9, fontweight="bold")
sec = {"ctrl": "controls", "mat": "matryoshka edits", "std": "standard edits"}
labels = []
last_grp = None
for label, grp, *_ in rows:
    pre = f"[{sec[grp]}]  " if grp != last_grp else ""
    labels.append(pre + label)
    last_grp = grp
ax.set_yticks(list(ys))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlim(0, 10)
ax.set_xlabel("continuations out of 10 (T=1, seed-matched across conditions)", fontsize=10)
ax.set_title("Editing the rhyme out of an NLA explanation steers the couplet — but the steerable bit\n"
             "lives in different places: matryoshka = the LEADING item; standard = the quoted line-1 text",
             fontsize=11.5)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=GH, label='second line rhymes with "head" (/ɛd/)'),
                   plt.Rectangle((0, 0), 1, 1, color=GM, label='second line rhymes with "moon" (/uːn/)')],
          fontsize=9, loc="lower right")
ax.grid(axis="x", color="#ccc", alpha=0.35)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.02, -0.01,
         'Context: "The sun goes down to rest its head" (couplet line 1). Each condition patches the L42 residual at the final token\n'
         "with critic(explanation-variant), norm-matched, then samples the second line. Edits modify one explanation per arm (mat\n"
         "rollout 6 / std rollout 0); full edit swaps every head/bed reference to moon/June. n=10 per condition.",
         fontsize=7.6, color="#777", ha="left", va="top")
fig.tight_layout()
fig.savefig(HERE / "results" / "fig_rhyme_steer.png", dpi=150, bbox_inches="tight")
print("[saved] results/fig_rhyme_steer.png")
for label, grp, nH, nM, n in rows:
    print(f"  {grp:4s} {label:34s} H={nH:2d} M={nM:2d} /{n}")
