"""CoT-Hints result figure: the rule is in the activation, not in the words.

Panel A — manipulation check: P(base model's sampled answer = the marked
option) at the extraction position. hint_correct ≈ 90% (rule learned, driving
behavior); hint_incorrect ≈ 39% (weakly learned; chance 25%).
Panel B — verbalization: P(AV explanation states the marker's meaning), the
paper's metric, per condition and model. All ~0%.

    python plot_hints.py   # results/hints_results.json (+ gen files) -> hints_null.png
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, RED, GRAY, INK, GREEN = "#2a78d6", "#e34948", "#8a8f98", "#22262b", "#2e9e62"


def behavioral():
    d = json.load(open(HERE / "results" / "hints_mat.json"))
    out = {}
    for cond in ("hint_correct", "hint_incorrect"):
        fol = valid = 0
        for e in d["entries"]:
            if e["condition"] != cond or not e["behav_picks"]:
                continue
            picks = [p for p in e["behav_picks"] if p in "ABCD"]
            valid += len(picks)
            fol += sum(p == e["final_marked_letter"] for p in picks)
        out[cond] = fol / max(1, valid)
    return out


def main():
    r = json.load(open(HERE / "results" / "hints_results.json"))
    beh = behavioral()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1.35]})

    # ── A: behavioral manipulation check ──────────────────────────────────────
    conds = ["hint_correct", "hint_incorrect"]
    vals = [beh[c] * 100 for c in conds]
    cols = [GREEN, "#b8762a"]
    for k, (v, c) in enumerate(zip(vals, cols)):
        axA.bar(k, v, 0.55, color=c, alpha=0.9, zorder=3)
        axA.text(k, v + 2, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold", color=c)
    axA.axhline(25, color=INK, ls=":", lw=1.4)
    axA.text(1.32, 26.5, "chance 25%", ha="right", fontsize=9.5, color=INK)
    axA.set_xticks(range(2))
    axA.set_xticklabels(["■ marks correct\n(hint_correct)", "■ marks a wrong one\n(hint_incorrect)"], fontsize=10.5)
    axA.set_ylim(0, 100)
    axA.set_ylabel("base model picks the ■-marked option (%)", fontsize=11)
    axA.set_title("A. The rule drives behavior\nat the extraction position", fontsize=12, fontweight="bold")
    axA.spines[["top", "right"]].set_visible(False)
    axA.grid(axis="y", color=GRAY, alpha=0.22)

    # ── B: verbalization rates (the paper's metric) ───────────────────────────
    conds3 = ["hint_correct", "hint_incorrect", "no_marker"]
    width = 0.38
    for mi, (m, c, lab) in enumerate([("mat", BLUE, "Matryoshka"), ("std", RED, "Standard")]):
        rates = [r["models"][m]["label_rates"].get(cond, {}).get("CORRECT", 0) * 100
                 for cond in conds3]
        xs = [k + (mi - 0.5) * width for k in range(3)]
        axB.bar(xs, [max(v, 0.6) for v in rates], width, color=c, alpha=0.9, label=lab, zorder=3)
        for x, v in zip(xs, rates):
            axB.text(x, max(v, 0.6) + 1.2, f"{v:.0f}%", ha="center", fontsize=11,
                     fontweight="bold", color=c)
    axB.set_xticks(range(3))
    axB.set_xticklabels(["hint_correct", "hint_incorrect", "no_marker\n(control)"], fontsize=10.5)
    axB.set_ylim(0, 100)
    axB.set_ylabel('AV explanation states "■ marks the correct answer" (%)', fontsize=10.5)
    axB.set_title("B. …but neither NLA ever verbalizes it\nD = P(says ■=correct | ■=correct) − P(same | ■=incorrect) = 0.0 for both",
                  fontsize=12, fontweight="bold")
    axB.legend(fontsize=10.5, loc="upper right")
    axB.spines[["top", "right"]].set_visible(False)
    axB.grid(axis="y", color=GRAY, alpha=0.22)

    fig.suptitle("CoT-Hints replication — Qwen3.6-27B NLAs, layer 42 @ the answer position of an unanswerable marked question",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.text(0.01, -0.03,
             "30 transcripts/condition × 30 in-context marked MCQs; behavioral = 8 samples/transcript (valid-letter picks); "
             "verbalization judged by Haiku-4.5 over 120 explanations/condition/model. The AVs instead describe the final\n"
             "question's 'nonsense/trap/benchmark' character — a real feature — while missing the marker rule entirely "
             "(0 explanations state marker semantics; 0 name the expected answer letter).",
             fontsize=8.2, color=GRAY, ha="left")
    fig.tight_layout()
    out = HERE / "hints_null.png"
    fig.savefig(out, dpi=145, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
