"""Render the Gate A cross-layer FVE graph. Draws 95% CI band/whiskers
automatically when results.json contains fve_ci95 fields."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = json.load(open("results.json"))
rows = sorted((v for k, v in res.items()
               if not k.startswith("_") and not k.endswith("_rep")),
              key=lambda e: e["layer"])
layers = [e["layer"] for e in rows]
fve = [e["fve_nrm"] for e in rows]
rep = res["L20_rep"]["fve_nrm"]
has_ci = "fve_ci95" in rows[0]
diff = res.get("_L21_minus_L20")

YMIN = -1.25
clip = lambda v: max(v, YMIN + 0.04)

fig, ax = plt.subplots(figsize=(10, 7.0))

if has_ci:
    yerr = [[clip(e["fve_nrm"]) - clip(e["fve_ci95"][0]) for e in rows],
            [clip(e["fve_ci95"][1]) - clip(e["fve_nrm"]) for e in rows]]
    eb_kw = dict(fmt="o-", color="#1a6faf", lw=1.8, ms=4.5, mfc="white", mew=1.4,
                 zorder=3, ecolor="#10456e", elinewidth=1.3, capsize=3.0, capthick=1.3)
    ax.errorbar(layers, [clip(v) for v in fve], yerr=yerr,
                label="FVE at layer L:  1 − ‖AR(AV($v_L$)) − $v_L$‖² / ‖$v_L$ − $\\bar{v}_L$‖²  (L2-normalized vectors)\n"
                      "whiskers: 95% CI, cluster bootstrap over the 60 documents — "
                      "±0.01–0.02, barely wider than the markers (see zoom)",
                **eb_kw)
    # zoom inset around the peak, where the whiskers are resolvable
    axins = ax.inset_axes([0.055, 0.56, 0.30, 0.36])
    axins.errorbar(layers, fve, yerr=[[e["fve_nrm"] - e["fve_ci95"][0] for e in rows],
                                      [e["fve_ci95"][1] - e["fve_nrm"] for e in rows]],
                   **{**eb_kw, "ms": 5.5, "capsize": 4.5})
    axins.plot([20], [rep], "D", color="#e07b39", ms=6, zorder=4)
    axins.set_xlim(18.4, 22.6)
    axins.set_ylim(0.40, 0.78)
    axins.set_xticks([19, 20, 21, 22])
    axins.tick_params(labelsize=7.5)
    axins.grid(alpha=0.3, lw=0.4)
    axins.set_title("zoom: layers 19–22, 95% CI whiskers", fontsize=8, color="#444444")
    ax.indicate_inset_zoom(axins, edgecolor="#888888", lw=0.9)
else:
    ax.plot(layers, [clip(v) for v in fve], "o-", color="#1a6faf", lw=2.2, ms=6, zorder=3,
            label="FVE at layer L:  1 − ‖AR(AV($v_L$)) − $v_L$‖² / ‖$v_L$ − $\\bar{v}_L$‖²  (L2-normalized vectors)")
ax.plot([20], [rep], "D", color="#e07b39", ms=7, zorder=4,
        label="Repeat run at layer 20 (sampling-noise check)")

ax.axhline(0, color="#777777", lw=1.1, zorder=1)
ax.text(10.6, 0.025, "FVE 0 — no better than always predicting the layer's mean activation",
        fontsize=8.5, color="#666666", va="bottom")
ax.axhline(-1, color="#aaaaaa", lw=0.9, ls="--", zorder=1)
ax.text(0.0, -0.975, "FVE −1 — explanations paired with the wrong position land here (measured: −0.94 to −1.3)",
        fontsize=8.5, color="#888888", va="bottom")

ax.axvline(20, color="#c0392b", lw=1.2, ls=":", zorder=1)
ax.annotate("NLA was trained\nonly on layer 20", xy=(20, -0.35), xytext=(15.0, -0.52),
            fontsize=9.5, color="#c0392b", ha="center",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1))

l21_note = "layer 21 beats the trained\nlayer: FVE 0.73 vs 0.60"
if diff:
    l21_note += (f"\n(paired Δ = {diff['diff']:+.2f},\n"
                 f"95% CI [{diff['ci_lo']:+.2f}, {diff['ci_hi']:+.2f}])")
ax.annotate(l21_note, xy=(21, fve[layers.index(21)]), xytext=(25.6, 0.82),
            fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))
ax.axvspan(14, 24, color="#1a6faf", alpha=0.06, zorder=0)
ax.annotate("usable band ≈ layers 14–24", xy=(16.4, 0.62), fontsize=10, color="#1a6faf",
            ha="center", style="italic")
ax.annotate("final layer: collapse\n(FVE −1.7, below scale;\n10% injection failures)",
            xy=(28, YMIN + 0.05), xytext=(24.6, -0.62), fontsize=9, ha="center",
            arrowprops=dict(arrowstyle="->", color="#444", lw=0.9))

ax.set_xlabel("Layer the activation vector was extracted from (Qwen2.5-7B-Instruct, 28 blocks; 0 = embeddings)",
              fontsize=10.5)
ax.set_ylabel("Fraction of variance explained, FVE (mean over 180 text positions)", fontsize=10.5)

fig.suptitle("A layer-20 NLA reads activations from other layers too",
             fontsize=14.5, weight="bold", y=0.975)
fig.text(0.5, 0.935,
         "Each activation $v_L$ is injected into the layer-20-trained verbalizer (AV); its text explanation is decoded back to a vector\n"
         "by the layer-20-trained reconstructor (AR). FVE measures how much of the activation's variance around the layer's mean\n"
         "activation $\\bar{v}_L$ that round trip recovers — 1 is perfect, 0 is no better than always guessing the mean.",
         ha="center", va="top", fontsize=9.3, color="#444444", linespacing=1.45)

ax.set_xticks(range(0, 29, 2))
ax.set_ylim(YMIN, 1.0)
ax.set_xlim(-0.6, 28.6)
ax.grid(alpha=0.25, lw=0.5)
ax.legend(loc="lower left", fontsize=8.8, framealpha=0.95, bbox_to_anchor=(0.02, 0.06))
footer = ("Setup: 180 positions (60 FineWeb docs × 3, position ≥ 50), released kitft/nla-qwen2.5-7b-L20 AV/AR checkpoints, temp 0.7. "
          "All vectors L2-normalized (direction-only,\nas in NLA training); FVE = 1 − 2(1−cos)/(1−‖mean unit vector‖²).")
if has_ci:
    footer += " CIs: percentile bootstrap, 10,000 resamples clustered by document."
footer += " Gate A of the NLA layer-diff study, 2026-06-10."
fig.text(0.01, 0.012, footer, fontsize=7.3, color="#777777", linespacing=1.4)
fig.tight_layout(rect=(0, 0.04, 1, 0.845))
fig.savefig("gate_a_transfer.png", dpi=170)
print("saved", "with CI band" if has_ci else "without CI")
