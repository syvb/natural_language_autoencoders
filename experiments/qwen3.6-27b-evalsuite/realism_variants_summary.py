"""Combined comparison figure: eval-y-score correlations across the 3 NLA/unit configs."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CFG = [("matryoshka RL\n(salience bullets)", "realism_variants_correlate_mat_rl.json", "#2a78d6"),
       ("standard\n(natural lines)", "realism_variants_correlate_std.json", "#e34948"),
       ("standard\n(10 word-chunks)", "realism_variants_correlate_std_chunks10.json", "#e39a49")]

def rr(rows, metric):
    es=np.array([r["eval_score"] for r in rows]); y=np.array([r[metric] for r in rows])
    m=np.isfinite(es)&np.isfinite(y); r,p=pearsonr(es[m],y[m]); return r,p

metrics=[("frac_flag","detection frequency\n(flags eval-awareness at all)"),
         ("mean_idx_present","WHERE, given flagged\n(salience/position rank)")]
fig,axs=plt.subplots(1,2,figsize=(12,5.2))
for ax,(mk,ttl) in zip(axs,metrics):
    labels,vals,cols,ps=[],[],[],[]
    for lbl,f,c in CFG:
        rows=json.load(open(HERE/f))["rows"]; r,p=rr(rows,mk)
        labels.append(lbl); vals.append(r); cols.append(c); ps.append(p)
    x=np.arange(len(labels))
    bars=ax.bar(x,vals,color=cols,width=0.6)
    for xi,(v,p) in enumerate(zip(vals,ps)):
        star="***" if p<1e-3 else "**" if p<1e-2 else "*" if p<0.05 else "n.s."
        ax.text(xi, v+(0.02 if v>=0 else -0.02), f"r={v:+.2f}\n{star}", ha="center",
                va="bottom" if v>=0 else "top", fontsize=9)
    ax.axhline(0,color="k",lw=0.8); ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=8.5)
    ax.set_ylabel("Pearson r  vs eval-y score"); ax.set_title(ttl,fontsize=11)
    ax.set_ylim(-0.6,0.6)
fig.suptitle("Does the NLA's eval-awareness track how eval-y the prompt is? (100 system-prompt variations)",
             fontsize=12,y=1.0)
fig.tight_layout()
fig.savefig(HERE/"realism_variants_summary.png",dpi=140,bbox_inches="tight")
print("[saved] realism_variants_summary.png")
