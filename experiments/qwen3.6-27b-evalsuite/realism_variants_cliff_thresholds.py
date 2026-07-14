"""Cliff scatter at several eval-awareness thresholds (P>=0.5/0.7/0.9)."""
import json, numpy as np
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
cache=json.load(open("realism_variants_line_scores.json"))
V=json.load(open("realism_variants_verb_mat_rl.json"))["variants"]
THRS=[0.5,0.7,0.9]
fig,axs=plt.subplots(1,3,figsize=(16.5,5.4),sharey=True)
for ax,thr in zip(axs,THRS):
    xs,ys=[],[]
    for v in V:
        pp=[]
        for pos in v["lines"]:
            u=[l.strip() for l in pos if l and l.strip()]
            if not u: continue
            aw=[i for i,x in enumerate(u) if cache.get(x) is not None and cache[x]>=thr]
            pp.append(aw[0] if aw else 10)
        if pp: xs.append(v["eval_score"]); ys.append(float(np.mean(pp)))
    x=np.array(xs); y=np.array(ys); lo=x<=85
    ax.scatter(x[lo],y[lo],s=26,alpha=0.8,color="#2a78d6",edgecolor="white",lw=0.3,label="evaliness ≤ 85")
    ax.scatter(x[~lo],y[~lo],s=28,alpha=0.85,color="#e34948",edgecolor="white",lw=0.3,label="evaliness > 85")
    b,a0=np.polyfit(x[lo],y[lo],1); xr=np.array([x[lo].min(),x[lo].max()]); ax.plot(xr,b*xr+a0,color="#2a78d6",lw=2,ls="--")
    b2,a2=np.polyfit(x,y,1); xr2=np.array([x.min(),x.max()]); ax.plot(xr2,b2*xr2+a2,color="#888",lw=1.6,ls=":")
    r_all,p_all=pearsonr(x,y); r_lo,p_lo=pearsonr(x[lo],y[lo])
    ax.set_title(f"eval-aware cutoff: P ≥ {thr}",fontsize=12)
    ax.set_xlabel("evaliness of the prompt")
    ax.text(0.03,0.03,f"all (n=100):  r={r_all:+.2f} (p={p_all:.0e})\n≤85 (n={lo.sum()}):  r={r_lo:+.2f} (p={p_lo:.2f})",
            transform=ax.transAxes,fontsize=8.8,va="bottom",bbox=dict(boxstyle="round",fc="white",ec="#ccc",alpha=0.95))
    if thr==THRS[0]:
        ax.set_ylabel("where the NLA first says \"this is a test\"\n(mean salience-rank of first eval-aware bullet; 10 = never)")
        ax.legend(loc="upper right",fontsize=8.5)
fig.suptitle("Where the NLA first flags eval-awareness vs. prompt evaliness — by eval-awareness cutoff",fontsize=13,y=1.0)
fig.tight_layout()
fig.savefig("realism_variants_cliff_thresholds.png",dpi=145,bbox_inches="tight")
print("[saved] realism_variants_cliff_thresholds.png")
