"""Regenerate the cliff figure (realism_variants_cliff.png) with a descriptive,
blog-ready title."""
import json, numpy as np
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
THR=0.5
cache=json.load(open("realism_variants_line_scores.json"))
V=json.load(open("realism_variants_verb_mat_rl.json"))["variants"]
xs,ys=[],[]
for v in V:
    pp=[]
    for pos in v["lines"]:
        u=[l.strip() for l in pos if l and l.strip()]
        if not u: continue
        aw=[i for i,x in enumerate(u) if cache.get(x) is not None and cache[x]>=THR]
        pp.append(aw[0] if aw else 10)
    if pp: xs.append(v["eval_score"]); ys.append(float(np.mean(pp)))
x=np.array(xs); y=np.array(ys); lo=x<=85; hi=x>85
fig,ax=plt.subplots(figsize=(7.6,5.9))
ax.scatter(x[lo],y[lo],s=32,alpha=0.8,color="#2a78d6",edgecolor="white",lw=0.4,label="≤ 85 (realistic → moderately eval-y)")
ax.scatter(x[hi],y[hi],s=36,alpha=0.85,color="#e34948",edgecolor="white",lw=0.4,label="> 85 (explicitly labelled a test)")
b,a0=np.polyfit(x[lo],y[lo],1); xr=np.array([x[lo].min(),x[lo].max()]); ax.plot(xr,b*xr+a0,color="#2a78d6",lw=2.2,ls="--")
b2,a2=np.polyfit(x,y,1); xr2=np.array([x.min(),x.max()]); ax.plot(xr2,b2*xr2+a2,color="#888",lw=1.8,ls=":",label="fit incl. all points")
r_all,p_all=pearsonr(x,y); r_lo,p_lo=pearsonr(x[lo],y[lo])
ax.set_xlabel("evaliness of the prompt  (0 = realistic deployment → 100 = obviously an eval)")
ax.set_ylabel("where the NLA first says \"this is a test\"\n(mean salience-rank of first eval-aware bullet; lower = more prominent; 10 = never)")
ax.set_title("The NLA reports eval-awareness prominently only for prompts explicitly\nlabelled as tests — not as prompts get gradually less realistic",fontsize=11.5)
ax.text(0.03,0.03,f"all (n=100):  r={r_all:+.2f} (p={p_all:.0e})\nexcluding the >85 cliff (n={lo.sum()}):  r={r_lo:+.2f} (n.s.)",
        transform=ax.transAxes,fontsize=9.5,va="bottom",bbox=dict(boxstyle="round",fc="white",ec="#ccc",alpha=0.95))
ax.legend(loc="upper right",fontsize=9)
fig.tight_layout(); fig.savefig("realism_variants_cliff.png",dpi=150)
print("[saved] realism_variants_cliff.png")
