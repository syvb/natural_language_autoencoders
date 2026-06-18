import csv, glob, re
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path("experiment_results")
runs=[]
for d in sorted(glob.glob(str(R/"pen*_seed*"))):
    tag=Path(d).name
    m=re.search(r"pen([0-9.]+)",tag); pen=float(m.group(1)) if m else 0
    rows=list(csv.DictReader(open(Path(d)/"per_step.csv")))
    runs.append((pen,tag,rows))
runs.sort()
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(16,4.6))
for pen,tag,rows in runs:
    st=[int(r["step"]) for r in rows]
    ln=[float(r["mean_response_len_tokens"]) for r in rows]
    fv=[float(r["fve_nrm"]) for r in rows if r["fve_nrm"]]
    a1.plot(st,ln,"-o",ms=3,label=f"λ={pen:g}")
    a2.plot(st[:len(fv)],fv,"-o",ms=3,label=f"λ={pen:g}")
a1.set_xlabel("RL step");a1.set_ylabel("mean explanation length (tokens)");a1.set_title("Length over training");a1.grid(alpha=.3);a1.legend()
a2.set_xlabel("RL step");a2.set_ylabel("fve_nrm (reconstruction)");a2.set_title("Reconstruction over training");a2.grid(alpha=.3);a2.legend()
# tradeoff scatter (final values)
xs=[float(rows[-1]["mean_response_len_tokens"]) for _,_,rows in runs]
ys=[float([r for r in rows if r["fve_nrm"]][-1]["fve_nrm"]) for _,_,rows in runs]
pens=[p for p,_,_ in runs]
a3.plot(xs,ys,"-o",color="#c0392b")
for x,y,p in zip(xs,ys,pens): a3.annotate(f"λ={p:g}",(x,y),textcoords="offset points",xytext=(6,5))
a3.set_xlabel("final mean length (tokens)");a3.set_ylabel("final fve_nrm");a3.set_title("Length vs reconstruction tradeoff");a3.grid(alpha=.3)
fig.suptitle("NLA length penalty — Qwen2.5-7B continue-RL (KL off, seed 1234, 35 steps)")
fig.tight_layout()
fig.savefig(R/"tradeoff_partial.png",dpi=140)
print("wrote",R/"tradeoff_partial.png")
