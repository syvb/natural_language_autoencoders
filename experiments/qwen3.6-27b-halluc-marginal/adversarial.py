"""Adversarial re-analysis of the halluc-marginal claim. Runs its own computations."""
import json, re
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path("/home/debian/nla-doll/.claude/worktrees/halluc-marginal/experiments/qwen3.6-27b-halluc-marginal")
HALL = {"CONTRADICTED", "FABRICATED"}
CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯]")

faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
mat = json.load(open(HERE / "results" / "subset_scores_mat.json"))["entries"]
std = json.load(open(HERE / "results" / "subset_scores_std.json"))["entries"]

def marg(pfx):
    return [pfx[0]] + [pfx[k] - pfx[k - 1] for k in range(1, len(pfx))]

def rows(entries, arm):
    out = []
    for e in entries:
        for ri, rec in enumerate(e["rollouts"]):
            if not rec:
                continue
            pfx = rec["pfx"]; m = marg(pfx); units = rec["units"]
            for k in range(len(m)):
                v = faith.get(f"{arm}|{e['ci']}|{ri}|{k}")
                if v is None:
                    continue
                u = units[k] if k < len(units) else ""
                prev = 0.0 if k == 0 else pfx[k - 1]
                out.append(dict(ci=e["ci"], ri=ri, k=k, marg=m[k], v=v, unit=u,
                                prev=prev, nchar=len(u), nword=len(u.split()),
                                cjk=bool(CJK.search(u))))
    return out

DATA = {"mat": rows(mat, "mat"), "std": rows(std, "std")}

def arr(rws, key):
    return np.array([r[key] for r in rws])

def isH(rws):
    return np.isin(arr(rws, "v"), list(HALL))

def bh(pvals):
    p = np.array(pvals); n = len(p); order = np.argsort(p)
    adj = np.empty(n); prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]; val = p[idx] * n / (i + 1); prev = min(prev, val); adj[idx] = prev
    return adj

def ols(y, X):
    # X includes intercept column; returns beta, se, t via normal eqns
    XtX = X.T @ X; XtXi = np.linalg.pinv(XtX)
    beta = XtXi @ X.T @ y
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = (resid @ resid) / dof
    cov = sigma2 * XtXi
    se = np.sqrt(np.diag(cov))
    t = beta / se
    from scipy.stats import t as tdist
    p = 2 * tdist.sf(np.abs(t), dof)
    return beta, se, t, p

print("="*80)
print("LOADED: mat n=%d  std n=%d" % (len(DATA["mat"]), len(DATA["std"])))
for arm in ("mat","std"):
    rws=DATA[arm]; h=isH(rws)
    print(f"  {arm}: halluc n={h.sum()}  faithful+meta n={(~h).sum()}  "
          f"pooled halluc mean {arr(rws,'marg')[h].mean():+.4f}  rest {arr(rws,'marg')[~h].mean():+.4f}")

print("\n"+"#"*80)
print("# ATTACK 1: alternative confound controls (cumulative pfx_prev, item length)")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]; h=isH(rws)
    y=arr(rws,"marg").astype(float); k=arr(rws,"k").astype(float)
    prev=arr(rws,"prev").astype(float); nchar=arr(rws,"nchar").astype(float)
    nword=arr(rws,"nword").astype(float); hf=h.astype(float)
    n=len(y)
    print(f"\n--- {arm} (n={n}) ---")
    # baseline OLS: marg ~ halluc only
    X=np.column_stack([np.ones(n), hf])
    b,se,t,p=ols(y,X); print(f"  marg ~ halluc            : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    # control position (as continuous + we also do categorical below)
    # categorical position dummies
    ks=sorted(set(arr(rws,"k")))
    posd=np.column_stack([ (arr(rws,"k")==kk).astype(float) for kk in ks[1:] ])  # drop first as ref
    X=np.column_stack([np.ones(n), hf, posd])
    b,se,t,p=ols(y,X); print(f"  + position (categorical) : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    # control cumulative prev instead of position
    X=np.column_stack([np.ones(n), hf, prev])
    b,se,t,p=ols(y,X); print(f"  + cumulative pfx_prev     : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    # control length only
    X=np.column_stack([np.ones(n), hf, nchar])
    b,se,t,p=ols(y,X); print(f"  + nchar                   : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    X=np.column_stack([np.ones(n), hf, nword])
    b,se,t,p=ols(y,X); print(f"  + nword                   : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    # everything: position cat + prev + length
    X=np.column_stack([np.ones(n), hf, posd, prev, nchar, nword])
    b,se,t,p=ols(y,X); print(f"  + pos+prev+nchar+nword    : halluc beta={b[1]:+.4f}  p={p[1]:.2e}")
    # also stratify by cumulative-prev quintile and compare within-bin means
    qs=np.quantile(prev,[0,.2,.4,.6,.8,1.0]); qs[-1]+=1e-9
    binidx=np.clip(np.digitize(prev,qs[1:-1]),0,4)
    print("  within cumulative-pfx quintile: halluc vs rest mean marginal")
    for bi in range(5):
        sel=binidx==bi
        hh=sel&h; rr=sel&~h
        if hh.sum() and rr.sum():
            print(f"     q{bi} prev[{qs[bi]:+.2f},{qs[bi+1]:+.2f}) n={sel.sum():4d} "
                  f"halluc {y[hh].mean():+.4f}(n{hh.sum()}) rest {y[rr].mean():+.4f}(n{rr.sum()}) "
                  f"diff {y[hh].mean()-y[rr].mean():+.4f}")

print("\n"+"#"*80)
print("# ATTACK 2: distribution tails within position bins (halluc vs faithful+meta)")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]; h=isH(rws); y=arr(rws,"marg").astype(float); ka=arr(rws,"k")
    print(f"\n--- {arm} ---  (per position: p10/p50/p90, frac<0, frac<-0.01)")
    for kk in sorted(set(ka)):
        if (ka==kk).sum()<40: continue
        hh=(ka==kk)&h; rr=(ka==kk)&~h
        if hh.sum()<10 or rr.sum()<10: continue
        qh=np.quantile(y[hh],[.1,.5,.9]); qr=np.quantile(y[rr],[.1,.5,.9])
        print(f"  k={kk} nH={hh.sum():4d} nR={rr.sum():4d} | "
              f"H p10/50/90 {qh[0]:+.3f}/{qh[1]:+.3f}/{qh[2]:+.3f} "
              f"neg {np.mean(y[hh]<0):.3f} <-.01 {np.mean(y[hh]<-0.01):.3f} | "
              f"R p10/50/90 {qr[0]:+.3f}/{qr[1]:+.3f}/{qr[2]:+.3f} "
              f"neg {np.mean(y[rr]<0):.3f} <-.01 {np.mean(y[rr]<-0.01):.3f}")

print("\n"+"#"*80)
print("# ATTACK 3: per-position Mann-Whitney (halluc<rest), BH + Bonferroni")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]; h=isH(rws); y=arr(rws,"marg").astype(float); ka=arr(rws,"k")
    pos=[k for k in sorted(set(ka)) if ((ka==k)&h).sum()>=10 and ((ka==k)&~h).sum()>=10]
    pv=[]; info=[]
    for kk in pos:
        hh=(ka==kk)&h; rr=(ka==kk)&~h
        # two-sided to detect either direction, plus record signed diff
        u=mannwhitneyu(y[hh],y[rr],alternative="two-sided")
        diff=y[hh].mean()-y[rr].mean()
        pv.append(u.pvalue); info.append((kk,diff,hh.sum(),rr.sum()))
    adj=bh(pv); bonf=np.minimum(np.array(pv)*len(pv),1.0)
    print(f"\n--- {arm} (m={len(pos)} positions tested) ---")
    for (kk,diff,nh,nr),p,a,bo in zip(info,pv,adj,bonf):
        star="  <== survives BH" if a<0.05 else ""
        dirn="H<R (halluc lower)" if diff<0 else "H>R (halluc HIGHER)"
        print(f"  k={kk}: diff{diff:+.4f} {dirn:22s} p={p:.2e} BH={a:.2e} Bonf={bo:.2e}{star}")

def detrend_test(rws, mask):
    sub=[r for r,m in zip(rws,mask) if m]
    h=isH(sub); y=arr(sub,"marg").astype(float); ka=arr(sub,"k")
    detr=y.copy()
    for kk in np.unique(ka):
        sel=ka==kk; detr[sel]=y[sel]-y[sel].mean()
    if h.sum()<5 or (~h).sum()<5: return None
    u=mannwhitneyu(detr[h],detr[~h],alternative="less")
    return (len(sub),h.sum(),(~h).sum(),y[h].mean()-y[~h].mean(),
            detr[h].mean()-detr[~h].mean(),u.pvalue)

print("\n"+"#"*80)
print("# ATTACK 4: subgroup robustness of 'vanishes under position control'")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]
    print(f"\n--- {arm} ---   (pos-detrended MW halluc<rest: pooled diff | detr diff | detr p)")
    subs={
      "ALL":                 [True]*len(rws),
      "exclude CJK":         [not r["cjk"] for r in rws],
      "exclude META(rest=SUP only)": [r["v"] in HALL or r["v"]=="SUPPORTED" for r in rws],
      "k>=1 only":           [r["k"]>=1 for r in rws],
    }
    for nm,mask in subs.items():
        res=detrend_test(rws,mask)
        if res is None: print(f"  {nm:32s} n/a"); continue
        n,nh,nr,pooled,dtr,p=res
        flag=" *** DETR SIG halluc<rest ***" if p<0.05 else ""
        print(f"  {nm:32s} n={n:5d} H={nh:4d} R={nr:4d} pooledΔ{pooled:+.4f} detrΔ{dtr:+.4f} p={p:.3f}{flag}")

print("\n"+"#"*80)
print("# ATTACK 5: are bottom-decile-by-position items enriched for hallucination?")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]; h=isH(rws); y=arr(rws,"marg").astype(float); ka=arr(rws,"k")
    base=h.mean()
    inbot=np.zeros(len(rws),bool)
    for kk in np.unique(ka):
        sel=ka==kk
        if sel.sum()<20: continue
        thr=np.quantile(y[sel],0.10)
        inbot[sel & (y<=thr)]=True
    ph_bot=h[inbot].mean(); ph_rest=h[~inbot].mean()
    print(f"\n--- {arm} --- base halluc rate {base:.3f}")
    print(f"  bottom-decile-by-position: n={inbot.sum()} halluc rate {ph_bot:.3f} "
          f"lift {ph_bot/base:.3f}x   (rest {ph_rest:.3f})")
    # also top decile for contrast
    intop=np.zeros(len(rws),bool)
    for kk in np.unique(ka):
        sel=ka==kk
        if sel.sum()<20: continue
        thr=np.quantile(y[sel],0.90)
        intop[sel & (y>=thr)]=True
    print(f"  top-decile-by-position   : n={intop.sum()} halluc rate {h[intop].mean():.3f} "
          f"lift {h[intop].mean()/base:.3f}x")

print("\n"+"#"*80)
print("# ATTACK 6: is std META -0.179 mean driven by outliers? median/IQR/robust")
print("#"*80)
for arm in ("mat","std"):
    rws=DATA[arm]; y=arr(rws,"marg").astype(float); vd=arr(rws,"v")
    for cat in ["META","SUPPORTED","FABRICATED","CONTRADICTED"]:
        sel=vd==cat
        if sel.sum()<5: continue
        yy=y[sel]; q=np.quantile(yy,[.25,.5,.75])
        print(f"  {arm} {cat:12s} n={sel.sum():4d} mean{yy.mean():+.4f} "
              f"median{q[1]:+.4f} IQR[{q[0]:+.4f},{q[2]:+.4f}] "
              f"min{yy.min():+.3f} max{yy.max():+.3f} frac<0 {np.mean(yy<0):.3f}")

print("\n"+"#"*80)
print("# SUPPLEMENT: dissect the mat full-OLS residual (categorical pos + length)")
print("#"*80)
rws=DATA["mat"]; h=isH(rws).astype(float)
y=arr(rws,"marg").astype(float); ka=arr(rws,"k")
nchar=arr(rws,"nchar").astype(float); nword=arr(rws,"nword").astype(float)
prev=arr(rws,"prev").astype(float); n=len(y)
ks=sorted(set(arr(rws,"k")))
posd=np.column_stack([(arr(rws,"k")==kk).astype(float) for kk in ks[1:]])
for lbl,extra in [("pos cat only",None),("pos+nchar",nchar[:,None]),
                  ("pos+nword",nword[:,None]),("pos+prev",prev[:,None]),
                  ("pos+prev only(no len)",prev[:,None])]:
    cols=[np.ones(n),h,posd]
    if extra is not None: cols.append(extra)
    X=np.column_stack(cols); b,se,t,p=ols(y,X)
    print(f"  {lbl:24s} halluc beta={b[1]:+.5f} p={p[1]:.2e}")
# is halluc correlated with length within position? and length with marginal?
print("  within-position corr(halluc,nchar) and corr(nchar,marg):")
for kk in ks[:6]:
    sel=arr(rws,"k")==kk
    ch=np.corrcoef(h[sel],nchar[sel])[0,1]; cm=np.corrcoef(nchar[sel],y[sel])[0,1]
    print(f"    k={kk} corr(H,nchar)={ch:+.3f} corr(nchar,marg)={cm:+.3f}")
