"""Line-level marginal FVE: lines containing a quote mark vs not, controlled by line index."""
import json
import numpy as np

D = "/home/debian/nla-doll/experiments/qwen2.5-matryoshka-warmstart-sonnet46/v3_faithfulness_results"

def lines_from_tokens(tokens, deltas):
    """Group tokens into lines (split at tokens containing \\n). Returns
    list of (has_quote, sum_delta, n_tokens). Newline tokens' deltas join the
    PRECEDING line (they terminate it)."""
    lines, cur_txt, cur_d, cur_n = [], "", 0.0, 0
    for t, d in zip(tokens, deltas):
        cur_txt += t; cur_d += d; cur_n += 1
        if "\n" in t:
            if cur_txt.strip():
                lines.append(('"' in cur_txt, cur_d, cur_n))
            cur_txt, cur_d, cur_n = "", 0.0, 0
    if cur_txt.strip():
        lines.append(('"' in cur_txt, cur_d, cur_n))
    return lines

def analyze(name, path, kmax=12):
    rows = json.load(open(path))
    per_sample = [lines_from_tokens(r["tokens"], r["delta_fve"]) for r in rows]
    def stats(sample_ids):
        q = {k: [] for k in range(kmax)}; n = {k: [] for k in range(kmax)}
        for si in sample_ids:
            for k, (hq, d, _) in enumerate(per_sample[si][:kmax]):
                (q if hq else n)[k].append(d)
        # index-stratified excess weighted by quote-line count
        num = den = 0.0
        for k in range(kmax):
            if len(q[k]) >= 5 and len(n[k]) >= 5:
                num += len(q[k]) * (np.mean(q[k]) - np.mean(n[k])); den += len(q[k])
        return (num / den if den else np.nan), q, n
    e, q, n = stats(range(len(per_sample)))
    rng = np.random.default_rng(0)
    boots = [stats(rng.integers(0, len(per_sample), len(per_sample)))[0] for _ in range(300)]
    boots = [b for b in boots if b == b]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"\n=== {name} ===")
    print(f"index-controlled excess ΔFVE (quote-line − no-quote line): {e:+.4f}  [95% CI {lo:+.4f},{hi:+.4f}]")
    print(f"{'line k':>7s} {'n_q':>5s} {'n_noq':>6s} {'mean ΔFVE quote':>16s} {'no-quote':>9s} {'quote-frac':>10s}")
    curves = {"q": [], "n": [], "frac": []}
    for k in range(kmax):
        nq, nn = len(q[k]), len(n[k])
        mq = np.mean(q[k]) if nq else np.nan
        mn = np.mean(n[k]) if nn else np.nan
        curves["q"].append(mq); curves["n"].append(mn)
        curves["frac"].append(nq / max(1, nq + nn))
        print(f"{k+1:>7d} {nq:>5d} {nn:>6d} {mq:>16.4f} {mn:>9.4f} {nq/max(1,nq+nn):>10.1%}")
    return e, lo, hi, curves

r1 = analyze("v3 RL", f"{D}/quote_marginal_v3rl.json")
r2 = analyze("v3 warm-start", f"{D}/quote_marginal_v3ws.json")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
for ax, (name, (e, lo, hi, c)) in zip(axes, (("v3 RL", r1), ("v3 warm-start", r2))):
    ks = np.arange(1, len(c["q"]) + 1)
    ax.plot(ks, c["q"], "o-", color="#1f77b4", lw=2, label='lines WITH a quote mark')
    ax.plot(ks, c["n"], "s--", color="#d62728", lw=2, label="lines without")
    ax2 = ax.twinx()
    ax2.bar(ks, c["frac"], color="#1f77b4", alpha=0.12, width=0.8)
    ax2.set_ylim(0, 1); ax2.set_ylabel("fraction of lines with quotes", color="#7799bb", fontsize=9)
    ax.axhline(0, color="k", lw=.7, alpha=.5)
    ax.set_xlabel("line index"); ax.set_ylabel("mean marginal ΔFVE of line")
    ax.set_title(f"{name}\nindex-controlled excess: {e:+.4f} [{lo:+.4f},{hi:+.4f}]")
    ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.suptitle("Line-level marginal FVE: quote-bearing lines vs quote-free lines (by line index)", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(f"{D}/fig_line_quote.png", dpi=140, bbox_inches="tight")
print("\nwrote fig_line_quote.png")
