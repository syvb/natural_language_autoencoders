"""Do the WORST hallucinations have low marginal FVE?

Reviewers + the blind inter-rater check show the faithfulness judge over-flags:
many CONTRADICTED/FABRICATED notes are really the NLA describing the model's
not-yet-written internal prediction (legitimate) rather than confabulating. So
within the SUBSTANTIVE (non-misquote) hallucinations, classify each as:
  UNVERBALIZED_PREDICTION — plausibly describes the model's internal direction /
      what it is about to generate; the mismatch is only that it isn't in the
      text yet (arguably not a hallucination at all), or
  GENUINE_FABRICATION — invents specific facts/entities with no grounding in the
      passage and no basis in what the model is plausibly computing (the "worst").
Then compare marginal FVE (within position, vs SUPPORTED). Hypothesis: the
GENUINE (worst) subset reconstructs worse; the PREDICTION subset reconstructs
like faithful content (which is why it was diluting the §3b signal).

    JUDGE_WORKERS=200 python judge_severity.py
"""
import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze

HERE, SE = analyze.HERE, analyze.SE
KEY, URL, JUDGE = analyze.KEY, analyze.URL, analyze.JUDGE
H = {"CONTRADICTED", "FABRICATED"}
GREEN, ORANGE, DARK = "#2f9c69", "#d98a3a", "#8a1c13"
PROMPT_VERSION = "severity-v1"
_slug = re.sub(r"[^a-z0-9]+", "-", JUDGE.lower())
CACHE = HERE / "results" / f".severity_cache_{_slug}.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
ANS = re.compile(r"<answer>\s*(UNVERBALIZED_PREDICTION|GENUINE_FABRICATION)\s*</answer>", re.I)

PROMPT = """An interpretability tool read a language model's internal state at the last token of a passage and wrote a note describing what the model represents / is about to generate. The note did NOT match the passage or its true continuation — but that alone does not make it a hallucination: a good note can legitimately describe what the model is INTERNALLY predicting or "thinking about" next, which need not be written in the passage yet.

Passage (what the model had read):
<passage>
{passage}
</passage>

Text that actually came next (the model had not emitted it yet):
<continuation>
{continuation}
</continuation>

The note:
<note>
{note}
</note>

Classify the note:
- UNVERBALIZED_PREDICTION: it plausibly describes the model's internal direction — a topic, entity, or continuation the model could reasonably be computing next given the passage — even though that content is not literally in the passage/continuation. The mismatch is one of timing/verbalization, not invention.
- GENUINE_FABRICATION: it asserts specific facts, entities, or events with no grounding in the passage and no basis in what the model could plausibly be computing — a real confabulation (e.g. inventing a named source, a wrong entity, an event that contradicts the passage).

Answer with exactly one in <answer> tags, e.g. <answer>GENUINE_FABRICATION</answer>."""


def ask(prompt):
    k = hashlib.sha1((PROMPT_VERSION + "|" + JUDGE + "|" + prompt).encode()).hexdigest()
    if k in cache and cache[k] is not None:
        return k, cache[k]
    body = {"model": JUDGE, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    for _ in range(3):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(),
                {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=180))
            m = ANS.search(r["choices"][0]["message"]["content"])
            if m:
                return k, m.group(1).upper()
        except Exception:
            continue
    return k, None


def marg(rec):
    p = rec["pfx"]
    return [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]


def ols(target, sup):
    rows = [(r[0], r[1], 1) for r in target] + [(r[0], r[1], 0) for r in sup]
    ks = np.array([r[0] for r in rows]); y = np.array([r[1] for r in rows])
    h = np.array([r[2] for r in rows], float)
    kv = sorted(set(ks.tolist()))
    X = np.column_stack([(ks == kk).astype(float) for kk in kv] + [h])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b; dof = len(y) - X.shape[1]
    se = float(np.sqrt((resid @ resid) / dof * np.linalg.inv(X.T @ X)[-1, -1]))
    return float(b[-1]), float(2 * stats.t.sf(abs(b[-1] / se), dof)), len(target)


def main():
    man = {c["ci"]: c for c in json.load(open(SE / "data" / "manifest.json"))["contexts"]}
    faith = json.load(open(HERE / "results" / "judged_faithfulness.json"))
    reason = json.load(open(HERE / "results" / "quote_reason.json"))

    items = {}
    for arm, fn in [("mat", "subset_scores_mat.json"), ("std", "subset_scores_std.json")]:
        items[arm] = []
        for e in json.load(open(HERE / "results" / fn))["entries"]:
            for ri, rec in enumerate(e["rollouts"]):
                if not rec:
                    continue
                m = marg(rec)
                for k in range(len(m)):
                    key = f"{arm}|{e['ci']}|{ri}|{k}"
                    v = faith.get(key)
                    if v is not None:
                        items[arm].append(dict(key=key, ci=e["ci"], pos=k, mg=m[k],
                                               v=v, reason=reason.get(key), text=rec["units"][k]))

    # classify substantive hallucinations only
    jobs, meta = [], []
    for arm in ("mat", "std"):
        for it in items[arm]:
            if it["v"] in H and it["reason"] == "SUBSTANTIVE":
                jobs.append(PROMPT.format(passage=man[it["ci"]]["prefix_text"],
                                          continuation=man[it["ci"]]["answer_text"], note=it["text"]))
                meta.append(it["key"])
    print(f"[judge] {len(jobs)} substantive hallucinations → PREDICTION vs GENUINE…", flush=True)
    sev = {}
    with ThreadPoolExecutor(max_workers=int(os.environ.get("JUDGE_WORKERS", "200"))) as ex:
        res = []
        for c0 in range(0, len(jobs), 500):
            res += list(ex.map(ask, jobs[c0:c0 + 500]))
            for kk, vv in res[c0:]:
                cache[kk] = vv
            json.dump(cache, open(CACHE, "w"))
            print(f"  {min(c0 + 500, len(jobs))}/{len(jobs)}", flush=True)
    for key, (_, v) in zip(meta, res):
        sev[key] = v
    json.dump(sev, open(HERE / "results" / "severity.json", "w"))
    print(f"[judge] done, {sum(v is None for v in sev.values())} unparsed", flush=True)

    out = {}
    print("\n" + "=" * 74)
    for arm, label, fn in [("mat", "matryoshka (lines)", None), ("std", "standard (sentences)", None)]:
        rows = items[arm]
        sup = [(it["pos"], it["mg"]) for it in rows if it["v"] == "SUPPORTED"]
        genuine = [(it["pos"], it["mg"]) for it in rows
                   if it["reason"] == "SUBSTANTIVE" and sev.get(it["key"]) == "GENUINE_FABRICATION"]
        pred = [(it["pos"], it["mg"]) for it in rows
                if it["reason"] == "SUBSTANTIVE" and sev.get(it["key"]) == "UNVERBALIZED_PREDICTION"]
        cg, pg, ng = ols(genuine, sup)
        cp, pp, npd = ols(pred, sup)
        mg_, mp, ms = np.mean([r[1] for r in genuine]), np.mean([r[1] for r in pred]), np.mean([r[1] for r in sup])
        print(f"\n{label}")
        print(f"  substantive split: GENUINE {len(genuine)} | UNVERBALIZED_PREDICTION {len(pred)}  (SUPPORTED {len(sup)})")
        print(f"  mean marginal: GENUINE {mg_:+.4f} | PREDICTION {mp:+.4f} | SUPPORTED {ms:+.4f}")
        print(f"  within-position OLS vs SUPPORTED:  GENUINE(worst) coef {cg:+.4f} p={pg:.2e}  | PREDICTION coef {cp:+.4f} p={pp:.3f}")
        out[arm] = dict(n_genuine=len(genuine), n_prediction=len(pred), n_supported=len(sup),
                        mean_genuine=float(mg_), mean_prediction=float(mp), mean_supported=float(ms),
                        ols_genuine=[cg, pg], ols_prediction=[cp, pp])

        # figure: position curves SUPPORTED / prediction / genuine
        ks = np.array([it["pos"] for it in rows]); kmax = int(np.percentile(ks, 99))
        xs = list(range(kmax + 1))
        def curve(rr):
            by = {}
            for pos, mgv in rr:
                by.setdefault(pos, []).append(mgv)
            return [np.mean(by[k]) if k in by else np.nan for k in xs]
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        ax.plot(xs, curve(sup), "-o", color=GREEN, ms=5, lw=2, label=f"faithful — SUPPORTED (n={len(sup)})")
        ax.plot(xs, curve(pred), "-o", color=ORANGE, ms=5, lw=2, label=f"unverbalized prediction (n={len(pred)})")
        ax.plot(xs, curve(genuine), "-o", color=DARK, ms=5, lw=2, label=f"GENUINE fabrication — worst (n={len(genuine)})")
        ax.axhline(0, color="#444", lw=0.8)
        ax.set_xlabel("item position", fontsize=11); ax.set_ylabel("mean marginal FVE", fontsize=11)
        ax.set_title(f"{label} — the worst hallucinations vs plausible unverbalized predictions\n"
                     f"GENUINE-vs-SUPPORTED within position: coef {cg:+.4f}, p={pg:.1e}", fontsize=11.5)
        ax.legend(fontsize=9); ax.grid(color="#ccc", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        import textwrap
        foot = ("Within the substantive (non-misquote) hallucinations: GENUINE = a real confabulation; "
                "UNVERBALIZED_PREDICTION = plausibly describes the model's not-yet-written internal direction "
                "(nex-n2-mini). The prediction curve tracks SUPPORTED; the genuine-confabulation curve sits below it.")
        fig.subplots_adjust(bottom=0.24)
        fig.text(0.02, 0.02, "\n".join(textwrap.wrap(foot, width=118)),
                 fontsize=7.6, color="#777", ha="left", va="bottom")
        fig.savefig(HERE / "results" / f"fig_severity_{arm}.png", dpi=150)
        plt.close(fig)
        print(f"  [saved] fig_severity_{arm}.png")
    json.dump(out, open(HERE / "results" / "severity_stats.json", "w"), indent=1)
    print("\n[saved] results/severity.json, severity_stats.json")


if __name__ == "__main__":
    main()
