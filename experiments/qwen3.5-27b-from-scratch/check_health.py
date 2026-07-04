"""Run-health check: parse a miles train log and assert the run is sane.

Usage:
    python check_health.py /path/to/train.log [--rl | --sft] [--taper]

Scans for the metric keys the NLA stack emits (raw_reward, fve_nrm,
loss_nonfinite, kl_loss, kl_flat, pred_norm_min, grad-guard skip lines) and
prints first/last/min/max per key plus a PASS/WARN/FAIL verdict:

  FAIL  loss_nonfinite > 0 anywhere            (weights may be poisoned)
  FAIL  raw_reward pinned at -2.0 (>=3 in a row) or non-finite
  FAIL  [--taper] kl_flat absent               (tapered loss did not dispatch)
  WARN  grad-guard skips > 40% of optimizer steps
  WARN  fve_nrm not improving (last <= first) over a >=20-step window
  PASS  otherwise

Exit code 0 on PASS/WARN, 1 on FAIL — safe to gate automation on.
"""
import re
import sys
from collections import defaultdict

WATCH = (
    "raw_reward", "fve_nrm", "loss_nonfinite", "kl_loss", "kl_flat",
    "pred_norm_min", "values_absmax", "pg_loss", "entropy_loss", "grad_norm",
    "loss",
)
# matches  'key': -1.23,  key=1.23,  "train/key": 1e-4
_PAT = re.compile(
    r"['\"]?([A-Za-z_][A-Za-z0-9_/]*)['\"]?\s*[:=]\s*"
    r"(-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)"
)
_SKIP_PAT = re.compile(r"SKIPPED optimizer step")
_STEP_PAT = re.compile(r"rollout_id[:=\s']+(\d+)|step[:=\s']+(\d+)", re.I)


def main():
    args = [a for a in sys.argv[1:]]
    taper = "--taper" in args
    path = next((a for a in args if not a.startswith("--")), None)
    if not path:
        sys.exit(__doc__)

    hist = defaultdict(list)
    skips = 0
    steps_seen = set()
    for line in open(path, errors="replace"):
        if _SKIP_PAT.search(line):
            skips += 1
        m = _STEP_PAT.search(line)
        if m:
            steps_seen.add(next(g for g in m.groups() if g))
        for key, val in _PAT.findall(line):
            short = key.split("/")[-1]
            if short in WATCH:
                try:
                    hist[short].append(float(val))
                except ValueError:
                    hist[short].append(float("nan"))

    if not hist:
        print(f"NO METRICS parsed from {path} — wrong file, or the run "
              f"produced no train-step logs yet.")
        sys.exit(1)

    print(f"{'metric':16s} {'n':>5s} {'first':>12s} {'last':>12s} {'min':>12s} {'max':>12s}")
    for k in WATCH:
        v = hist.get(k)
        if not v:
            continue
        print(f"{k:16s} {len(v):5d} {v[0]:12.4g} {v[-1]:12.4g} {min(v):12.4g} {max(v):12.4g}")
    n_steps = max(len(steps_seen), len(hist.get("loss", [])), 1)
    print(f"grad-guard skips: {skips} (~{skips / n_steps:.0%} of ~{n_steps} steps)")

    failures, warnings = [], []

    lnf = hist.get("loss_nonfinite", [])
    if any(x > 0 for x in lnf):
        failures.append(f"loss_nonfinite>0 on {sum(x > 0 for x in lnf)} steps")

    rr = hist.get("raw_reward", [])
    if rr:
        if any(x != x for x in rr):
            failures.append("raw_reward is NaN")
        run = 0
        for x in rr:
            run = run + 1 if x <= -1.999 else 0
            if run >= 3:
                failures.append("raw_reward pinned at -2.0 (reward path dead / critic diverged)")
                break

    if taper:
        if "kl_flat" not in hist:
            failures.append("--taper: kl_flat metric absent — nla_policy_loss_tapered_kl "
                            "did NOT dispatch (check NLA_KL_TAPER_HALF_LIFE reached the actor)")
        elif hist.get("kl_loss") and hist.get("kl_flat"):
            kw, kf = hist["kl_loss"][-1], hist["kl_flat"][-1]
            if kf > 0 and not (kw < kf):
                warnings.append(f"tapered kl_loss ({kw:.4g}) not < kl_flat ({kf:.4g}) "
                                f"— taper weights may not be applying")
            else:
                print(f"taper active: kl_loss/kl_flat = {kw / kf:.2f}" if kf else "")

    if skips > 0.4 * n_steps:
        warnings.append(f"grad-guard skipping {skips / n_steps:.0%} of steps — "
                        f"learning is starved; investigate before a long run")

    fv = hist.get("fve_nrm", [])
    if len(fv) >= 20 and fv[-1] <= fv[0]:
        warnings.append(f"fve_nrm not improving ({fv[0]:.3f} -> {fv[-1]:.3f})")

    for w in warnings:
        print("WARN:", w)
    for f in failures:
        print("FAIL:", f)
    if failures:
        print("VERDICT: FAIL")
        sys.exit(1)
    print("VERDICT:", "WARN" if warnings else "PASS")


if __name__ == "__main__":
    main()
