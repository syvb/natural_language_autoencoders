"""Gate D2 arm assembly (code review M2/M3).

Builds matched-n pair files {text, tidx} (tidx indexes attn_out.npy rows):
  armC — content control: cpre.jsonl reads of v_pre (s=0)
  armZ — zero-shot AV reads of the write: attn_read.jsonl s=0 ONLY
         (2 samples/pos exist; eval must be one-per-tidx for evaluate_d)
  armT — templated mechanical labels (from build_bundles_d's armT files)
  armH — Haiku labels (armH_labels.json; run after the batch + regen pass)

Pre-registration: matched n=8000 train per arm; quiet positions capped at a
5% stratum (natural rate ~10%); null/unparsed labels DROPPED (never raw-
substituted); CJK/garble filter on generated reads. Train positions are the
intersection of clean positions across all arms present, so every arm
trains on identical positions. Eval = all clean holdout positions per arm.
"""

import json
import re
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
CJK = re.compile(r"[぀-ヿ一-鿿]")
MATCHED_N = 8000
QUIET_FRAC = 0.05
SEED = 0

meta = json.loads((WORK / "meta_d.json").read_text())
bundles = json.loads((WORK / "bundles_d.json").read_text())
quiet = {int(i) for i, b in bundles.items() if b["quiet"]}


def clean(t):
    return t and not CJK.search(t) and len(t) > 40


def load_gen(name, s_keep=0):
    out = {}
    p = WORK / "gend_out" / f"{name}.jsonl"
    for line in p.open():
        r = json.loads(line)
        if r["s"] != s_keep:
            continue
        t = (r["explanation"] or "").strip()  # M3: drop nulls, no raw fallback
        if clean(t):
            out[r["i"]] = t
    return out


def load_labels(path):
    out = {}
    if not Path(path).exists():
        return out
    for i, r in json.loads(Path(path).read_text()).items():
        t = (r.get("label") or "").strip()
        if clean(t):
            out[int(i)] = t
    return out


arms = {
    "armC": load_gen("cpre"),
    "armZ": load_gen("attn_read"),
    "armT": {r["tidx"]: r["text"]
             for r in json.loads((WORK / "armT_train.json").read_text())
             + json.loads((WORK / "armT_eval.json").read_text())},
    "armH": load_labels(WORK / "armH_labels.json"),
}
arms = {k: v for k, v in arms.items() if v}
print("arms present:", {k: len(v) for k, v in arms.items()})

train_all = [i for i, m in enumerate(meta) if m["split"] == "train"]
hold = [i for i, m in enumerate(meta) if m["split"] == "holdout"]
common = [i for i in train_all if all(i in a for a in arms.values())]
rng = np.random.default_rng(SEED)
q = [i for i in common if i in quiet]
nq = [i for i in common if i not in quiet]
n_quiet = min(len(q), int(MATCHED_N * QUIET_FRAC))
picked = (rng.choice(nq, min(len(nq), MATCHED_N - n_quiet), replace=False).tolist()
          + rng.choice(q, n_quiet, replace=False).tolist())
picked = sorted(picked)
print(f"train positions: {len(picked)} ({n_quiet} quiet) "
      f"from {len(common)} common-clean")

for name, a in arms.items():
    tr = [{"text": a[i], "tidx": i} for i in picked]
    ev = [{"text": a[i], "tidx": i} for i in hold if i in a]
    assert len({r["tidx"] for r in ev}) == len(ev), f"{name}: dup eval tidx"
    (WORK / f"{name}_train8k.json").write_text(json.dumps(tr))
    (WORK / f"{name}_eval8k.json").write_text(json.dumps(ev))
    print(f"{name}: train {len(tr)}, eval {len(ev)}")
