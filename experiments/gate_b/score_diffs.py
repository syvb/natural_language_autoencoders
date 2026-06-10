"""Gate B step 2 (runs on a GPU pod): AR-reconstruct every diff label.

Feeds each Claude-produced difference text through the released L20 AR
exactly like a normal explanation; saves the predicted vectors so the
grounding analysis (cos vs true diff, with controls) runs locally.
"""

import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic

critic = NLACritic(WORK / "models" / "ar", device="cuda")
labels = json.loads((WORK / "gate_b_labels.json").read_text())

conds = sorted(labels.keys())
n = len(labels[conds[0]])
recons = np.zeros((len(conds), n, 3584), dtype=np.float16)
ok = np.zeros((len(conds), n), dtype=bool)

for ci, cond in enumerate(conds):
    for i, r in enumerate(labels[cond]):
        if r.get("difference"):
            recons[ci, i] = critic.reconstruct(r["difference"]).numpy().astype(np.float16)
            ok[ci, i] = True
    print(f"[score_diffs] {cond}: {int(ok[ci].sum())}/{n}", flush=True)

np.save(WORK / "diff_recons.npy", recons)
np.save(WORK / "diff_recons_ok.npy", ok)
(WORK / "diff_recons_conds.json").write_text(json.dumps(conds))
print("[score_diffs] saved", recons.shape)
