"""Gate G pure-text pre-test arms (user's AV2/AR2 idea, frozen-AV side).

Tests the AR2 side with FROZEN AVs: reconstruct h24 from TEXT only (no vector
injection). Three arms on the common positions (both AV texts present):

  ptA: AV(h24)-text only               -> after-description (the diff carrier)
  ptC: AV(h16)-text only               -> before-description (the s-control)
  ptB: AV(h16) [before] + AV(h24) [after]  -> the user's AR2 input

Key reads (paired bootstrap vs acts24, no legibility confound):
  ptA - ptC : pure-text diff-specificity (cf. Gate G vector-conditioned +0.0128)
  ptB - ptA : does the before-text add anything beyond the after-text?
"""
import glob
import json
from pathlib import Path

W = Path("/work/gate_g")
meta = json.load(open("/work/gate_c/meta2.json"))


def load(d):
    av = {}
    for fn in glob.glob(f"{d}/*.jsonl"):
        for l in open(fn):
            r = json.loads(l); av[r["i"]] = r.get("explanation")
    return av


h16 = load("av_h16_out"); h24 = load("av_h24_out")
def has(i): return bool((h16.get(i) or "").strip()) and bool((h24.get(i) or "").strip())

arms = {
    "ptA": lambda i: h24[i],
    "ptC": lambda i: h16[i],
    "ptB": lambda i: f"[before] {h16[i]} [after] {h24[i]}",
}
for arm, fn in arms.items():
    for split, tag in (("train", "train"), ("holdout", "eval")):
        rows = [{"text": fn(i), "tidx": i} for i, m in enumerate(meta)
                if m["split"] == split and has(i)]
        (W / f"arm_{arm}_{tag}.json").write_text(json.dumps(rows))
        print(arm, tag, len(rows))
