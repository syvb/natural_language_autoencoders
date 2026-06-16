"""Gate G Stage 0: build critic arms.

All arms reconstruct h_22 (acts22) while CONDITIONED on h_20 (acts20, injected
at the critic's marker token). They differ only in the explanation text:

  none : ""                       -> baseline; critic predicts h_22 from h_20 alone.
  cat  : raw 60-token context tail -> the control that must be beaten (program-wide).
  text : AV's description of h_22   -> the treatment; should carry h_22-info beyond
                                       h_20 AND beyond raw context.

value-add = cos(text) - cos(none);  the real win = cos(text) - cos(cat) > 0.
tidx indexes acts20.npy / acts22.npy rows (aligned to gate_c/meta2.json).
Doc-level split is inherited from meta2 (train / holdout).
"""
import json
from pathlib import Path

WORK = Path(__file__).parent
GC = WORK.parent / "gate_c"

meta = json.loads((GC / "meta2.json").read_text())
av = {}
for fn in (WORK / "av_out").glob("av*.jsonl"):
    for line in open(fn):
        r = json.loads(line)
        av[r["i"]] = r.get("explanation")

missing = [i for i in range(len(meta)) if i not in av or not av[i]]
print(f"[build] {len(meta)} positions; AV explanations present {len(av)}; "
      f"missing/empty {len(missing)}")

arms = {"none": lambda i, m: "",
        "cat": lambda i, m: m["context_tail"],
        "text": lambda i, m: av.get(i) or ""}

for arm, fn in arms.items():
    for split, tag in (("train", "train"), ("holdout", "eval")):
        rows = [{"text": fn(i, m), "tidx": i}
                for i, m in enumerate(meta)
                if m["split"] == split
                # for the text arm, drop positions with no usable AV explanation
                and not (arm == "text" and not (av.get(i) or "").strip())]
        out = WORK / f"arm_{arm}_{tag}.json"
        out.write_text(json.dumps(rows))
        print(f"  arm_{arm}_{tag}: {len(rows)} rows -> {out.name}")

print("[build] done")
