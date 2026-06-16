import json
from pathlib import Path
import numpy as np
W = Path("/work/gate_g")
rng = np.random.default_rng(123)
for split in ("train", "eval"):
    rows = json.loads((W / f"arm_text_{split}.json").read_text())
    texts = [r["text"] for r in rows]
    perm = rng.permutation(len(texts))
    for i in range(len(perm)):          # break fixed points (no self-pairing)
        if perm[i] == i:
            j = (i + 1) % len(perm); perm[i], perm[j] = perm[j], perm[i]
    shuf = [{"text": texts[perm[k]], "tidx": rows[k]["tidx"]} for k in range(len(rows))]
    (W / f"arm_textshuf_{split}.json").write_text(json.dumps(shuf))
    print(split, len(shuf), "shuffled")
