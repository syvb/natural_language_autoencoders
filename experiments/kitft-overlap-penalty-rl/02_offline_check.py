"""Pre-launch validation of the copied-bits penalty — runs the REAL scorer.

Imports nla.reward itself (ray/miles stubbed — no GPU stack needed) with the
production env knobs and the real unigram table, then:
  1. verifies the RL parquet carries prompt + activation_vector +
     detokenized_text_truncated and the v1 tagged template,
  2. scores the 16-sample calibration set (pre-RL / iter-50 / gold / shuffled)
     with _copied_bits — the exact function the reward path will call,
  3. prints the implied step-0 mean penalty at the chosen λ.

Run after 01:  python 02_offline_check.py  (dev box; needs OUT_DIR artifacts
plus eval/compare_pre_vs_50.md + av_eval.parquet for the calibration texts)
"""
import os
import re
import statistics
import sys
from pathlib import Path
from unittest.mock import MagicMock

OUT_DIR = Path(os.environ.get("OUT_DIR", os.path.expanduser("~/overlap_build")))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("NLA_OVERLAP_PENALTY", "0.01")
os.environ.setdefault("NLA_OVERLAP_DEDUCTIBLE", "15")
os.environ.setdefault("NLA_OVERLAP_MIN_SPAN", "2")
os.environ.setdefault("NLA_OVERLAP_UNIGRAMS", str(OUT_DIR / "unigrams.json"))

# stub the GPU-stack imports so nla.reward loads on a CPU box
for mod in ("ray", "miles", "miles.utils", "miles.utils.processing_utils",
            "miles.utils.types"):
    sys.modules.setdefault(mod, MagicMock())
from nla.reward import _copied_bits, _OVERLAP_PENALTY  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402
import yaml  # noqa: E402

print("=== 1. parquet plumbing ===")
rl = OUT_DIR / "rl_tagged_ctx.parquet"
pf = pq.ParquetFile(rl)
cols = pf.schema_arrow.names
for need in ("prompt", "activation_vector", "detokenized_text_truncated", "doc_id"):
    assert need in cols, f"missing column {need}: {cols}"
side = yaml.safe_load(open(f"{rl}.nla_meta.yaml"))
tpl = side["prompt_templates"]["actor"]
assert "2-3 text snippets" in tpl and "<explanation>" in tpl, (
    f"not the v1 tagged template: {tpl[:120]!r}")
head = pq.read_table(rl, columns=["detokenized_text_truncated"]).column(0)
n_empty = sum(1 for i in range(min(2000, len(head))) if not head[i].as_py())
assert n_empty == 0, f"{n_empty} empty contexts in first 2000 rows"
print(f"OK: {pf.metadata.num_rows} rows, v1 tagged template, contexts present")

print("=== 2. scorer on the calibration set (REAL _copied_bits) ===")
md = open(REPO_ROOT / "experiments/kitft-quote-penalty-rl/eval/compare_pre_vs_50.md",
          encoding="utf-8").read()
blocks = re.split(r"\n## Sample \d+/\d+ — `(.+?)`\n", md)[1:]
samples = [(cid, dict(re.findall(
    r"\*\*(\w+)\*\* \(quote chars: \d+\):\n```text\n(.*?)\n```", body, re.S)))
    for cid, body in zip(blocks[0::2], blocks[1::2])]
ev = pq.read_table(OUT_DIR / "av_eval.parquet",
                   columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated", "response"])
idx = {f"{d}@{n}": i for i, (d, n) in enumerate(
    zip(ev.column("doc_id").to_pylist(), ev.column("n_raw_tokens").to_pylist()))}
ctx_of = lambda cid: ev.column("detokenized_text_truncated")[idx[cid]].as_py()
def gold_of(cid):
    g = ev.column("response")[idx[cid]].as_py()
    m = re.search(r"<explanation>\n?(.*?)\n?</explanation>", g, re.S)
    return m.group(1) if m else g

conds = {}
for name, gen in (("preRL", lambda s, g: g["preRL"]), ("iter50", lambda s, g: g["iter50"]),
                  ("gold", lambda s, g: None), ("shuffled", lambda s, g: None)):
    xs = []
    for si, (cid, gens) in enumerate(samples):
        if name == "gold":
            xs.append(_copied_bits(gold_of(cid), ctx_of(cid)))
        elif name == "shuffled":
            xs.append(_copied_bits(gens["preRL"], ctx_of(samples[(si + 1) % 16][0])))
        else:
            xs.append(_copied_bits(gens[name], ctx_of(cid)))
    conds[name] = xs
    nz = sum(1 for x in xs if x > 0.5)
    print(f"  {name:<9} mean={statistics.mean(xs):6.1f}b  median={statistics.median(xs):6.1f}b  nonzero={nz}/16")

assert statistics.mean(conds["shuffled"]) < 2.0, "chance floor not ~0 — check deductible/table"
assert statistics.mean(conds["gold"]) > statistics.mean(conds["preRL"]) > 5.0, (
    "ordering gold > preRL > floor violated")
lam = _OVERLAP_PENALTY
m = statistics.mean(conds["preRL"])
print(f"=== 3. λ = {lam}: implied step-0 mean penalty ≈ {lam * m:.3f} "
      f"(target ~0.15–0.3; within-group reward spread ~0.1) ===")
assert 0.1 <= lam * m <= 0.4, f"step-0 penalty {lam * m:.3f} outside sane band — retune λ"
print("OFFLINE_CHECK_PASS")
