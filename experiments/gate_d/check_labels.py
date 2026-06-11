"""Mechanical label checker for the regen-on-violation pass (code review B1).

Checks each label for: parse failure, banned/mechanism/speculation words,
numeric weights, template opener, future-token leak (quoted word that only
occurs after the position), quiet-position length violation.

Usage:
  python check_labels.py --labels armH_labels.json [--out regen_ids.json]
Prints a violation breakdown; writes the offending position ids.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

WORK = Path(__file__).parent
WORD = re.compile(r"[a-zA-Z]{3,}")
BANNED = re.compile(
    r"\b(layer|vector|representation|neural|head|heads|channel|channels|"
    r"path|paths|share|shares|weight|weights|weighted|attention|retrieval|"
    r"retrievals|suggesting|appears to|likely|preparing|setting up|about to|"
    r"the model|the network)\b", re.I)
NUMS = re.compile(r"\b0\.\d+|\b\d+%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=str(WORK / "regen_ids.json"))
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    bundles = json.loads((WORK / "bundles_d.json").read_text())
    meta = json.loads((WORK / "meta_d.json").read_text())
    doc_tokens = {}
    for line in (WORK / "doc_tokens.jsonl").open():
        r = json.loads(line)
        doc_tokens[r["doc_idx"]] = r["ids"]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    reasons = Counter()
    bad = {}
    for i, r in labels.items():
        t = r.get("label")
        why = []
        if not t:
            why.append("unparsed")
        else:
            if BANNED.search(t):
                why.append("banned-word")
            if NUMS.search(t):
                why.append("numeric-weight")
            if t.strip().lower().startswith(("the write", "the attention")):
                why.append("template-opener")
            m = meta[int(i)]
            ids = doc_tokens[m["doc_idx"]]
            pos = m["pos"]
            past = {w.lower() for w in WORD.findall(tok.decode(ids[:pos + 1]))}
            fut = {w.lower()
                   for w in WORD.findall(tok.decode(ids[pos + 1:pos + 12]))} - past
            qw = {w.lower() for q in re.findall(r'"([^"]+)"', t)
                  for w in WORD.findall(q)}
            if qw & fut:
                why.append("future-leak")
            if bundles[i].get("quiet") and t.count(".") > 3:
                why.append("quiet-too-long")
        if why:
            bad[i] = why
            reasons.update(why)
    Path(args.out).write_text(json.dumps(sorted(bad, key=int)))
    print(f"[check] {len(bad)}/{len(labels)} flagged -> {args.out}")
    for k, v in reasons.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
