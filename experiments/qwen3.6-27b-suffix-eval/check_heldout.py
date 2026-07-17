"""Verify the eval docs are held out from the NLA's TRAINING corpus (no GPU).

The warmstart corpus `ceselder/nla-matryoshka-warmstart-sonnet46` stores its
source-document prefixes in `input_text`, so we can check overlap directly. For
every training row we hash the normalized 120-char document opening; then we
check each of the 250 eval docs' openings against that set. A raw hash-match can
be shared boilerplate (journal nav menus etc.), so any hit is confirmed with an
8-gram Jaccard against the matched training doc — a true doc match is ~1.0,
boilerplate is ~0.

Caveat this does NOT cover: the base model's own pretraining, which almost
certainly overlaps FineWeb/CommonCrawl and cannot be inspected.

    /home/debian/nanoNLA-multi-input/.venv-cpu/bin/python check_heldout.py
"""
import hashlib
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

HERE = Path(__file__).resolve().parent
TRAIN = "datasets/ceselder/nla-matryoshka-warmstart-sonnet46/data/train-00000-of-00001.parquet"


def norm(s, n=None):
    s = " ".join((s or "").split()).lower()
    return s[:n] if n else s


def shingles(s, k=8):
    w = norm(s).split()
    return set(" ".join(w[i:i + k]) for i in range(len(w) - k + 1))


def main():
    man = json.load(open(HERE / "data" / "manifest.json"))
    contexts = man["contexts"]
    fs = HfFileSystem(token=open("/home/debian/.hf_token").read().strip())
    pf = pq.ParquetFile(fs.open(TRAIN))

    # training doc openings (120-char) -> a representative full text + custom_id
    openings = {}
    docidx = []
    n = 0
    for b in pf.iter_batches(batch_size=20000, columns=["input_text", "custom_id"]):
        for t, cid in zip(b.column("input_text").to_pylist(), b.column("custom_id").to_pylist()):
            n += 1
            m = re.match(r"[a-z]+-(\d+)-", cid or "")
            if m:
                docidx.append(int(m.group(1)))
            o = norm(t, 120)
            if len(o) >= 40 and (o not in openings or len(t) > len(openings[o][1])):
                openings[o] = (cid, t)
    di = sorted(docidx)
    print(f"[train] {n} rows | doc-index range {di[0]}..{di[-1]} "
          f"(p50 {di[len(di)//2]}) | {len(openings)} distinct openings")

    genuine = 0
    for c in contexts:
        o = norm(c["prefix_text"], 120)
        if o in openings:
            cid, ttext = openings[o]
            jac = len(shingles(c["prefix_text"]) & shingles(ttext)) / \
                max(1, len(shingles(c["prefix_text"]) | shingles(ttext)))
            kind = "GENUINE DOC OVERLAP" if jac > 0.5 else "boilerplate only"
            print(f"  ci{c['ci']}: opening matches train {cid} | 8-gram Jaccard {jac:.3f} -> {kind}")
            genuine += jac > 0.5
    print(f"\n[eval] {len(contexts)} docs | genuine training-doc overlaps: {genuine}")
    print("NLA training: HELD OUT" if genuine == 0 else "NLA training: CONTAMINATED")
    print("NB base-model pretraining overlap is separate and unverifiable.")


if __name__ == "__main__":
    main()
