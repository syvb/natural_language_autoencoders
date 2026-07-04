"""Step 2 — document-level split + stage3_build into trainable parquets.

- Picks (and caches) the injection token for THIS tokenizer via
  find_injection_token — do not hardcode another model's marker char.
- Holds out EVAL_TARGET rows per split (av, ar) at the DOCUMENT level (whole
  docs, SPLIT_SEED) so no eval position shares a document with training.
- Builds the bullets-format v3 recipe directly:
    av_sft / av_eval        bullets prompt, plain one-line-per-item targets
    ar_sft                  critic inputs token-truncated K ~ U[MIN, MAX]
                            (matches RL's truncation draw; pre-calibrates the
                            critic for RL step 0), AR_KEEP_FULL_FRAC left full
    ar_eval                 untruncated (clean full-text eval)

Run after 01:  python 02_build_datasets.py
"""
import argparse
import os
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config

import pyarrow as pa
import pyarrow.parquet as pq

from nla.datagen._common import load_tokenizer, make_storage
from nla.datagen.injection_tokens import find_injection_token
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar

_config.load()
WORK = _config.env("WORK")
OUT = f"{WORK}/out"
MODEL = os.environ.get("BASE_MODEL_DIR", f"{WORK}/models/base")
MODEL_TAG = _config.env("MODEL_TAG")
LAYER = int(_config.env("LAYER_INDEX"))
D_MODEL = int(_config.env("D_MODEL"))
EVAL_TARGET = int(_config.env("EVAL_TARGET"))
SEED = int(_config.env("SPLIT_SEED"))
MIN_TOKENS = _config.env("NLA_TRUNC_MIN_TOKENS")
MAX_TOKENS = _config.env("NLA_TRUNC_MAX_TOKENS")
KEEP_FULL = _config.env("AR_KEEP_FULL_FRAC")
AR_SEED = _config.env("AR_TRUNCATE_SEED")

storage = make_storage(argparse.Namespace(
    storage_cls="nla.datagen.storage.LocalStorage", storage_kwargs=None))

# 1) injection token for this tokenizer (searched + cached; sidecar must match)
tok = load_tokenizer(MODEL)
char, tid = find_injection_token(tok)
print(f"injection token: {char!r} -> {tid}  ({tok.name_or_path})", flush=True)


def base_meta(split_id, n):
    return NLADatasetMeta(
        dataset_id=f"base_{MODEL_TAG}_matryoshka-sonnet46_{split_id}",
        stage="base", row_count=n,
        extraction=NLAExtractionMeta(
            base_model=MODEL, d_model=D_MODEL, layer_index=LAYER, norm="none",
            corpus="openbmb/Ultra-FineWeb",
            corpus_slice={"start": 0, "length": 100000}, positions_per_doc=10),
        created_by="01_extract_activations.py (last-token activation of input_text)")


def doc_split(tbl):
    """Document-level holdout: shuffle docs, take whole docs until >= EVAL_TARGET rows."""
    docs = tbl.column("doc_id").to_pylist()
    by_doc = defaultdict(list)
    for i, d in enumerate(docs):
        by_doc[d].append(i)
    order = list(by_doc)
    random.Random(SEED).shuffle(order)
    eval_idx = []
    for d in order:
        if len(eval_idx) >= EVAL_TARGET:
            break
        eval_idx.extend(by_doc[d])
    eval_set = set(eval_idx)
    train_idx = [i for i in range(tbl.num_rows) if i not in eval_set]
    return train_idx, eval_idx


def write_subset(tbl, idx, path, split_id):
    sub = tbl.take(pa.array(idx))
    pq.write_table(sub, path)
    write_sidecar(storage, path, base_meta(split_id, sub.num_rows))
    print(f"  wrote {path} ({sub.num_rows} rows)", flush=True)


for split in ("av", "ar"):
    print(f"\n=== split {split} ===", flush=True)
    tbl = pq.read_table(f"{OUT}/base_{split}.parquet")
    tr, ev = doc_split(tbl)
    print(f"  total={tbl.num_rows} train={len(tr)} eval={len(ev)} (eval docs whole)",
          flush=True)
    write_subset(tbl, tr, f"{OUT}/base_{split}_train.parquet", f"{split}_train")
    write_subset(tbl, ev, f"{OUT}/base_{split}_eval.parquet", f"{split}_eval")


def run(stage, inp, out, extra):
    print(f"\n=== stage3_build {stage} {inp} ===", flush=True)
    cmd = [sys.executable, "-m", "nla.datagen.stage3_build",
           "--input", inp, "--stage", stage, "--output", out,
           "--explanation-format", "bullets", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-3000:])
        raise SystemExit(f"stage3_build failed: {stage} {inp}")


print(f"\nAR truncation: K ~ U[{MIN_TOKENS}, {MAX_TOKENS}] tokens, "
      f"keep-full {KEEP_FULL}, seed {AR_SEED}", flush=True)
run("av_sft", f"{OUT}/base_av_train.parquet", f"{OUT}/av_sft.parquet", [])
run("av_sft", f"{OUT}/base_av_eval.parquet", f"{OUT}/av_eval.parquet", [])
run("ar_sft", f"{OUT}/base_ar_train.parquet", f"{OUT}/ar_sft.parquet",
    ["--ar-truncate-max-tokens", MAX_TOKENS, "--ar-truncate-min-tokens", MIN_TOKENS,
     "--ar-truncate-keep-full-frac", KEEP_FULL, "--ar-truncate-seed", AR_SEED])
run("ar_sft", f"{OUT}/base_ar_eval.parquet", f"{OUT}/ar_eval.parquet", [])
print("\nBUILD_DONE", flush=True)
