"""Stage B — document-level train/eval split + stage3_build into trainable parquets.

- Seeds the injection-token cache so the dataset sidecar matches the model (㈎ / 149705).
- Holds out ~5k samples per split (av, ar) at the DOCUMENT level (whole docs, seed 42),
  so no eval position shares a document with a training position (~10k total held out).
- Runs the stock nla.datagen.stage3_build on each base parquet -> stage-3 training format.

Produces: av_sft / av_eval / ar_sft / ar_eval parquets (+ .nla_meta.yaml sidecars).
Run on the same box after 01_extract_activations.py:  python 02_build_datasets.py
"""
import subprocess, sys, random, argparse, os
import pyarrow as pa, pyarrow.parquet as pq
from collections import defaultdict
from nla.datagen._common import load_tokenizer, make_storage
from nla.datagen.injection_tokens import _load_cache, _save_cache
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar

MODEL = os.environ.get("BASE_MODEL", "/workspace/models/qwen2.5-7b-instruct")
EVAL_TARGET = 5000   # per split (av, ar) -> ~10k total held out
SEED = 42
storage = make_storage(argparse.Namespace(storage_cls="nla.datagen.storage.LocalStorage", storage_kwargs=None))

# 1) seed injection cache (㈎ / 149705) so the dataset sidecar matches the model
tok = load_tokenizer(MODEL)
ids = tok("㈎", add_special_tokens=False)["input_ids"]
print("㈎ ->", ids, "name_or_path:", tok.name_or_path, flush=True)
if len(ids) == 1:
    c = _load_cache(); c[tok.name_or_path] = {"char": "㈎", "token_id": ids[0]}; _save_cache(c); print("seeded cache", flush=True)

def base_meta(split_id, n):
    return NLADatasetMeta(
        dataset_id=f"base_qwen2.5-7b-L20_matryoshka-sonnet46_{split_id}",
        stage="base", row_count=n,
        extraction=NLAExtractionMeta(base_model=MODEL, d_model=3584, layer_index=20, norm="none",
            corpus="openbmb/Ultra-FineWeb", corpus_slice={"start": 0, "length": 100000}, positions_per_doc=10),
        created_by="01_extract_activations.py (last-token activation of input_text)")

def doc_split(tbl):
    """Document-level holdout: shuffle docs, take whole docs until >=EVAL_TARGET rows."""
    docs = tbl.column("doc_id").to_pylist()
    by_doc = defaultdict(list)
    for i, d in enumerate(docs): by_doc[d].append(i)
    order = list(by_doc); random.Random(SEED).shuffle(order)
    eval_idx = []
    for d in order:
        if len(eval_idx) >= EVAL_TARGET: break
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
    tbl = pq.read_table(f"/workspace/out/base_{split}.parquet")
    tr, ev = doc_split(tbl)
    print(f"  total={tbl.num_rows} train={len(tr)} eval={len(ev)} (eval docs whole)", flush=True)
    write_subset(tbl, tr, f"/workspace/out/base_{split}_train.parquet", f"{split}_train")
    write_subset(tbl, ev, f"/workspace/out/base_{split}_eval.parquet",  f"{split}_eval")

def run(stage, inp, out):
    print(f"\n=== stage3_build {stage} {inp} ===", flush=True)
    r = subprocess.run([sys.executable, "-m", "nla.datagen.stage3_build", "--input", inp, "--stage", stage, "--output", out],
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0: print("STDERR:\n", r.stderr[-3000:]); raise SystemExit(f"stage3_build failed: {stage} {inp}")

run("av_sft", "/workspace/out/base_av_train.parquet", "/workspace/out/av_sft.parquet")
run("av_sft", "/workspace/out/base_av_eval.parquet",  "/workspace/out/av_eval.parquet")
run("ar_sft", "/workspace/out/base_ar_train.parquet", "/workspace/out/ar_sft.parquet")
run("ar_sft", "/workspace/out/base_ar_eval.parquet",  "/workspace/out/ar_eval.parquet")
print("\nBUILD_DONE", flush=True)
