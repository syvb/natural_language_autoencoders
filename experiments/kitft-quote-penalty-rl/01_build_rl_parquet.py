"""Build the tagged (v1) RL parquet for the quote-penalty run, box-side.

Downloads `base_av.parquet` from the sonnet46 warm-start dataset (raw vectors,
norm="none" — the same Qwen2.5-7B-Instruct L20 activations the kitft pair was
trained on), reproduces the seed-42 DOCUMENT-level split bit-identically to
`../qwen2.5-matryoshka-warmstart-sonnet46/02_build_datasets.py` (same seed, same
EVAL_TARGET, same doc iteration order), keeps the TRAIN half, and runs
`stage3_build --stage rl --explanation-format tagged` so the RL prompt is the v1
template the kitft AV was trained on (asserted against its sidecar at RL startup).

Verification: the eval half's row count must equal the published
`av_eval.parquet`'s (same split ⇒ RL data is doc-disjoint from the FVE holdout).

Run after setup:  python 01_build_rl_parquet.py
Outputs: /workspace/out/base_av_train_kitftrl.parquet, /workspace/out/rl_tagged.parquet
"""
import argparse
import os
import random
import subprocess
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from nla.datagen._common import make_storage
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar

DATA_REPO = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
OUT_DIR = "/workspace/out"
TRAIN_BASE = f"{OUT_DIR}/base_av_train_kitftrl.parquet"
RL_OUT = os.environ.get("RL_PARQUET", f"{OUT_DIR}/rl_tagged.parquet")
# The tokenizer stage3_build loads comes from this sidecar field — must be a
# path/id resolvable ON THE BOX. The instruct model dir is downloaded by setup.
BASE_MODEL = os.environ.get("BASE_MODEL", "/workspace/models/qwen2.5-7b-instruct")
# MUST match 02_build_datasets.py or the doc split (and hence train/eval
# disjointness with the published av_eval) silently changes.
EVAL_TARGET = 5000
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
tok = open("/root/.hf_token").read().strip()

print("=== download base_av.parquet + av_eval.parquet ===", flush=True)
base_av = hf_hub_download(DATA_REPO, "base_av.parquet", repo_type="dataset",
                          token=tok, local_dir=OUT_DIR)
for f in ("av_eval.parquet", "av_eval.parquet.nla_meta.yaml"):
    hf_hub_download(DATA_REPO, f, repo_type="dataset", token=tok, local_dir=OUT_DIR)

print("=== seed-42 document-level split (identical to 02_build_datasets) ===", flush=True)
tbl = pq.read_table(base_av)
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
print(f"total={tbl.num_rows} train={len(train_idx)} eval={len(eval_idx)}", flush=True)

n_eval_published = pq.ParquetFile(f"{OUT_DIR}/av_eval.parquet").metadata.num_rows
assert len(eval_idx) == n_eval_published, (
    f"reproduced eval half has {len(eval_idx)} rows but the published av_eval has "
    f"{n_eval_published} — the split did NOT reproduce; RL data would leak into the "
    f"FVE holdout. Do not proceed."
)
print(f"split verified: eval half == published av_eval ({n_eval_published} rows)", flush=True)

sub = tbl.take(pa.array(train_idx))
pq.write_table(sub, TRAIN_BASE)
storage = make_storage(argparse.Namespace(
    storage_cls="nla.datagen.storage.LocalStorage", storage_kwargs=None))
write_sidecar(storage, TRAIN_BASE, NLADatasetMeta(
    dataset_id="base_qwen2.5-7b-L20_matryoshka-sonnet46_av_train_kitftrl",
    stage="base", row_count=sub.num_rows,
    extraction=NLAExtractionMeta(
        base_model=BASE_MODEL, d_model=3584, layer_index=20, norm="none",
        corpus="openbmb/Ultra-FineWeb", corpus_slice={"start": 0, "length": 100000},
        positions_per_doc=10),
    created_by="kitft-quote-penalty-rl/01_build_rl_parquet.py (seed-42 doc-split train half)",
))
print(f"wrote {TRAIN_BASE} ({sub.num_rows} rows)", flush=True)

print(f"=== stage3_build rl (tagged): {TRAIN_BASE} -> {RL_OUT} ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "nla.datagen.stage3_build",
     "--stage", "rl", "--input", TRAIN_BASE, "--output", RL_OUT,
     "--explanation-format", "tagged"],
    capture_output=True, text=True,
)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("STDERR:\n", r.stderr[-3000:])
    raise SystemExit("stage3_build rl failed")
print(f"RL_PARQUET_BUILT {RL_OUT}", flush=True)
