"""Build the overlap-penalty RL parquet + unigram table — LOCALLY (CPU, dev box).

Same tagged (v1) RL parquet as the quote-penalty run, with two additions:
  1. --keep-debug-metadata so every row carries detokenized_text_truncated —
     the input context the reward's copied-bits scorer matches against
     (NLADataSource already forwards it into sample.metadata).
  2. unigrams.json — corpus word counts for the surprisal pricing, built from
     the TRAIN split's documents, DEDUPED per doc (each doc appears as ~10
     truncation rows; keep only the longest so counts reflect the corpus, not
     the sampling multiplicity).

Streams everything (fits in ~2 GB RAM); verifies the seed-42 doc split against
the published av_eval row count (holdout disjointness), then uploads parquet +
sidecar + unigrams to the experiment's HF repo so GPU boxes just download.

Run:  python 01_build_rl_parquet_ctx.py            (dev box, nla-doll/.venv)
Env:  OUT_DIR (default ~/overlap_build), HF_TOKEN_FILE (default ~/.hf_token),
      SKIP_UPLOAD=1 to build only.
"""
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DATA_REPO = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
DEST_REPO = "syvb/nla-qwen2.5-7b-L20-rl-overlappen"
OUT_DIR = Path(os.environ.get("OUT_DIR", os.path.expanduser("~/overlap_build")))
TOKEN = open(os.environ.get("HF_TOKEN_FILE", os.path.expanduser("~/.hf_token"))).read().strip()
# MUST match 02_build_datasets.py / the quote-penalty run.
EVAL_TARGET, SEED = 5000, 42
# stage3_build reads the tokenizer from the input sidecar's base_model — must
# resolve locally AND on the box. The HF id does both (cache-seeded ㈎/149705).
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

OUT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_BASE = OUT_DIR / "base_av_train_overlap.parquet"
RL_OUT = OUT_DIR / "rl_tagged_ctx.parquet"
UNIGRAMS = OUT_DIR / "unigrams.json"

from huggingface_hub import HfApi, hf_hub_download

print("=== download base_av.parquet (+ av_eval row count for verification) ===", flush=True)
base_av = hf_hub_download(DATA_REPO, "base_av.parquet", repo_type="dataset",
                          token=TOKEN, local_dir=str(OUT_DIR))
av_eval = hf_hub_download(DATA_REPO, "av_eval.parquet", repo_type="dataset",
                          token=TOKEN, local_dir=str(OUT_DIR))

print("=== seed-42 doc split (streaming; identical to 02_build_datasets) ===", flush=True)
import random
pf = pq.ParquetFile(base_av)
docs = pq.read_table(base_av, columns=["doc_id"]).column("doc_id").to_pylist()
by_doc = defaultdict(list)
for i, d in enumerate(docs):
    by_doc[d].append(i)
order = list(by_doc)
random.Random(SEED).shuffle(order)
eval_idx: list[int] = []
eval_docs: set = set()
for d in order:
    if len(eval_idx) >= EVAL_TARGET:
        break
    eval_idx.extend(by_doc[d])
    eval_docs.add(d)
n_eval_published = pq.ParquetFile(av_eval).metadata.num_rows
assert len(eval_idx) == n_eval_published, (
    f"split did NOT reproduce: eval half {len(eval_idx)} != published av_eval "
    f"{n_eval_published}. RL data would leak into the FVE holdout — stopping."
)
print(f"split verified: {len(eval_idx)} eval rows == published av_eval; "
      f"{len(docs) - len(eval_idx)} train rows", flush=True)

print("=== write train subset (streaming, whole-doc filter) ===", flush=True)
writer = None
n_train = 0
for batch in pf.iter_batches(batch_size=4096):
    mask = pa.array([d not in eval_docs for d in batch.column(
        batch.schema.get_field_index("doc_id")).to_pylist()])
    tb = pa.Table.from_batches([batch]).filter(mask)
    if writer is None:
        writer = pq.ParquetWriter(str(TRAIN_BASE), tb.schema)
    writer.write_table(tb)
    n_train += tb.num_rows
writer.close()
assert n_train == len(docs) - len(eval_idx)
print(f"wrote {TRAIN_BASE} ({n_train} rows)", flush=True)

# sidecar for the train subset (stage3_build asserts stage=base, norm=none)
from nla.datagen._common import make_storage
from nla.datagen.sidecar import NLADatasetMeta, NLAExtractionMeta, write_sidecar
import argparse
storage = make_storage(argparse.Namespace(
    storage_cls="nla.datagen.storage.LocalStorage", storage_kwargs=None))
write_sidecar(storage, str(TRAIN_BASE), NLADatasetMeta(
    dataset_id="base_qwen2.5-7b-L20_matryoshka-sonnet46_av_train_overlaprl",
    stage="base", row_count=n_train,
    extraction=NLAExtractionMeta(
        base_model=BASE_MODEL, d_model=3584, layer_index=20, norm="none",
        corpus="openbmb/Ultra-FineWeb", corpus_slice={"start": 0, "length": 100000},
        positions_per_doc=10),
    created_by="kitft-overlap-penalty-rl/01_build_rl_parquet_ctx.py",
))

print("=== unigram table (train docs, deduped per doc by longest truncation) ===", flush=True)
best: dict = {}  # doc_id -> (n_raw_tokens, row-in-batch pointer resolved on 2nd pass)
for batch in pq.ParquetFile(str(TRAIN_BASE)).iter_batches(
        batch_size=4096, columns=["doc_id", "n_raw_tokens"]):
    for d, n in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
        if d not in best or n > best[d]:
            best[d] = n
counts: Counter = Counter()
total = 0
word_re = re.compile(r"\w+")
for batch in pq.ParquetFile(str(TRAIN_BASE)).iter_batches(
        batch_size=2048, columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated"]):
    for d, n, txt in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(),
                         batch.column(2).to_pylist()):
        if best.get(d) == n:
            best.pop(d)  # count each doc once
            ws = word_re.findall(txt.lower())
            counts.update(ws)
            total += len(ws)
vocab_full = len(counts)
pruned = {w: c for w, c in counts.items() if c >= 2}  # singletons -> smoothing ceiling
UNIGRAMS.write_text(json.dumps(
    {"__total__": total, "__vocab__": vocab_full, "counts": pruned}))
print(f"unigrams: {total} words, vocab {vocab_full} ({len(pruned)} kept >=2), "
      f"{UNIGRAMS.stat().st_size / 1e6:.1f} MB", flush=True)

print(f"=== stage3_build rl (tagged, +debug metadata): -> {RL_OUT} ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "nla.datagen.stage3_build",
     "--stage", "rl", "--input", str(TRAIN_BASE), "--output", str(RL_OUT),
     "--explanation-format", "tagged", "--keep-debug-metadata"],
    capture_output=True, text=True, cwd=str(REPO_ROOT),
)
print(r.stdout[-1500:])
if r.returncode != 0:
    print("STDERR:\n", r.stderr[-3000:])
    raise SystemExit("stage3_build rl failed")
cols = pq.ParquetFile(str(RL_OUT)).schema_arrow.names
assert "detokenized_text_truncated" in cols, f"context column missing: {cols}"
print(f"RL parquet columns: {cols}", flush=True)

if os.environ.get("SKIP_UPLOAD") != "1":
    print(f"=== upload to {DEST_REPO}/data ===", flush=True)
    api = HfApi(token=TOKEN)
    api.create_repo(DEST_REPO, repo_type="model", private=True, exist_ok=True)
    for p in (RL_OUT, Path(str(RL_OUT) + ".nla_meta.yaml"), UNIGRAMS):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"data/{p.name}",
                        repo_id=DEST_REPO)
        print("uploaded", p.name, flush=True)
print("BUILD_DONE", flush=True)
