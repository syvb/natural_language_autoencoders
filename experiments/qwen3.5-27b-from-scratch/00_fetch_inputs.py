"""Step 0 — download the base model and the matryoshka source dataset.

Downloads:
  $WORK/models/base          the base HF model (BASE_MODEL)
  $WORK/data/matryoshka/     the text-only Sonnet-4.6 explanation dataset
                             (input_text + analysis keyed by custom_id;
                             tokenizer-independent — see 01)

Idempotent (snapshot_download resumes). Needs HF_TOKEN or HF_TOKEN_FILE.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config

_config.load()
WORK = _config.env("WORK")
tok_file = os.environ.get("HF_TOKEN_FILE", "")
token = os.environ.get("HF_TOKEN") or (
    open(tok_file).read().strip() if tok_file and os.path.exists(tok_file) else None
)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download  # noqa: E402

base = snapshot_download(
    _config.env("BASE_MODEL"), local_dir=f"{WORK}/models/base",
    token=token, max_workers=16,
)
print("base model ->", base, flush=True)

mat = snapshot_download(
    _config.env("MATRYOSHKA_DATASET"), repo_type="dataset",
    local_dir=f"{WORK}/data/matryoshka", token=token, max_workers=16,
)
print("matryoshka dataset ->", mat, flush=True)
print("FETCH_DONE", flush=True)
