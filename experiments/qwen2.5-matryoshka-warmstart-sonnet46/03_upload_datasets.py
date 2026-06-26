"""Stage C — upload the joined warm-start data (vectors + Sonnet-4.6 text) to a public HF dataset."""
import os
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
REPO = os.environ.get("DATA_REPO", "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46")
api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)

def nrows(p):
    import pyarrow.parquet as pq; return pq.ParquetFile(p).metadata.num_rows
counts = {f: nrows(f"/workspace/out/{f}.parquet") for f in
          ["av_sft", "av_eval", "ar_sft", "ar_eval", "base_av", "base_ar"]}

README = f"""---
license: apache-2.0
language: [en]
tags: [nla, natural-language-autoencoders, qwen2.5, activation-vectors, warm-start]
---

# NLA Qwen2.5-7B L20 warm-start data — Sonnet-4.6 "matryoshka" explanations + activations

Pairs **Qwen2.5-7B-Instruct layer-20 residual-stream activations** with the **Claude
Sonnet-4.6 explanations** from
[`ceselder/nla-matryoshka-warmstart-sonnet46`](https://huggingface.co/datasets/ceselder/nla-matryoshka-warmstart-sonnet46).
The source is text-only; this adds the activations (layer-20 hidden state at the last token
of each `input_text`, which is exactly `token_ids[:position]`). Reproducible from text alone.

## Files
| file | rows | use |
|---|---|---|
| `av_sft.parquet` | {counts['av_sft']} | actor (AV) warm-start SFT — train |
| `av_eval.parquet` | {counts['av_eval']} | actor eval holdout (document-level) |
| `ar_sft.parquet` | {counts['ar_sft']} | critic (AR) warm-start SL — train |
| `ar_eval.parquet` | {counts['ar_eval']} | critic eval holdout (document-level) |
| `base_av.parquet` | {counts['base_av']} | master: vectors+explanation+text+custom_id (all av) |
| `base_ar.parquet` | {counts['base_ar']} | master: vectors+explanation+text+custom_id (all ar) |

SFT/eval parquets are stage-3 format (drop-in for `configs/actor_sft.sh` / `critic_sft.sh`).
`base_*` masters carry the raw activation vectors (`norm="none"`) + `api_explanation` +
`detokenized_text_truncated` + `custom_id`; run `nla.datagen.stage3_build` to regenerate.

## Holdout
~10k samples held out for eval (≈5k av + ≈5k ar), document-level (whole docs, seed 42).

## Provenance
- base model `Qwen/Qwen2.5-7B-Instruct`, layer 20, d_model 3584, raw vectors (`norm="none"`)
- explanations `claude-sonnet-4-6` (av-*/ar-* succeeded rows of the matryoshka dataset)
"""
open("/workspace/out/README.md", "w").write(README)

files = ["av_sft", "av_eval", "ar_sft", "ar_eval", "base_av", "base_ar"]
todo = []
for f in files:
    todo.append(f"{f}.parquet"); todo.append(f"{f}.parquet.nla_meta.yaml")
for f in todo:
    p = f"/workspace/out/{f}"
    assert os.path.exists(p), p
    print("uploading", f, f"{os.path.getsize(p)/1e6:.0f}MB", flush=True)
    api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=REPO, repo_type="dataset")
api.upload_file(path_or_fileobj="/workspace/out/README.md", path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
print("DATASET_UPLOAD_DONE https://huggingface.co/datasets/" + REPO, flush=True)
