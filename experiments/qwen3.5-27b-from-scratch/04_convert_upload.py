"""Step 5 — convert the warm-start checkpoints to usable HF form (+ upload).

  AV: $WORK/ckpt/av_ws/iter_*  (FSDP DCP)  ->  $WORK/hf_out/av_ws  (HF)
  AR: $WORK/ckpt/ar_ws/iter_*/hf           ->  verified in place
  parquets: $WORK/out/{av,ar}_{sft,eval}.parquet + sidecars

Verifies before anything ships: AR value_head finite (the half-shard
corruption lesson), sidecars carry the bullets template (not a stale tagged
one). UPLOAD=0 converts/verifies only — use that for smokes; the RL launcher
takes the LOCAL hf_out/av_ws + iter_*/hf dirs directly.

Repos (UPLOAD=1): {HF_REPO_PREFIX}-{av,ar}-warmstart (models),
{HF_REPO_PREFIX}-data (dataset). Run: python 04_convert_upload.py [av|ar|data|all]
"""
import glob
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config

import torch
from safetensors.torch import load_file

_config.load()
WORK = _config.env("WORK")
OUT = f"{WORK}/out"
PREFIX = _config.env("HF_REPO_PREFIX")
UPLOAD = os.environ.get("UPLOAD", "1") == "1"
NLA_REPO = os.environ.get("NLA_REPO", f"{WORK}/nla")
BASE_DIR = os.environ.get("BASE_MODEL_DIR", f"{WORK}/models/base")

api = None
if UPLOAD:
    from huggingface_hub import HfApi

    tok_file = _config.env("HF_TOKEN_FILE")
    api = HfApi(token=os.environ.get("HF_TOKEN") or open(tok_file).read().strip())


def latest_iter(d):
    its = sorted(glob.glob(f"{d}/iter_*"))
    assert its, f"no iter_* in {d}"
    return its[-1]


def assert_bullets_sidecar(path):
    assert os.path.exists(path), f"no sidecar at {path}"
    assert "<explanation>" not in open(path).read(), (
        f"{path} carries a TAGGED template — the --nla-sidecar-source fix "
        f"regressed; refusing to ship a wrong-prompt checkpoint."
    )


def upload_dir(d, repo, note):
    if not UPLOAD:
        print(f"[UPLOAD=0] verified {d} (would upload to {repo})", flush=True)
        return
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    open(f"{d}/README.md", "w").write(
        f"---\nlicense: apache-2.0\nbase_model: {_config.env('BASE_MODEL')}\n"
        f"tags: [nla, natural-language-autoencoders]\n---\n"
        f"# {repo.split('/')[-1]}\n\n{note}\n\n"
        f"From-scratch NLA warm-start of {_config.env('BASE_MODEL')} at layer "
        f"{_config.env('LAYER_INDEX')} on Claude Sonnet-4.6 explanations "
        f"(bullets format; see the repo's experiments/qwen3.5-27b-from-scratch/RUNBOOK.md).\n"
    )
    print(f"uploading {d} -> {repo}", flush=True)
    api.upload_folder(folder_path=d, repo_id=repo, repo_type="model")
    print(f"DONE https://huggingface.co/{repo}", flush=True)


def do_av():
    it = latest_iter(f"{WORK}/ckpt/av_ws")
    print("AV iter:", it, flush=True)
    out = f"{WORK}/hf_out/av_ws"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run(
        [sys.executable, f"{NLA_REPO}/tools/convert_fsdp_to_hf.py",
         "--input-dir", it, "--output-dir", out, "--origin-hf-dir", BASE_DIR, "-f"],
        capture_output=True, text=True,
    )
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("CONVERT STDERR:\n", r.stderr[-3000:])
        raise SystemExit("AV convert failed")
    sidecar = f"{it}/nla_meta.yaml"
    assert_bullets_sidecar(sidecar)
    shutil.copy(sidecar, f"{out}/nla_meta.yaml")
    print(f"AV_HF_READY {out}", flush=True)
    upload_dir(out, f"{PREFIX}-av-warmstart", "Actor (AV) warm-start.")


def do_ar():
    it = latest_iter(f"{WORK}/ckpt/ar_ws")
    hf = f"{it}/hf"
    print("AR hf:", hf, flush=True)
    assert os.path.exists(f"{hf}/config.json"), hf
    vh = f"{hf}/value_head.safetensors"
    assert os.path.exists(vh), f"missing {vh}"
    for k, v in load_file(vh).items():
        assert torch.isfinite(v).all(), f"non-finite value_head tensor {k} — do not upload"
    print("value_head finite ✓", flush=True)
    assert_bullets_sidecar(f"{hf}/nla_meta.yaml")
    print(f"AR_HF_READY {hf}", flush=True)
    upload_dir(hf, f"{PREFIX}-ar-warmstart", "Critic (AR) warm-start.")


def do_data():
    if not UPLOAD:
        print("[UPLOAD=0] skipping dataset upload", flush=True)
        return
    repo = f"{PREFIX}-data"
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    for stem in ("av_sft", "av_eval", "ar_sft", "ar_eval"):
        for suf in (".parquet", ".parquet.nla_meta.yaml"):
            p = f"{OUT}/{stem}{suf}"
            assert os.path.exists(p), p
            print(f"uploading {p} -> {repo}/", flush=True)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{stem}{suf}",
                            repo_id=repo, repo_type="dataset")
    print(f"DONE https://huggingface.co/datasets/{repo}", flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("av", "all"):
    do_av()
if which in ("ar", "all"):
    do_ar()
if which in ("data", "all"):
    do_data()
print("CONVERT_UPLOAD_DONE", flush=True)
