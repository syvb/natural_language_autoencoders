"""Stage D (v3) — convert + upload the v3 warm-started AV (DCP->HF) and AR (HF)
checkpoints, plus the v3 dataset parquets. Everything PUBLIC.

Same shape as 04_convert_upload_checkpoints.py but v3 paths/repos:
  /workspace/ckpt/av_v3/iter_*      -> syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3
  /workspace/ckpt/ar_v3/iter_*/hf   -> syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3
  /workspace/out/{av,ar}_{sft,eval}_v3.parquet (+ sidecars)
      -> datasets/syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46 under v3/

The AV sidecar copied from the iter dir carries the BULLETS template (the SFT
ran with --nla-sidecar-source = the v3 parquet); a tagged template here means
the sidecar-inheritance bug regressed — assert loudly. The AR value_head is
checked finite before upload (the v1 half-shard corruption lesson).

Run:  python 04c_convert_upload_v3.py [av|ar|data|all]
"""
import glob
import os
import shutil
import subprocess
import sys

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file

tok = open("/root/.hf_token").read().strip()
api = HfApi(token=tok)
NLA_REPO = os.environ.get("NLA_REPO", "/workspace/nla")
DS_REPO = "syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
OUT = os.environ.get("OUT_DIR", "/workspace/out")


def latest_iter(d):
    its = sorted(glob.glob(f"{d}/iter_*"))
    assert its, f"no iter_* in {d}"
    return its[-1]


def card(role_long, role):
    return f"""---
license: apache-2.0
base_model: kitft/nla-qwen2.5-7b-L20-{role}
tags: [nla, natural-language-autoencoders, qwen2.5, {role}]
---
# nla-qwen2.5-7b-L20-{role}-matryoshka-sonnet46-v3 — v3 warm-start (bullets prompt)

{role_long} of the [NLA](https://github.com/kitft/natural_language_autoencoders) Qwen2.5-7B
(layer-20) pair, **v3 warm-start**: SFT from the base
[`kitft/nla-qwen2.5-7b-L20-{role}`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-{role})
on Claude Sonnet-4.6 explanations in the v3 format.

v3 changes vs v2 (see the repo's experiment README):
- The prompt actually says **bullet points** ("a list of bullet points, one per line");
  output text is plain one-item-per-line (no literal "- " markers). The `nla_meta.yaml`
  sidecar carries the real trained template (v1/v2 shipped a stale tagged one).
- {"AV: trained with NLA_NO_TRAIN_EOS=1 (never taught to stop; RL caps generation)." if role == "av"
   else "AR: critic inputs token-truncated to K ~ U[1,120] (2% left full) — pre-calibrated for the RL-time uniform token truncation."}
- Intended as the starting point for the v3 RL phase (KL 0.03, uniform token
  truncation ~U[1,120]).

Warm-start data: [v3 splits](https://huggingface.co/datasets/{DS_REPO}) (`v3/` folder).
Hyperparameters: global batch 256, lr 2e-5→2e-6 cosine, warmup 50, 1 epoch{", injection_scale 150" if role == "av" else ""}.
"""


def upload_dir(d, repo, role_long, role):
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    try:
        api.update_repo_settings(repo, private=False)
    except Exception:
        pass
    open(f"{d}/README.md", "w").write(card(role_long, role))
    print(f"uploading {d} -> {repo}", flush=True)
    api.upload_folder(folder_path=d, repo_id=repo, repo_type="model")
    print(f"DONE https://huggingface.co/{repo}", flush=True)


def do_av():
    it = latest_iter("/workspace/ckpt/av_v3")
    print("AV iter:", it, flush=True)
    out = "/workspace/hf_out/av_v3"
    shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, f"{NLA_REPO}/tools/convert_fsdp_to_hf.py",
                        "--input-dir", it, "--output-dir", out,
                        "--origin-hf-dir", "/workspace/models/rl-av", "-f"],
                       capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0:
        print("CONVERT STDERR:\n", r.stderr[-3000:])
        raise SystemExit("AV convert failed")
    # the SFT sidecar written to iter_dir is the bullets one (convert's
    # copy_assets pulled rl-av's stale tagged sidecar — overwrite it)
    sidecar = f"{it}/nla_meta.yaml"
    assert os.path.exists(sidecar), f"no sidecar in {it}"
    assert "<explanation>" not in open(sidecar).read(), (
        f"{sidecar} carries the TAGGED template — the --nla-sidecar-source fix "
        f"regressed; refusing to ship a wrong-prompt checkpoint again."
    )
    shutil.copy(sidecar, f"{out}/nla_meta.yaml")
    upload_dir(out, "syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3", "Actor (AV)", "av")


def do_ar():
    it = latest_iter("/workspace/ckpt/ar_v3")
    hf = f"{it}/hf"
    print("AR hf:", hf, flush=True)
    assert os.path.exists(f"{hf}/config.json"), hf
    vh = f"{hf}/value_head.safetensors"
    assert os.path.exists(vh), f"missing {vh}"
    for k, v in load_file(vh).items():
        assert torch.isfinite(v).all(), f"non-finite value_head tensor {k} — do not upload"
    print("value_head finite ✓", flush=True)
    sidecar = f"{hf}/nla_meta.yaml"
    assert os.path.exists(sidecar) and "<explanation>" not in open(sidecar).read(), (
        f"AR sidecar missing or carries the tagged template: {sidecar}"
    )
    upload_dir(hf, "syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46-v3", "Critic (AR)", "ar")


def do_data():
    api.create_repo(DS_REPO, repo_type="dataset", private=False, exist_ok=True)
    try:
        api.update_repo_settings(DS_REPO, private=False, repo_type="dataset")
    except Exception:
        pass
    for stem in ("av_sft_v3", "av_eval_v3", "ar_sft_v3", "ar_eval_v3"):
        for suf in (".parquet", ".parquet.nla_meta.yaml"):
            p = f"{OUT}/{stem}{suf}"
            assert os.path.exists(p), p
            print(f"uploading {p} -> {DS_REPO}/v3/", flush=True)
            api.upload_file(path_or_fileobj=p, path_in_repo=f"v3/{stem}{suf}",
                            repo_id=DS_REPO, repo_type="dataset")
    print(f"DONE https://huggingface.co/datasets/{DS_REPO}", flush=True)


which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("av", "all"):
    do_av()
if which in ("ar", "all"):
    do_ar()
if which in ("data", "all"):
    do_data()
print("UPLOAD_V3_ALL_DONE", flush=True)
