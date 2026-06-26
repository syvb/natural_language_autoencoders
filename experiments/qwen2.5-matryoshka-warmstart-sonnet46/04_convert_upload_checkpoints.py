"""Stage D — convert + upload the warm-started AV (actor, DCP->HF) and AR (critic, HF) checkpoints.

The actor saves a DCP (distributed checkpoint, no safetensors); convert with
tools/convert_fsdp_to_hf.py. The critic already saves HF (iter_*/hf/ with value_head).

Run:  python 04_convert_upload_checkpoints.py [av|ar|both]
"""
import os, sys, glob, shutil, subprocess
from huggingface_hub import HfApi
tok = open("/root/.hf_token").read().strip(); api = HfApi(token=tok)
DS = "https://huggingface.co/datasets/syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46"
NLA_REPO = os.environ.get("NLA_REPO", "/workspace/nla")

def latest_iter(d):
    its = sorted(glob.glob(f"{d}/iter_*"))
    assert its, f"no iter_* in {d}"; return its[-1]

def card(role_long, role):
    return f"""---
license: apache-2.0
base_model: kitft/nla-qwen2.5-7b-L20-{role}
tags: [nla, natural-language-autoencoders, qwen2.5, {role}]
---
# nla-qwen2.5-7b-L20-{role} — re-warm-started on Sonnet-4.6 "matryoshka" data

{role_long} of the [NLA](https://github.com/kitft/natural_language_autoencoders) Qwen2.5-7B
(layer-20) pair, **re-warm-started** (SFT) from the released RLed checkpoint
[`kitft/nla-qwen2.5-7b-L20-{role}`](https://huggingface.co/kitft/nla-qwen2.5-7b-L20-{role})
on Claude Sonnet-4.6 explanations to change its writing style.

- Warm-start data: [matryoshka warm-start data]({DS}) ({role}-* split, document-level; ~5k held out)
- Hyperparameters (Qwen2.5 recipe): global batch 256, lr 2e-5->2e-6 cosine, warmup 50, 1 epoch{', injection_scale 150' if role=='av' else ''}
- Warm-start checkpoint intended as the starting point for a subsequent RL phase.
"""

def upload_dir(d, repo, role_long, role):
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    open(f"{d}/README.md", "w").write(card(role_long, role))
    print(f"uploading {d} -> {repo}", flush=True)
    api.upload_folder(folder_path=d, repo_id=repo, repo_type="model")
    print(f"DONE https://huggingface.co/{repo}", flush=True)

def do_av():
    it = latest_iter("/workspace/ckpt/av"); print("AV iter:", it, flush=True)
    out = "/workspace/hf_out/av"; shutil.rmtree(out, ignore_errors=True)
    r = subprocess.run([sys.executable, f"{NLA_REPO}/tools/convert_fsdp_to_hf.py",
        "--input-dir", it, "--output-dir", out, "--origin-hf-dir", "/workspace/models/rl-av", "-f"],
        capture_output=True, text=True)
    print(r.stdout[-2500:])
    if r.returncode != 0: print("CONVERT STDERR:\n", r.stderr[-3000:]); raise SystemExit("AV convert failed")
    # use the SFT sidecar written to iter_dir (convert's copy_assets pulled rl-av's stale one)
    if os.path.exists(f"{it}/nla_meta.yaml"): shutil.copy(f"{it}/nla_meta.yaml", f"{out}/nla_meta.yaml")
    upload_dir(out, "syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46", "Actor (AV)", "av")

def do_ar():
    it = latest_iter("/workspace/ckpt/ar"); hf = f"{it}/hf"; print("AR hf:", hf, flush=True)
    assert os.path.exists(f"{hf}/config.json"), hf
    upload_dir(hf, "syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46", "Critic (AR)", "ar")

which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("av", "both"): do_av()
if which in ("ar", "both"): do_ar()
print("UPLOAD_CKPT_ALL_DONE", flush=True)
