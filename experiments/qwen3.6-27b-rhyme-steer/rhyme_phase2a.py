"""Phase 2a: encode original + rhyme-edited explanation texts through the arm's
OWN critic -> steering vectors (run ON the GPU box, after edits.json is written).

edits.json format (authored locally after reviewing phase-1 explanations):
    {"mat": {"texts": {"orig": "<full unedited explanation>",
                       "edit_moon": "<same text, rhyme bits swapped>", ...}},
     "std": {"texts": {...}}}

For each named text: vec = critic_predict(critic, tokenized critic template).
Saves work/critic_vecs_{arm}.npz (name -> [5120] fp32) and prints cos + FVE
vs the true v_head (same convention as score_subsets.py: pred & gold normalized
to mse_scale, denominator MSE(mu, gold)).

Usage:  python rhyme_phase2a.py --model mat
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import critic_predict

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK", "/workspace/rs/work"))

ARMS = {
    "mat": {"repo": "ceselder/nla-qwen36-27b-matryoshka",
            "tok_sub": "warmstart_av_lora", "critic_sub": "rl_critic_step400",
            "sidecar": str(HERE / "sidecar_mat")},
    "std": {"repo": "ceselder/qwen3.6-27b-nla-L42",
            "tok_sub": "av_sft_lora", "critic_sub": "rl_critic_step400",
            "sidecar": str(HERE / "sidecar_std")},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mat", "std"], required=True)
    ap.add_argument("--edits", default=str(HERE / "edits.json"))
    ap.add_argument("--mu", default=str(HERE / "data" / "mu.npy"))
    args = ap.parse_args()
    arm = ARMS[args.model]

    texts = json.load(open(args.edits))[args.model]["texts"]
    root = snapshot_download(arm["repo"], allow_patterns=[f"{arm['tok_sub']}/*",
                                                          f"{arm['critic_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(arm["sidecar"], tok)
    MSE_SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template

    print(f"[load] critic {arm['repo']}/{arm['critic_sub']}…", flush=True)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{arm['critic_sub']}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    critic.to("cuda").eval()

    v_head = np.load(WORK / "v_head.npy")
    gn = torch.tensor(v_head, dtype=torch.float32)
    gn = gn / gn.norm().clamp_min(1e-12) * MSE_SCALE
    MU = torch.tensor(np.load(args.mu), dtype=torch.float32)
    denom = ((gn - MU) ** 2).mean().item()

    vecs = {}
    pad = tok.eos_token_id
    with torch.inference_mode():
        for name, text in texts.items():
            ids = tok.encode(TPL.format(explanation=text), add_special_tokens=False)
            assert len(ids) <= 1024, f"{name}: critic input {len(ids)} tokens > 1024"
            bx = torch.tensor([ids], dtype=torch.long, device="cuda")
            attn = torch.ones_like(bx)
            pred = critic_predict(critic, bx, attn, MSE_SCALE)[0].float().cpu()
            vecs[name] = pred.numpy()
            pn = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
            fve = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            cos = float(torch.dot(pred, torch.tensor(v_head)) /
                        (pred.norm() * np.linalg.norm(v_head)))
            print(f"  {name:24s} ||pred||={pred.norm():.1f} cos(v_head)={cos:+.3f} "
                  f"FVE={fve:+.3f} ({len(ids)} tok)", flush=True)

    np.savez(WORK / f"critic_vecs_{args.model}.npz", **vecs)
    print(f"[write] work/critic_vecs_{args.model}.npz ({len(vecs)} vecs)", flush=True)
    print(f"PHASE2A_DONE_{args.model.upper()}", flush=True)


if __name__ == "__main__":
    main()
