"""Leave-one-out (+ solo) line-ablation FVE for the matryoshka space precache.

Matryoshka explanations are independent bullet lines, so the critic can score
arbitrary line SUBSETS without going OOD — unlike prefix-FVE (which conflates
"unimportant" with "redundant with earlier lines"), leave-one-out measures each
line's causal NECESSITY. For every precached position with lines L1..Ln:

  loo[k]  = FVE of lines \\ {k}   (necessity: full − loo[k] = what only line k carries)
  solo[k] = FVE of {k} alone      (sufficiency: what line k reconstructs by itself)
  full    = FVE of all lines      (sanity: must match precache fve[-1])

Runs on ONE A100 80GB by loading the two 27B models sequentially: phase 1
extracts gold L42 activations with the raw base (one full-text forward per
text — causal ⇒ position i equals per-prefix extraction), frees it, then
phase 2 loads the critic and batch-scores every subset. Resumable: activations
and partial scores are checkpointed to --workdir.

    python loo_precompute.py --space space --out space/loo.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import critic_predict
from nla.utils.arch_adapters import resolve_decoder_layers

# KEEP IN SYNC with space/app.py
MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"
TOK_SUBDIR = "warmstart_av_lora"
CRITIC_SUBDIR = "rl_critic_step400"
LAYER = 42


class _StopForward(Exception):
    """Raised from the L42 hook to skip blocks L43..N."""


@torch.inference_mode()
def extract_all(entries, dev):
    """Gold L42 activations for every position of every text: one forward per
    text, hook captures the full [T, d] block output (causal ⇒ per-position
    extraction). Returns list of fp32 numpy [T, d]."""
    print("[phase1] loading raw base for extraction…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=dev)
    base.eval()
    layers = resolve_decoder_layers(base)
    grabbed = {}

    def grab(module, inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
        raise _StopForward

    h = layers[LAYER].register_forward_hook(grab)
    acts = []
    try:
        for e in entries:
            ids = torch.tensor([e["ids"]], device=dev)
            try:
                base(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
            acts.append(grabbed["h"][0].float().cpu().numpy())
            print(f"  extracted [{len(acts)}/{len(entries)}] T={len(e['ids'])}", flush=True)
    finally:
        h.remove()
    del base
    torch.cuda.empty_cache()
    return acts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--space", default="space", help="dir with precache.json, mu.npy, nla_meta.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default="loo_work", help="checkpoint dir (resume)")
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    space = Path(args.space)
    work = Path(args.workdir); work.mkdir(exist_ok=True)
    dev = "cuda"

    entries = json.load(open(space / "precache.json"))["entries"]
    root = snapshot_download(MODEL_REPO, allow_patterns=[f"{TOK_SUBDIR}/*", f"{CRITIC_SUBDIR}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{TOK_SUBDIR}")
    cfg = load_nla_config(str(space), tok)
    MSE_SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(space / "mu.npy"), dtype=torch.float32)

    # ── phase 1: gold activations (checkpointed) ─────────────────────────────
    acts_path = work / "acts.npz"
    if acts_path.exists():
        z = np.load(acts_path)
        acts = [z[f"a{i}"] for i in range(len(entries))]
        print(f"[phase1] resumed {len(acts)} activation mats from {acts_path}", flush=True)
    else:
        acts = extract_all(entries, dev)
        np.savez_compressed(acts_path, **{f"a{i}": a for i, a in enumerate(acts)})
        print(f"[phase1] saved → {acts_path}", flush=True)

    # ── build the job list: (entry, pos, kind, k, explanation_text) ─────────
    jobs = []
    for ei, e in enumerate(entries):
        for pos, r in enumerate(e["results"]):
            if not r or not r.get("lines"):
                continue
            lines = r["lines"]
            jobs.append((ei, pos, "full", -1, "\n".join(lines)))
            if len(lines) > 1:
                for k in range(len(lines)):
                    jobs.append((ei, pos, "loo", k, "\n".join(lines[:k] + lines[k + 1:])))
            for k in range(len(lines)):
                jobs.append((ei, pos, "solo", k, lines[k]))
    print(f"[jobs] {len(jobs)} critic scorings "
          f"({sum(1 for j in jobs if j[2] == 'loo')} loo, "
          f"{sum(1 for j in jobs if j[2] == 'solo')} solo)", flush=True)

    # ── phase 2: critic scores every subset (checkpointed every 200 batches) ─
    print("[phase2] loading critic…", flush=True)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{CRITIC_SUBDIR}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    critic.to(dev).eval()

    scores_path = work / "scores.npy"
    scores = np.load(scores_path) if scores_path.exists() else np.full(len(jobs), np.nan)
    start = int(np.argmax(np.isnan(scores))) if np.isnan(scores).any() else len(jobs)
    if start:
        print(f"[phase2] resuming at job {start}/{len(jobs)}", flush=True)

    # per-position gold normalization (cached per (ei,pos))
    gold_cache = {}

    def gold(ei, pos):
        if (ei, pos) not in gold_cache:
            v = torch.tensor(acts[ei][pos], dtype=torch.float32)
            gn = v / v.norm().clamp_min(1e-12) * MSE_SCALE
            gold_cache[(ei, pos)] = (gn, ((gn - MU) ** 2).mean().item())
        return gold_cache[(ei, pos)]

    pad = tok.eos_token_id
    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(start, len(jobs), args.batch):
            chunk = jobs[b0: b0 + args.batch]
            idlists = [tok.encode(TPL.format(explanation=j[4]),
                                  add_special_tokens=False)[:1024] for j in chunk]
            m = max(len(x) for x in idlists)
            bx = torch.full((len(chunk), m), pad, dtype=torch.long, device=dev)
            attn = torch.zeros((len(chunk), m), dtype=torch.long, device=dev)
            for r, q in enumerate(idlists):
                bx[r, : len(q)] = torch.tensor(q, dtype=torch.long)
                attn[r, : len(q)] = 1
            preds = critic_predict(critic, bx, attn, MSE_SCALE).float().cpu()
            for ci, (j, pred) in enumerate(zip(chunk, preds)):
                gn, denom = gold(j[0], j[1])
                pn = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
                scores[b0 + ci] = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            nb = (b0 - start) // args.batch + 1
            if nb % 200 == 0:
                np.save(scores_path, scores)
                done = b0 + len(chunk) - start
                rate = done / (time.time() - t0)
                eta = (len(jobs) - b0 - len(chunk)) / max(rate, 1e-9) / 60
                print(f"  [{b0 + len(chunk)}/{len(jobs)}] {rate:.0f} jobs/s eta {eta:.0f}min", flush=True)
    np.save(scores_path, scores)

    # ── assemble loo.json aligned with precache entries/positions ────────────
    out = {"entries": []}
    by_pos = {}
    for j, s in zip(jobs, scores):
        by_pos.setdefault((j[0], j[1]), {}).setdefault(j[2], {})[j[3]] = float(s)
    n_mismatch = 0
    for ei, e in enumerate(entries):
        loo_arr, solo_arr, full_arr = [], [], []
        for pos, r in enumerate(e["results"]):
            d = by_pos.get((ei, pos))
            if not d:
                loo_arr.append(None); solo_arr.append(None); full_arr.append(None)
                continue
            n = len(r["lines"])
            full = d["full"][-1]
            full_arr.append(round(full, 5))
            loo_arr.append([round(d["loo"][k], 5) for k in range(n)] if "loo" in d else None)
            solo_arr.append([round(d["solo"][k], 5) for k in range(n)])
            # sanity: recomputed full-lines FVE vs the precached fve[-1].
            # T=1 critic? no — the critic is deterministic; differences mean an
            # extraction/pipeline mismatch. Allow small numeric drift (bf16).
            if abs(full - r["fve"][-1]) > 0.02:
                n_mismatch += 1
        out["entries"].append({"text": e["text"], "full": full_arr,
                               "loo": loo_arr, "solo": solo_arr})
    print(f"[sanity] full-FVE mismatch (> 0.02 vs precache) at {n_mismatch} positions", flush=True)

    outp = args.out or str(space / "loo.json")
    json.dump(out, open(outp, "w"))
    # quick corpus stats
    fl, lo = [], []
    for ee in out["entries"]:
        for f, l in zip(ee["full"], ee["loo"]):
            if f is not None and l is not None:
                fl += [f] * len(l); lo += l
    fl, lo = np.array(fl), np.array(lo)
    d = fl - lo
    print(f"[stats] {len(d)} (position,line) LOO deltas: mean={d.mean():.4f} "
          f"p50={np.percentile(d, 50):.4f} p95={np.percentile(d, 95):.4f} "
          f"max={d.max():.4f} min={d.min():.4f}")
    print(f"[saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
