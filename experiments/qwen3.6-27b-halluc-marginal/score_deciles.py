"""EXACT token-decile marginal FVE (GPU-side). Like score_subsets.py but the
critic is scored on cumulative TOKEN-DECILE prefixes of each explanation (real
tokenizer) instead of line subsets:

  pfx[d]      = FVE( first d+1 token deciles )
  marginal[d] = pfx[d] - pfx[d-1]   (pfx[0] itself for d=0)

Boundaries use the arm's own tokenizer on "\\n".join(lines) — identical to the
chunk_judge.py decile boundaries, so decile d's marginal aligns with decile d's
hallucination verdict.

    python score_deciles.py --arm mat --out results/decile_scores_mat.json
"""
import argparse
import json
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

HERE = Path(__file__).resolve().parent
BASE_ID = "Qwen/Qwen3.6-27B"
LAYER = 42
D = 5120
NCHUNK = 10

ARMS = {
    "mat": {"repo": "ceselder/nla-qwen36-27b-matryoshka", "tok_sub": "warmstart_av_lora",
            "critic_sub": "rl_critic_step400", "sidecar": "sidecar_mat", "expl": "explanations_mat.json"},
    "std": {"repo": "ceselder/qwen3.6-27b-nla-L42", "tok_sub": "av_sft_lora",
            "critic_sub": "rl_critic_step400", "sidecar": "sidecar_std", "expl": "explanations_std.json"},
}


class _StopForward(Exception):
    pass


@torch.inference_mode()
def extract_acts(contexts, dev):
    print("[phase1] loading raw base for extraction…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0}).eval()
    layers = resolve_decoder_layers(base)
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        raise _StopForward

    h = layers[LAYER].register_forward_hook(grab)
    vecs = np.empty((len(contexts), D), np.float32)
    t0 = time.time()
    try:
        for k, c in enumerate(contexts):
            ids = torch.tensor([c["prefix_ids"]], device=dev)
            try:
                base(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
            vecs[k] = grabbed["h"][0, -1].float().cpu().numpy()
            if (k + 1) % 50 == 0:
                print(f"  extracted [{k+1}/{len(contexts)}] ({time.time()-t0:.0f}s)", flush=True)
    finally:
        h.remove()
    del base
    torch.cuda.empty_cache()
    return vecs


def decile_prefixes(lines, tok):
    """10 cumulative token-decile prefix texts of '\\n'.join(lines); None if <10 tokens."""
    ids = tok.encode("\n".join(lines), add_special_tokens=False)
    n = len(ids)
    if n < NCHUNK:
        return None
    bnd = [round(d * n / NCHUNK) for d in range(NCHUNK + 1)]
    return [tok.decode(ids[:bnd[d + 1]]) for d in range(NCHUNK)]   # cumulative: first d+1 deciles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["mat", "std"], required=True)
    ap.add_argument("--suffix-eval", default=str(HERE.parent / "qwen3.6-27b-suffix-eval"))
    ap.add_argument("--mu", default=str(HERE / "data" / "mu.npy"))
    ap.add_argument("--workdir", default=str(HERE / "decile_work"))
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    arm = ARMS[args.arm]
    se = Path(args.suffix_eval)
    work = Path(args.workdir); work.mkdir(exist_ok=True)
    dev = "cuda"

    contexts = json.load(open(se / "data" / "manifest.json"))["contexts"]
    expl = json.load(open(se / "results" / arm["expl"]))["entries"]
    assert len(expl) == len(contexts) and all(expl[i]["ci"] == contexts[i]["ci"] for i in range(len(expl)))
    if args.limit:
        contexts, expl = contexts[:args.limit], expl[:args.limit]

    root = snapshot_download(arm["repo"], allow_patterns=[f"{arm['tok_sub']}/*", f"{arm['critic_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(str(se / arm["sidecar"]), tok)
    MSE_SCALE = float(cfg.mse_scale); TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(args.mu), dtype=torch.float32)
    print(f"[cfg] arm={args.arm} mse_scale={MSE_SCALE:.2f}", flush=True)

    # phase 1 (cached, shared across arms)
    acts_path = work / "acts.npy"
    if acts_path.exists():
        vecs = np.load(acts_path)[:len(contexts)]
        print(f"[phase1] reusing {vecs.shape}", flush=True)
    else:
        vecs = extract_acts(contexts, dev)
        np.save(acts_path, vecs)

    # build cumulative-decile-prefix jobs
    jobs = []   # (ei, ri, d, prefix_text)
    nskip = 0
    for ei, e in enumerate(expl):
        for ri, lines in enumerate(e["explanations"]):
            if not lines:
                continue
            pfxs = decile_prefixes(lines, tok)
            if pfxs is None:
                nskip += 1
                continue
            for d in range(NCHUNK):
                jobs.append((ei, ri, d, pfxs[d]))
    print(f"[jobs] {len(jobs)} decile-prefix scorings ({nskip} rollouts <10 tok)", flush=True)

    print("[phase2] loading critic…", flush=True)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{arm['critic_sub']}", torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()

    scores_path = work / f"scores_{args.arm}.npy"
    scores = np.load(scores_path) if scores_path.exists() else np.full(len(jobs), np.nan)
    assert len(scores) == len(jobs), "resume mismatch — clear decile_work"
    start = int(np.argmax(np.isnan(scores))) if np.isnan(scores).any() else len(jobs)
    if start:
        print(f"[phase2] resuming at {start}/{len(jobs)}", flush=True)

    gold_cache = {}

    def gold(ei):
        if ei not in gold_cache:
            v = torch.tensor(vecs[ei], dtype=torch.float32)
            gn = v / v.norm().clamp_min(1e-12) * MSE_SCALE
            gold_cache[ei] = (gn, ((gn - MU) ** 2).mean().item())
        return gold_cache[ei]

    pad = tok.eos_token_id
    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(start, len(jobs), args.batch):
            chunk = jobs[b0:b0 + args.batch]
            idlists = [tok.encode(TPL.format(explanation=j[3]), add_special_tokens=False)[:1024] for j in chunk]
            m = max(len(x) for x in idlists)
            bx = torch.full((len(chunk), m), pad, dtype=torch.long, device=dev)
            attn = torch.zeros((len(chunk), m), dtype=torch.long, device=dev)
            for r, q in enumerate(idlists):
                bx[r, :len(q)] = torch.tensor(q, dtype=torch.long); attn[r, :len(q)] = 1
            preds = critic_predict(critic, bx, attn, MSE_SCALE).float().cpu()
            for ci, (j, pred) in enumerate(zip(chunk, preds)):
                gn, denom = gold(j[0])
                pn = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
                scores[b0 + ci] = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            if ((b0 - start) // args.batch + 1) % 100 == 0:
                np.save(scores_path, scores)
                rate = (b0 + len(chunk) - start) / (time.time() - t0)
                print(f"  [{b0+len(chunk)}/{len(jobs)}] {rate:.0f}/s eta {(len(jobs)-b0-len(chunk))/max(rate,1e-9)/60:.0f}min", flush=True)
    np.save(scores_path, scores)
    assert np.isfinite(scores).all(), "non-finite scores"

    # assemble: pfx per (ei,ri) over 10 deciles
    by = {}
    for (ei, ri, d, _), s in zip(jobs, scores):
        by.setdefault((ei, ri), {})[d] = float(s)
    out = {"meta": {"arm": args.arm, "mse_scale": MSE_SCALE, "gpu": torch.cuda.get_device_name(0)}, "entries": []}
    for ei, e in enumerate(expl):
        rolls = []
        for ri in range(len(e["explanations"])):
            d = by.get((ei, ri))
            rolls.append([round(d[k], 5) for k in range(NCHUNK)] if d else None)
        out["entries"].append({"ci": e["ci"], "pfx_decile": rolls})
    json.dump(out, open(args.out, "w"))
    print(f"[saved] {args.out}", flush=True)
    print(f"SCORE_DECILES_DONE_{args.arm.upper()}", flush=True)


if __name__ == "__main__":
    main()
