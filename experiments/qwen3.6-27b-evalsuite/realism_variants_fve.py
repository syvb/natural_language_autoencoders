"""Per-line MARGINAL FVE for the matryoshka RL verbalizations of the 100 variations.

marginal[k] = FVE(nested prefix lines 1..k) - FVE(lines 1..k-1)  (FVE(0 lines)=0)
i.e. how much salience-ordered line k adds to the critic's reconstruction. NOT LOO.

Phase 1: re-extract the gold layer-42 activation for each (variation, position),
reproducing exactly the system-block rendering + saved positions from the
verbalization run. Phase 2: load the matryoshka critic and score every nested
prefix, then difference to get marginal FVE.

  python3 realism_variants_fve.py            # -> realism_variants_fve.json

Adapted from loo_precompute.py. One A100 80GB (base then critic, sequential).
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

MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"
TOK_SUBDIR = "warmstart_av_lora"
CRITIC_SUBDIR = "rl_critic_step400"
LAYER = 42
HERE = Path(__file__).resolve().parent
HEAD, TAIL = "<|im_start|>system\n", "<|im_end|>\n"


class _Stop(Exception):
    pass


def extract_gold(variants, text_by_id, render_tok, dev):
    """Phase 1 in its own scope so base+layers are freed before the critic loads."""
    print("[phase1] loading raw base…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=dev)
    base.eval()
    layers = resolve_decoder_layers(base)
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()
        raise _Stop

    h = layers[LAYER].register_forward_hook(grab)
    gold = {}
    try:
        for vi, v in enumerate(variants):
            text = HEAD + text_by_id[v["id"]].strip() + TAIL
            ids = render_tok.encode(text, add_special_tokens=False)
            try:
                base(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            except _Stop:
                pass
            full = grabbed["h"][0].float().cpu().numpy()
            for pi, p in enumerate(v["positions"]):
                gold[(vi, pi)] = full[p]
            if vi % 10 == 0:
                print(f"  extracted {vi+1}/{len(variants)}", flush=True)
    finally:
        h.remove()
    del base, layers, grabbed
    torch.cuda.empty_cache()
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="space")
    ap.add_argument("--verb", default="realism_variants_verb_mat_rl.json")
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()
    space = Path(args.space)
    dev = "cuda"

    variants = json.load(open(HERE / args.verb))["variants"]
    text_by_id = {v["id"]: v["text"] for v in json.load(open(HERE / "realism_variants_scored.json"))}
    root = snapshot_download(MODEL_REPO, allow_patterns=[f"{TOK_SUBDIR}/*", f"{CRITIC_SUBDIR}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{TOK_SUBDIR}")
    cfg = load_nla_config(str(space), tok)
    MSE_SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(space / "mu.npy"), dtype=torch.float32)
    render_tok = AutoTokenizer.from_pretrained(BASE_ID)  # match the verbalization rendering

    # ── phase 1: gold activations at the exact saved positions ──────────────
    gold = extract_gold(variants, text_by_id, render_tok, dev)
    print(f"[phase1] {len(gold)} gold activations", flush=True)

    # ── build nested-prefix jobs ────────────────────────────────────────────
    jobs = []   # (vi, pi, k, prefix_text)   k = number of lines in prefix (1..n)
    for vi, v in enumerate(variants):
        for pi, lines in enumerate(v["lines"]):
            ls = [l for l in lines if l and l.strip()]
            for k in range(1, len(ls) + 1):
                jobs.append((vi, pi, k, "\n".join(ls[:k])))
    print(f"[jobs] {len(jobs)} nested-prefix critic scorings", flush=True)

    # ── phase 2: critic FVE per prefix ──────────────────────────────────────
    print("[phase2] loading critic…", flush=True)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{CRITIC_SUBDIR}", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    critic.to(dev).eval()

    gold_norm = {}

    def gnorm(vi, pi):
        if (vi, pi) not in gold_norm:
            g = torch.tensor(gold[(vi, pi)], dtype=torch.float32)
            gn = g / g.norm().clamp_min(1e-12) * MSE_SCALE
            gold_norm[(vi, pi)] = (gn, ((gn - MU) ** 2).mean().item())
        return gold_norm[(vi, pi)]

    fve = np.full(len(jobs), np.nan)
    pad = tok.eos_token_id
    # pre-encode once, sort by length + greedy token-budget batching to kill padding waste
    print("[phase2] encoding + sorting prompts…", flush=True)
    enc = [tok.encode(TPL.format(explanation=j[3]), add_special_tokens=False)[:1024] for j in jobs]
    order = sorted(range(len(jobs)), key=lambda i: len(enc[i]))
    BUDGET, MAX_B = 24000, 192
    t0 = time.time(); done = 0
    with torch.inference_mode():
        i = 0
        while i < len(order):
            maxlen = len(enc[order[i]]); cnt = 1
            while i + cnt < len(order) and cnt < MAX_B:
                ml = max(maxlen, len(enc[order[i + cnt]]))
                if ml * (cnt + 1) > BUDGET:
                    break
                maxlen = ml; cnt += 1
            idxs = order[i: i + cnt]
            m = max(len(enc[x]) for x in idxs)
            bx = torch.full((cnt, m), pad, dtype=torch.long, device=dev)
            attn = torch.zeros((cnt, m), dtype=torch.long, device=dev)
            for r, x in enumerate(idxs):
                q = enc[x]; bx[r, :len(q)] = torch.tensor(q, dtype=torch.long); attn[r, :len(q)] = 1
            preds = critic_predict(critic, bx, attn, MSE_SCALE).float().cpu()
            for r, x in enumerate(idxs):
                vi, pi = jobs[x][0], jobs[x][1]
                gn, denom = gnorm(vi, pi)
                pn = preds[r] / preds[r].norm().clamp_min(1e-12) * MSE_SCALE
                fve[x] = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            i += cnt; done += cnt
            if done % 4000 < cnt:
                rate = done / (time.time() - t0)
                print(f"  [{done}/{len(jobs)}] {rate:.0f} jobs/s "
                      f"eta {(len(jobs)-done)/max(rate,1e-9)/60:.0f}min", flush=True)

    # ── assemble: cumulative -> marginal per (variation, position) ──────────
    cum = {}
    for (vi, pi, k, _), s in zip(jobs, fve):
        cum.setdefault((vi, pi), {})[k] = float(s)
    out = []
    for vi, v in enumerate(variants):
        marg = []
        for pi, lines in enumerate(v["lines"]):
            ls = [l for l in lines if l and l.strip()]
            c = cum.get((vi, pi), {})
            prev = 0.0; row = []
            for k in range(1, len(ls) + 1):
                cur = c.get(k, prev)
                row.append(round(cur - prev, 5)); prev = cur
            marg.append(row)
        out.append({"id": v["id"], "eval_score": v["eval_score"],
                    "positions": v["positions"], "marginal_fve": marg})
    json.dump({"model": "mat_rl", "variants": out}, open(HERE / "realism_variants_fve.json", "w"))
    print("[saved] realism_variants_fve.json", flush=True)
    print("FVE_DONE", flush=True)


if __name__ == "__main__":
    main()
