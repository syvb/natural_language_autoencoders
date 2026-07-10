"""Composition + reordering tests: are explanation units reusable handles?

Runs on the same box/critic as the LOO jobs (both model repos cached).

COMPOSE — sample position pairs (A, B) from different texts; build a hybrid
explanation from A's first k units + B's first k units (k=3 matryoshka lines,
k=2 standard sentences — comparable token counts); critic-encode. Project
v̂(mix) onto span(v_A, v_B): report R² (does the mix stay on the plane
spanned by the two originals?) and the balance (cos to A vs cos to B). Also
encode A's units alone / B's alone (same subset sizes) as anchors.

SHUFFLE — re-encode each explanation with its units randomly permuted;
report FVE(shuffled) − FVE(original order). Order-free units should lose ~0;
prose sentences shuffled into word salad should break the standard critic.
(Risk acknowledged: the matryoshka critic was TRAINED on salience-ordered
prefixes, so order-sensitivity here is a finding either way.)

    python compose_shuffle.py --model mat   # matryoshka (space/, line units)
    python compose_shuffle.py --model std   # standard (space_std/, sentence units)
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import critic_predict
from nla.utils.arch_adapters import resolve_decoder_layers

CFG = {
    # acts: gold activations come from the RAW BASE (shared by both NLAs) and
    # the two precaches tokenize identically (verified: ids equal) — so the
    # matryoshka run reuses the std extraction already on the box.
    "mat": dict(repo="ceselder/nla-qwen36-27b-matryoshka", tok_sub="warmstart_av_lora",
                critic_sub="rl_critic_step400", space="space",
                acts="std_loo_work/acts.npz", k_mix=3),
    "std": dict(repo="ceselder/qwen3.6-27b-nla-L42", tok_sub="av_sft_lora",
                critic_sub="rl_critic_step400", space="space_std",
                acts="std_loo_work/acts.npz", k_mix=2),
}
BASE_ID = "Qwen/Qwen3.6-27B"
N_PAIRS = 400
N_SHUFFLE = 1500
DEV = "cuda"


def sent_units(line):
    bounds = []
    for m in re.finditer(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])', line):
        if (line[: m.end()].count('"') + line[: m.end()].count('“')
                + line[: m.end()].count('”')) % 2 == 0:
            bounds.append(m.end())
    units, prev = [], 0
    for b in bounds:
        units.append(line[prev:b]); prev = b
    units.append(line[prev:])
    return [u.strip() for u in units if u.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mat", "std"], required=True)
    ap.add_argument("--batch", type=int, default=24)
    args = ap.parse_args()
    C = CFG[args.model]
    space = Path(C["space"])
    entries = json.load(open(space / "precache.json"))["entries"]
    z = np.load(C["acts"])
    acts = [z[f"a{i}"] for i in range(len(entries))]

    root = snapshot_download(C["repo"], allow_patterns=[f"{C['tok_sub']}/*", f"{C['critic_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok_sub']}")
    cfg = load_nla_config(str(space), tok)
    SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(space / "mu.npy"), dtype=torch.float32)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{C['critic_sub']}", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    critic.to(DEV).eval()

    # positions with units + gold vectors
    pool = []
    for ei, e in enumerate(entries):
        for pos, r in enumerate(e["results"]):
            if not r or not r.get("lines"):
                continue
            units = (r["lines"] if args.model == "mat"
                     else [u for ln in r["lines"] for u in sent_units(ln)])
            if len(units) >= C["k_mix"]:
                pool.append((ei, pos, units))
    print(f"[pool] {len(pool)} positions", flush=True)

    def encode(texts):
        out = []
        with torch.inference_mode():
            for b0 in range(0, len(texts), args.batch):
                chunk = texts[b0: b0 + args.batch]
                idl = [tok.encode(TPL.format(explanation=t), add_special_tokens=False)[:1024]
                       for t in chunk]
                m = max(len(x) for x in idl)
                bx = torch.full((len(chunk), m), tok.eos_token_id, dtype=torch.long, device=DEV)
                at = torch.zeros((len(chunk), m), dtype=torch.long, device=DEV)
                for r, q in enumerate(idl):
                    bx[r, : len(q)] = torch.tensor(q); at[r, : len(q)] = 1
                out.append(critic_predict(critic, bx, at, SCALE).float().cpu())
                if (b0 // args.batch) % 100 == 0:
                    print(f"  encode {b0}/{len(texts)}", flush=True)
        return torch.cat(out)

    def nrm(v):
        return v / v.norm().clamp_min(1e-12) * SCALE

    def fve(pred, gold):
        pn, gn = nrm(pred), nrm(gold)
        return 1.0 - ((pn - gn) ** 2).mean().item() / ((gn - MU) ** 2).mean().item()

    rng = random.Random(0)

    # ── COMPOSE ──────────────────────────────────────────────────────────────
    k = C["k_mix"]
    pairs = []
    while len(pairs) < N_PAIRS:
        a, b = rng.sample(range(len(pool)), 2)
        if pool[a][0] != pool[b][0]:  # different texts
            pairs.append((a, b))
    texts, meta = [], []
    for a, b in pairs:
        ua, ub = pool[a][2][:k], pool[b][2][:k]
        texts += ["\n".join(ua + ub), "\n".join(ua), "\n".join(ub)]
        meta.append((a, b))
    preds = encode(texts)
    rows = []
    for i, (a, b) in enumerate(meta):
        mix, pa, pb = preds[3 * i], preds[3 * i + 1], preds[3 * i + 2]
        va = torch.tensor(acts[pool[a][0]][pool[a][1]], dtype=torch.float32)
        vb = torch.tensor(acts[pool[b][0]][pool[b][1]], dtype=torch.float32)
        # R² of mix onto span(vA, vB) (normalized gold directions)
        A = torch.stack([nrm(va), nrm(vb)], dim=1)
        x = nrm(mix)
        sol = torch.linalg.lstsq(A, x.unsqueeze(1)).solution.squeeze()
        resid = x - A @ sol
        r2 = 1.0 - (resid @ resid).item() / (x @ x).item()
        cos_a = float(nrm(mix) @ nrm(va) / (nrm(mix).norm() * nrm(va).norm()))
        cos_b = float(nrm(mix) @ nrm(vb) / (nrm(mix).norm() * nrm(vb).norm()))
        # anchors: how well do the k-unit halves alone reconstruct their own vector?
        rows.append(dict(r2=round(r2, 4), cos_a=round(cos_a, 4), cos_b=round(cos_b, 4),
                         wa=round(float(sol[0]), 4), wb=round(float(sol[1]), 4),
                         fve_a_half=round(fve(pa, va), 4), fve_b_half=round(fve(pb, vb), 4)))
    r2s = np.array([r["r2"] for r in rows])
    wa = np.array([r["wa"] for r in rows]); wb = np.array([r["wb"] for r in rows])
    print(f"[compose] R2 onto span(vA,vB): mean {r2s.mean():.3f} p50 {np.percentile(r2s, 50):.3f} "
          f"p10 {np.percentile(r2s, 10):.3f}", flush=True)
    print(f"[compose] weights: wa {wa.mean():.3f}±{wa.std():.3f} wb {wb.mean():.3f}±{wb.std():.3f} "
          f"balance |wa-wb| p50 {np.percentile(np.abs(wa - wb), 50):.3f}", flush=True)

    # ── SHUFFLE ──────────────────────────────────────────────────────────────
    sample = rng.sample(pool, min(N_SHUFFLE, len(pool)))
    texts, golds = [], []
    for ei, pos, units in sample:
        perm = units[:]
        while len(units) > 1 and perm == units:
            rng.shuffle(perm)
        texts += ["\n".join(units), "\n".join(perm)]
        golds.append(torch.tensor(acts[ei][pos], dtype=torch.float32))
    preds = encode(texts)
    d = []
    for i, g in enumerate(golds):
        d.append(fve(preds[2 * i + 1], g) - fve(preds[2 * i], g))
    d = np.array(d)
    print(f"[shuffle] ΔFVE (shuffled − original): mean {d.mean():+.4f} "
          f"p10 {np.percentile(d, 10):+.4f} p50 {np.percentile(d, 50):+.4f} "
          f"p90 {np.percentile(d, 90):+.4f}", flush=True)

    out = {"model": args.model, "compose": rows,
           "shuffle_dfve": [round(float(x), 5) for x in d]}
    json.dump(out, open(f"compose_shuffle_{args.model}.json", "w"))
    print(f"[saved] compose_shuffle_{args.model}.json", flush=True)


if __name__ == "__main__":
    main()
