"""Sentence-level ablation FVE for the STANDARD 27B NLA (space_std precache).

The matryoshka-vs-traditional comparison needs the same leave-one-out
measurement on the standard model, at the fair unit size: its `<explanation>`
prose parsed into SENTENCES (quote-parity-aware; slices reconstruct the
original byte-for-byte), matching the granularity of matryoshka's authored
lines (~4.5 sentences vs ~10 lines per explanation, near-identical full-FVE
baselines: 0.530 vs 0.503).

Per precached position with sentence units s1..sS:
  pfx[k]  = FVE of s1..sk           (cumulative curve → marginal credit m_k)
  loo[k]  = FVE of all \\ {sk}       (damage d_k = full − loo[k])
  solo[k] = FVE of {sk} alone
  full    = FVE of everything       (sanity vs precache fve[-1])

If the standard critic goes OOD on subset inputs, d_k will systematically
exceed m_k (deleting a sentence destroys more than it carried) — matryoshka's
measured relationship is the opposite (redundancy: d_k < m_k).

Same one-A100 sequential recipe as loo_precompute.py. Resumable via --workdir.

    python std_loo_precompute.py --space space_std --out space_std/loo.json
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

# KEEP IN SYNC with space_std/app.py
MODEL_REPO = "ceselder/qwen3.6-27b-nla-L42"
BASE_ID = "Qwen/Qwen3.6-27B"
TOK_SUBDIR = "av_sft_lora"
CRITIC_SUBDIR = "rl_critic_step400"
LAYER = 42


def sent_units(line: str) -> list[str]:
    """Slice a snippet line into sentence units at [.!?] boundaries outside
    double quotes; ''.join(units) == line exactly (whitespace kept on the
    left unit), so any subset rebuilds clean text."""
    bounds = []
    for m in re.finditer(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])', line):
        prefix = line[: m.end()]
        if (prefix.count('"') + prefix.count('“') + prefix.count('”')) % 2 == 0:
            bounds.append(m.end())
    units, prev = [], 0
    for b in bounds:
        units.append(line[prev:b]); prev = b
    units.append(line[prev:])
    return [u for u in units if u.strip()]


def units_of(lines: list[str]) -> list[tuple[int, str]]:
    return [(li, u) for li, ln in enumerate(lines) for u in sent_units(ln)]


def rebuild(units: list[tuple[int, str]]) -> str:
    """Subset of (line_idx, text) units → explanation body with the original
    snippet/newline structure (empty snippets dropped)."""
    by_line: dict[int, list[str]] = {}
    for li, u in units:
        by_line.setdefault(li, []).append(u)
    return "\n".join("".join(us).strip() for _, us in sorted(by_line.items()))


class _StopForward(Exception):
    pass


@torch.inference_mode()
def extract_all(entries, dev):
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
    ap.add_argument("--space", default="space_std")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", default="std_loo_work")
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

    acts_path = work / "acts.npz"
    if acts_path.exists():
        z = np.load(acts_path)
        acts = [z[f"a{i}"] for i in range(len(entries))]
        print(f"[phase1] resumed {len(acts)} activation mats", flush=True)
    else:
        acts = extract_all(entries, dev)
        np.savez_compressed(acts_path, **{f"a{i}": a for i, a in enumerate(acts)})
        print(f"[phase1] saved → {acts_path}", flush=True)

    # job list: (entry, pos, kind, k, explanation_text)
    jobs = []
    for ei, e in enumerate(entries):
        for pos, r in enumerate(e["results"]):
            if not r or not r.get("lines"):
                continue
            U = units_of(r["lines"])
            jobs.append((ei, pos, "full", -1, rebuild(U)))
            for k in range(len(U)):
                jobs.append((ei, pos, "pfx", k, rebuild(U[: k + 1])))
                jobs.append((ei, pos, "solo", k, U[k][1].strip()))
                if len(U) > 1:
                    jobs.append((ei, pos, "loo", k, rebuild(U[:k] + U[k + 1:])))
    from collections import Counter
    print(f"[jobs] {len(jobs)} critic scorings {Counter(j[2] for j in jobs)}", flush=True)

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

    # assemble aligned with precache entries/positions
    by_pos = {}
    for j, s in zip(jobs, scores):
        by_pos.setdefault((j[0], j[1]), {}).setdefault(j[2], {})[j[3]] = float(s)
    out = {"unit": "sentence", "entries": []}
    n_mismatch = 0
    for ei, e in enumerate(entries):
        full_a, loo_a, solo_a, pfx_a, units_a = [], [], [], [], []
        for pos, r in enumerate(e["results"]):
            d = by_pos.get((ei, pos))
            if not d:
                for a in (full_a, loo_a, solo_a, pfx_a, units_a):
                    a.append(None)
                continue
            U = units_of(r["lines"])
            S = len(U)
            full = d["full"][-1]
            full_a.append(round(full, 5))
            pfx_a.append([round(d["pfx"][k], 5) for k in range(S)])
            solo_a.append([round(d["solo"][k], 5) for k in range(S)])
            loo_a.append([round(d["loo"][k], 5) for k in range(S)] if "loo" in d else None)
            units_a.append([u for _, u in U])
            if abs(full - r["fve"][-1]) > 0.02:
                n_mismatch += 1
        out["entries"].append({"text": e["text"], "full": full_a, "loo": loo_a,
                               "solo": solo_a, "pfx": pfx_a, "units": units_a})
    print(f"[sanity] full-FVE mismatch (> 0.02 vs precache) at {n_mismatch} positions", flush=True)

    outp = args.out or str(space / "loo.json")
    json.dump(out, open(outp, "w"))
    dmg, marg = [], []
    for ee in out["entries"]:
        for f, l, p in zip(ee["full"], ee["loo"], ee["pfx"]):
            if f is None or l is None:
                continue
            m = [p[0]] + [p[k] - p[k - 1] for k in range(1, len(p))]
            dmg += [f - x for x in l]; marg += m
    dmg, marg = np.array(dmg), np.array(marg)
    print(f"[stats] {len(dmg)} units: damage mean={dmg.mean():.4f} p50={np.percentile(dmg,50):.4f} "
          f"p95={np.percentile(dmg,95):.4f}; damage−marginal mean={(dmg-marg).mean():.4f} "
          f"p50={np.percentile(dmg-marg,50):.4f}")
    print(f"[saved] {outp}", flush=True)


if __name__ == "__main__":
    main()
