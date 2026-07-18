"""Subset-FVE scoring for the hallucination-marginal comparison (GPU-side).

Scores every prefix / solo / leave-one-out subset of BOTH 27B NLAs'
already-generated explanations (suffix-eval corpus: 250 clean held-out
contexts x 4 rollouts per model) through each model's OWN critic:

  mat : items = the matryoshka model's authored lines
  std : items = sentence units of the standard model's <explanation> prose
        (quote-parity splitter, byte-exact reassembly — same as
        std_loo_precompute.py)

Marginal FVE of item k is computed downstream as pfx[k] - pfx[k-1]
(pfx[0] itself for k=0); this script only records the raw subset scores.

Phase 1 (arm-independent, cached to --workdir/acts.npy): raw-base L42
activation at the LAST token of each context's verbatim prefix_ids —
identical indexing to suffix_gen.py, so the vectors are the ones the
explanations were generated from. Extracted norms are cross-checked
against the act_norm recorded per entry at generation time.

Phase 2: the arm's rl_critic_step400 batch-scores every subset. FVE
convention matches the space/loo pipeline: pred and gold both normalized
to mse_scale (sqrt(5120)), denominator MSE(mu, gold) with the shared
corpus mean mu.npy (byte-identical between the two Spaces).

Run ON the GPU box, once per arm (acts.npy is shared):
    python score_subsets.py --arm mat --out results/subset_scores_mat.json
    python score_subsets.py --arm std --out results/subset_scores_std.json
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

HERE = Path(__file__).resolve().parent
BASE_ID = "Qwen/Qwen3.6-27B"
LAYER = 42
D = 5120

ARMS = {
    "mat": {"repo": "ceselder/nla-qwen36-27b-matryoshka",
            "tok_sub": "warmstart_av_lora", "critic_sub": "rl_critic_step400",
            "sidecar": "sidecar_mat"},
    "std": {"repo": "ceselder/qwen3.6-27b-nla-L42",
            "tok_sub": "av_sft_lora", "critic_sub": "rl_critic_step400",
            "sidecar": "sidecar_std"},
}


# ── sentence units for the std arm (verbatim from std_loo_precompute.py) ──────
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
    by_line: dict[int, list[str]] = {}
    for li, u in units:
        by_line.setdefault(li, []).append(u)
    return "\n".join("".join(us).strip() for _, us in sorted(by_line.items()))


class _StopForward(Exception):
    """Raised from the L42 hook to skip blocks L43..N."""


@torch.inference_mode()
def extract_acts(contexts, dev):
    """L42 activation at the last token of each context's prefix_ids (the
    manifest guarantees last token == position t). Returns [N, D] fp32."""
    print("[phase1] loading raw base for extraction…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    base.eval()
    layers = resolve_decoder_layers(base)
    grabbed = {}

    def grab(module, inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
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
                print(f"  extracted [{k+1}/{len(contexts)}] "
                      f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        h.remove()
    del base
    torch.cuda.empty_cache()
    return vecs


def build_jobs(arm_name, entries):
    """(ei, ri, kind, k, text) for every subset of every rollout.
    kind: pfx (cumulative first k+1 items), solo, loo. full == pfx[n-1]."""
    jobs, units_store = [], {}
    for ei, e in enumerate(entries):
        for ri, lines in enumerate(e["explanations"]):
            if not lines:
                continue
            if arm_name == "mat":
                items = [(k, ln) for k, ln in enumerate(lines)]
                mk_text = lambda sub: "\n".join(t for _, t in sub)
            else:
                items = units_of(lines)
                mk_text = rebuild
            if not items:
                continue
            units_store[(ei, ri)] = [t for _, t in items]
            n = len(items)
            for k in range(n):
                jobs.append((ei, ri, "pfx", k, mk_text(items[: k + 1])))
                jobs.append((ei, ri, "solo", k,
                             items[k][1].strip() if arm_name == "std" else items[k][1]))
                if n > 1:
                    jobs.append((ei, ri, "loo", k, mk_text(items[:k] + items[k + 1:])))
    return jobs, units_store


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["mat", "std"], required=True)
    ap.add_argument("--suffix-eval", default=str(HERE.parent / "qwen3.6-27b-suffix-eval"),
                    help="dir with data/manifest.json + results/explanations_*.json")
    ap.add_argument("--mu", default=str(HERE / "data" / "mu.npy"))
    ap.add_argument("--workdir", default=str(HERE / "score_work"))
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first N contexts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    arm = ARMS[args.arm]
    se = Path(args.suffix_eval)
    work = Path(args.workdir); work.mkdir(exist_ok=True)
    dev = "cuda"

    contexts = json.load(open(se / "data" / "manifest.json"))["contexts"]
    payload = json.load(open(se / "results" / f"explanations_{args.arm}.json"))
    entries = payload["entries"]
    assert len(entries) == len(contexts)
    if args.limit:
        contexts, entries = contexts[: args.limit], entries[: args.limit]

    root = snapshot_download(arm["repo"],
                             allow_patterns=[f"{arm['tok_sub']}/*",
                                             f"{arm['critic_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(str(se / arm["sidecar"]), tok)
    MSE_SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(args.mu), dtype=torch.float32)
    print(f"[cfg] arm={args.arm} mse_scale={MSE_SCALE:.2f} tpl={TPL!r}", flush=True)

    # ── phase 1: activations (shared across arms; cached) ────────────────────
    acts_path = work / "acts.npy"
    if acts_path.exists():
        vecs = np.load(acts_path)[: len(contexts)]
        print(f"[phase1] reusing {vecs.shape} from {acts_path}", flush=True)
    else:
        vecs = extract_acts(contexts, dev)
        np.save(acts_path, vecs)
        print(f"[phase1] saved → {acts_path}", flush=True)

    # sanity: norms must match the ones recorded at generation time
    norms = np.linalg.norm(vecs, axis=1)
    gen_norms = np.array([e["act_norm"] for e in entries])
    rel = np.abs(norms - gen_norms) / np.maximum(gen_norms, 1e-9)
    print(f"[sanity] extracted-vs-generation act_norm rel-diff: "
          f"median {np.median(rel):.2e} max {rel.max():.2e}", flush=True)
    assert np.median(rel) < 0.02, "extraction does not reproduce generation-time norms"

    # ── job list ─────────────────────────────────────────────────────────────
    jobs, units_store = build_jobs(args.arm, entries)
    from collections import Counter
    print(f"[jobs] {len(jobs)} critic scorings {Counter(j[2] for j in jobs)}", flush=True)

    # ── phase 2: critic scoring (checkpointed) ───────────────────────────────
    print("[phase2] loading critic…", flush=True)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{arm['critic_sub']}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    critic.to(dev).eval()

    scores_path = work / f"scores_{args.arm}.npy"
    scores = np.load(scores_path) if scores_path.exists() else np.full(len(jobs), np.nan)
    assert len(scores) == len(jobs), "resume file does not match job list"
    start = int(np.argmax(np.isnan(scores))) if np.isnan(scores).any() else len(jobs)
    if start:
        print(f"[phase2] resuming at job {start}/{len(jobs)}", flush=True)

    gold_cache = {}

    def gold(ei):
        if ei not in gold_cache:
            v = torch.tensor(vecs[ei], dtype=torch.float32)
            gn = v / v.norm().clamp_min(1e-12) * MSE_SCALE
            gold_cache[ei] = (gn, ((gn - MU) ** 2).mean().item())
        return gold_cache[ei]

    pad = tok.eos_token_id
    CAP = 1024
    t0 = time.time()
    with torch.inference_mode():
        for b0 in range(start, len(jobs), args.batch):
            chunk = jobs[b0: b0 + args.batch]
            idlists = [tok.encode(TPL.format(explanation=j[4]),
                                  add_special_tokens=False) for j in chunk]
            # the critic anchors on the trailing "<summary>" of TPL; truncating
            # from the right would sever it and silently poison the score. Fail
            # loud instead of clipping (measured max here ~272 tokens << CAP).
            over = [len(x) for x in idlists if len(x) > CAP]
            assert not over, f"critic input exceeds {CAP} tokens ({max(over)}); would sever suffix anchor"
            m = max(len(x) for x in idlists)
            bx = torch.full((len(chunk), m), pad, dtype=torch.long, device=dev)
            attn = torch.zeros((len(chunk), m), dtype=torch.long, device=dev)
            for r, q in enumerate(idlists):
                bx[r, : len(q)] = torch.tensor(q, dtype=torch.long)
                attn[r, : len(q)] = 1
            preds = critic_predict(critic, bx, attn, MSE_SCALE).float().cpu()
            for ci, (j, pred) in enumerate(zip(chunk, preds)):
                gn, denom = gold(j[0])
                pn = pred / pred.norm().clamp_min(1e-12) * MSE_SCALE
                scores[b0 + ci] = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            nb = (b0 - start) // args.batch + 1
            if nb % 100 == 0:
                np.save(scores_path, scores)
                done = b0 + len(chunk) - start
                rate = done / (time.time() - t0)
                eta = (len(jobs) - b0 - len(chunk)) / max(rate, 1e-9) / 60
                print(f"  [{b0 + len(chunk)}/{len(jobs)}] {rate:.0f} jobs/s "
                      f"eta {eta:.0f}min", flush=True)
    np.save(scores_path, scores)
    # this script has no external FVE reference to check against (suffix_gen
    # computes none), so guard what we can: every job must have a finite score.
    assert np.isfinite(scores).all(), \
        f"{int((~np.isfinite(scores)).sum())} non-finite scores — critic misconfig?"

    # ── assemble ─────────────────────────────────────────────────────────────
    by_ro = {}
    for j, s in zip(jobs, scores):
        by_ro.setdefault((j[0], j[1]), {}).setdefault(j[2], {})[j[3]] = float(s)
    out_entries = []
    fulls = []
    for ei, e in enumerate(entries):
        rollouts = []
        for ri in range(len(e["explanations"])):
            d = by_ro.get((ei, ri))
            if not d:
                rollouts.append(None)
                continue
            n = len(d["pfx"])
            rec = {"units": units_store[(ei, ri)],
                   "pfx": [round(d["pfx"][k], 5) for k in range(n)],
                   "solo": [round(d["solo"][k], 5) for k in range(n)],
                   "loo": ([round(d["loo"][k], 5) for k in range(n)]
                           if "loo" in d else None),
                   "full": round(d["pfx"][n - 1], 5)}
            rollouts.append(rec)
            fulls.append(rec["full"])
        out_entries.append({"ci": e["ci"], "act_norm": e["act_norm"],
                            "norm_outlier": e["norm_outlier"],
                            "rollouts": rollouts})

    out = {"meta": {"arm": args.arm, "repo": arm["repo"],
                    "critic": arm["critic_sub"], "unit":
                    "line" if args.arm == "mat" else "sentence",
                    "mse_scale": MSE_SCALE, "n_jobs": len(jobs),
                    "gpu": torch.cuda.get_device_name(0),
                    "norm_reldiff_median": float(np.median(rel))},
           "entries": out_entries}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"))

    fulls = np.array(fulls)
    print(f"[stats] {len(fulls)} rollouts | full-FVE mean {fulls.mean():.3f} "
          f"p50 {np.percentile(fulls, 50):.3f}", flush=True)
    marg = []
    for e in out_entries:
        for r in e["rollouts"]:
            if r:
                p = r["pfx"]
                marg += [p[0]] + [p[k] - p[k - 1] for k in range(1, min(3, len(p)))]
    marg = np.array(marg)
    print(f"[stats] top-3 marginals: n={len(marg)} neg-frac "
          f"{(marg < 0).mean():.3f} p05 {np.percentile(marg, 5):.4f}", flush=True)
    print(f"[saved] {args.out}", flush=True)
    print(f"SCORE_DONE_{args.arm.upper()}", flush=True)


if __name__ == "__main__":
    main()
