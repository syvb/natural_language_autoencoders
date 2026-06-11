"""Gate C causal KL judge (tie-breaker metric, pre-registered as veto-only).

For each selected holdout position: forward the doc, verify the live v18
matches stored acts18 (cos>=0.999), take the natural next-token distribution
at p as reference; then patch candidate v22_hat = v18 + scaled diff
prediction into the output of decoder layer 21 (== hidden_states[22]) and
measure next-token KL(ref || patched). Lower = functionally closer.

Candidates come from kl_candidates.json: {name: {"preds": file, "tidx": file}}.
Each prediction is rescaled to the true per-position ||d|| (uniform across
arms). Built-ins: zero_diff (patch v18 alone) and _sanity_true_diff (patch
the exact v22; should give KL ~ 0 up to dtype noise).

Positions = holdout split, intersected across all candidates' tidx, capped
at --max-pos (seeded subsample). doc_tokens.jsonl supplies token IDs — no
corpus re-streaming.

Usage (on pod): python kl_judge.py --max-pos 500
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(WORK / "kl_candidates.json"))
    ap.add_argument("--max-pos", type=int, default=500)
    ap.add_argument("--out", default=str(WORK / "kl_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = json.loads((WORK / "meta2.json").read_text())
    acts18 = np.load(WORK / "acts18.npy", mmap_mode="r")
    dtrue = np.load(WORK / "diff_targets.npy", mmap_mode="r")

    cand_files = json.loads(Path(args.candidates).read_text())
    cands = {}
    for name, files in cand_files.items():
        preds = np.load(WORK / files["preds"]).astype(np.float32)
        tidx = np.load(WORK / files["tidx"])
        cands[name] = dict(zip(tidx.tolist(), preds))

    usable = [i for i, m in enumerate(meta)
              if m["split"] == "holdout" and all(i in c for c in cands.values())]
    rng = np.random.default_rng(args.seed)
    if len(usable) > args.max_pos:
        usable = sorted(rng.choice(usable, args.max_pos, replace=False).tolist())
    print(f"[kl] {len(usable)} positions ({len(cands)} candidate arms)", flush=True)

    doc_tokens = {}
    for line in (WORK / "doc_tokens.jsonl").open():
        r = json.loads(line)
        doc_tokens[r["doc_idx"]] = r["ids"]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda").eval()

    PATCH = {}

    def hook(mod, inp, out):
        # decoder layers return a tuple in older transformers, a bare tensor in newer
        h = out[0] if isinstance(out, tuple) else out
        h[:, PATCH["pos"]] = PATCH["vec"]
        return out

    layer = model.model.layers[21]  # output == hidden_states[22]

    names = ["zero_diff", "_sanity_true_diff"] + list(cands.keys())
    results = {name: [] for name in names}
    results["_positions"] = usable
    n_verified = n_skipped = 0
    for irow in usable:
        m = meta[irow]
        ids = doc_tokens[m["doc_idx"]]
        p = m["pos"]
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            ref_out = model(x, output_hidden_states=True)
        v18_live = ref_out.hidden_states[18][0, p].float().cpu().numpy()
        gold18 = np.asarray(acts18[irow], dtype=np.float32)
        cos18 = float(v18_live @ gold18
                      / (np.linalg.norm(v18_live) * np.linalg.norm(gold18)))
        if cos18 < 0.999:
            n_skipped += 1
            continue
        n_verified += 1
        ref_logp = F.log_softmax(ref_out.logits[0, p].float(), dim=-1)

        d = np.asarray(dtrue[irow], dtype=np.float32)
        dn = np.linalg.norm(d)
        vecs = [np.zeros(3584, np.float32), d]
        for name in list(cands.keys()):
            g = cands[name][irow]
            gn = np.linalg.norm(g)
            vecs.append(g / max(gn, 1e-9) * dn if gn > 0
                        else np.zeros(3584, np.float32))
        patch = torch.tensor(np.stack([gold18 + v for v in vecs]),
                             dtype=torch.bfloat16, device="cuda")

        PATCH["vec"], PATCH["pos"] = patch, p
        h = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(x.repeat(len(vecs), 1))
        finally:
            h.remove()
        logp = F.log_softmax(out.logits[:, p].float(), dim=-1)
        kls = (ref_logp.exp() * (ref_logp - logp)).sum(-1).cpu().numpy()
        for k, name in enumerate(names):
            results[name].append(float(kls[k]))
        if n_verified % 50 == 0:
            print(f"[kl] {n_verified} verified, {n_skipped} skipped", flush=True)

    print(f"[kl] verified {n_verified}, skipped {n_skipped}")
    Path(args.out).write_text(json.dumps(results))
    for name in names:
        a = np.array(results[name])
        print(f"  {name:24} KL mean {a.mean():.4f}  median {np.median(a):.4f}")


if __name__ == "__main__":
    main()
