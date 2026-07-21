"""Phase 2b: patched couplet continuation (run ON the GPU box, after 2a).

For each steering vector, REPLACE the clean base's L42 residual at the final
context position (the " head" token) during prefill, then sample N T=1
continuations. Conditions:

  baseline        no intervention (sanity vs phase-1 cache)
  self_vhead      replace with the true v_head itself (patching no-op control)
  donor_moon      replace with the real activation from the "…moon" couplet
                  (direct-patch ceiling: what CAN this site do?)
  {arm}:{name}    each critic vector from work/critic_vecs_{arm}.npz,
                  rescaled to ||v_head|| (norm-matched so magnitude is not
                  the confound; raw critic norms are recorded in meta)

Usage:  python rhyme_phase2b.py --out results/steered_cont.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.utils.arch_adapters import resolve_decoder_layers

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK", "/workspace/rs/work"))
LAYER = 42


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=os.environ.get("BASE", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ctx = json.load(open(WORK / "context.json"))
    ctx_ids = ctx["ctx_ids"]
    patch_pos = len(ctx_ids) - 1
    v_head = np.load(WORK / "v_head.npy")
    v_moon = np.load(WORK / "v_moon.npy")
    nh = float(np.linalg.norm(v_head))

    conds, raw_norms = {}, {}
    conds["baseline"] = None
    conds["self_vhead"] = v_head.copy()
    conds["donor_moon"] = v_moon.copy()          # real activation: patch raw
    for arm in ("mat", "std"):
        p = WORK / f"critic_vecs_{arm}.npz"
        if not p.exists():
            print(f"[skip] {p} missing", flush=True)
            continue
        z = np.load(p)
        for name in z.files:
            v = z[name].astype(np.float32)
            raw_norms[f"{arm}:{name}"] = float(np.linalg.norm(v))
            conds[f"{arm}:{name}"] = v / np.linalg.norm(v) * nh   # norm-matched

    tok = AutoTokenizer.from_pretrained(args.base)
    assert tok.decode(tok.encode(ctx["ctx_text"], add_special_tokens=False)) == ctx["ctx_text"]
    print(f"[load] base {args.base} (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
    layers = resolve_decoder_layers(base)

    vec_ref = [None]     # torch [D] on cuda, or None

    def patch(m, i, o):
        if vec_ref[0] is None:
            return
        h = o[0] if isinstance(o, tuple) else o
        if h.shape[1] >= patch_pos + 1:          # prefill covers the site
            h[:, patch_pos, :] = vec_ref[0].to(h.dtype)
        return o

    hook = layers[LAYER].register_forward_hook(patch)
    out_all = {}
    try:
        pt = torch.tensor([ctx_ids], device="cuda").repeat(args.samples, 1)
        for name, vec in conds.items():
            vec_ref[0] = None if vec is None else torch.tensor(vec, device="cuda")
            torch.manual_seed(args.seed + 1234)   # same seed across conditions
            with torch.inference_mode():
                out = base.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                    max_new_tokens=args.max_new, do_sample=True,
                                    temperature=1.0, top_p=1.0, top_k=0,
                                    pad_token_id=tok.eos_token_id)
            vec_ref[0] = None
            conts = [tok.decode(o[pt.shape[1]:], skip_special_tokens=True) for o in out]
            out_all[name] = conts
            print(f"  {name:24s} | {conts[0][:90].replace(chr(10), ' ⏎ ')}", flush=True)
    finally:
        hook.remove()

    payload = {"meta": {"base": args.base, "layer": LAYER, "patch_pos": patch_pos,
                        "samples": args.samples, "seed": args.seed,
                        "v_head_norm": nh, "v_moon_norm": float(np.linalg.norm(v_moon)),
                        "critic_raw_norms": raw_norms,
                        "gpu": torch.cuda.get_device_name(0)},
               "context": ctx["ctx_text"], "continuations": out_all}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"[write] {args.out} ({len(out_all)} conditions)", flush=True)
    print("PHASE2B_DONE", flush=True)


if __name__ == "__main__":
    main()
