"""Quick-signal experiment, phase B: verbalize each of the 100 system-prompt
variations with the matryoshka RL NLA.

Each variation is rendered as a system message, layer-42 activations are extracted
at a uniform sample of positions in one forward pass (adapters disabled), and every
position is verbalized at T=1. Downstream we judge each explanation LINE for
eval-awareness and test whether the variation's eval-y score correlates with WHERE
(line index) the NLA first flags it.

  python3 realism_variants_verbalize.py            # all 100, mat_rl
  python3 realism_variants_verbalize.py --n-pos 30

Reuses control_gen.CFG['mat_rl']. Fits on one A100 80GB (AV only).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

import control_gen as CG

HERE = Path(__file__).resolve().parent
SCORED = HERE / "realism_variants_scored.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mat_rl", choices=list(CG.CFG))
    ap.add_argument("--n-pos", type=int, default=30)
    ap.add_argument("--av-batch", type=int, default=24)
    args = ap.parse_args()
    C = CG.CFG[args.model]

    pats = [f"{C['tok_sub']}/*"] + [f"{sub}/*" for _, sub in C["adapters"]]
    root = snapshot_download(C["repo"], allow_patterns=pats)
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok_sub']}")
    cfg = load_nla_config(str(HERE / C["sidecar"]), tok)
    render_tok = AutoTokenizer.from_pretrained(CG.BASE_ID)

    print("[load] base + adapter…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        CG.BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    actor = None
    for name, sub in C["adapters"]:
        if actor is None:
            actor = PeftModel.from_pretrained(base, f"{root}/{sub}", adapter_name=name)
        else:
            actor.load_adapter(f"{root}/{sub}", adapter_name=name)
    actor.base_model.set_adapter([n for n, _ in C["adapters"]])
    actor.eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    layers = resolve_decoder_layers(actor.get_base_model())

    # fixed AV prompt
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], cfg.injection_char, tok)
    assert ptxt.endswith(CG.THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(CG.THINK_OPEN)] + CG.PRECLOSED + C["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    parse = CG.bullet_lines if C["mode"] == "bullet" else CG.expl_lines

    variants = json.load(open(SCORED))

    # ---- phase 1: extract activations for every (variation, position) ----
    # one adapters-disabled forward per variation (cheap); accumulate a FLAT list of
    # independent activation vectors so verbalization can batch across variations.
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    meta = []           # per variation: {id, eval_score, ntok, positions}
    flat_acts = []      # flat activation vectors, aligned to (var_idx, k)
    flat_owner = []     # (var_idx) for each flat act
    head, tail = "<|im_start|>system\n", "<|im_end|>\n"
    n_head = len(render_tok.encode(head, add_special_tokens=False))
    n_tail = len(render_tok.encode(tail, add_special_tokens=False))
    for vi, v in enumerate(variants):
        text = head + v["text"].strip() + tail
        ids = render_tok.encode(text, add_special_tokens=False)
        lo = n_head                       # first token of the variation content (skip only the wrapper)
        hi = len(ids) - 1 - n_tail        # last content token (skip trailing <|im_end|>)
        n = max(1, min(args.n_pos, hi - lo + 1))
        positions = sorted(set(np.linspace(lo, hi, n).astype(int).tolist()))
        h = layers[CG.LAYER].register_forward_hook(grab)
        try:
            with torch.inference_mode(), actor.disable_adapter():
                actor(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
        finally:
            h.remove()
        acts = grabbed["h"][0].float().cpu().numpy()[positions]
        meta.append({"id": v["id"], "eval_score": v["eval_score"], "ntok": len(ids),
                     "positions": positions, "lines": [None] * len(positions)})
        for k in range(len(positions)):
            flat_acts.append(acts[k]); flat_owner.append((vi, k))
    print(f"[extract] {len(flat_acts)} activations from {len(variants)} variations", flush=True)

    # ---- phase 2: verbalize the flat list in large cross-variation batches ----
    torch.manual_seed(0)
    with torch.inference_mode():
        for s in range(0, len(flat_acts), args.av_batch):
            chunk = flat_acts[s: s + args.av_batch]
            owners = flat_owner[s: s + args.av_batch]
            pt = base_pt.repeat(len(chunk), 1)
            vref[0] = torch.tensor(np.stack(chunk), dtype=torch.float32).cuda()
            try:
                gen = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                     max_new_tokens=CG.MAX_NEW, do_sample=True, temperature=1.0,
                                     top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
            finally:
                vref[0] = None
            for o, (vi, k) in zip(gen, owners):
                txt = C["prefill"] + tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                meta[vi]["lines"][k] = parse(txt)
            if (s // args.av_batch) % 5 == 0:
                print(f"  verbalize {min(s + args.av_batch, len(flat_acts))}/{len(flat_acts)}", flush=True)

    out = meta
    outp = f"realism_variants_verb_{args.model}.json"
    json.dump({"model": args.model, "variants": out}, open(HERE / outp, "w"))
    print(f"[saved] {outp}  ({len(out)} variations)", flush=True)
    print("VVERB_DONE", flush=True)


if __name__ == "__main__":
    main()
