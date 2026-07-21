"""Phase 1 of the couplet rhyme-steering probe (run ON the GPU box).

Fixed context = the exact start of a Qwen3.6-27B chat response:

    <|im_start|>user
    Write a rhyming couplet.<|im_end|>
    <|im_start|>assistant
    <think>

    </think>

    The sun goes down to rest its head

This script, per arm (--model mat|std, same protocol as suffix_gen.py):
  1. extracts the CLEAN-base L42 activation at the final token (" head")
     -> cached to work/v_head.npy (bit-identical across arms)
  2. extracts a DONOR activation at " moon" from a parallel couplet context
     (direct-patch ceiling control for phase 2) -> work/v_moon.npy
  3. samples 10 T=1 baseline continuations of the couplet (no intervention)
     -> cached to work/baseline_cont.json
  4. samples 10 T=1 AV explanations of v_head with the arm's actor
     -> results/explanations_rhyme_{arm}.json  (raw text + item lines)

NOTE: the " head" token sits at position ~35, below stage-0's _MIN_POSITION=50
— slightly OOD for the NLAs, acceptable for a qualitative probe (recorded in
meta so the writeup can flag it).

Usage:  python rhyme_phase1.py --model mat --out results/explanations_rhyme_mat.json
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("WORK", "/workspace/rs/work"))
LAYER = 42
D = 5120
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")

COUPLET_CTX = ("<|im_start|>user\nWrite a rhyming couplet.<|im_end|>\n"
               "<|im_start|>assistant\n<think>\n\n</think>\n\n"
               "The sun goes down to rest its head")
DONOR_CTX = ("<|im_start|>user\nWrite a rhyming couplet.<|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n"
             "The night is calm beneath the moon")

ARMS = {
    "mat": {"repo": "ceselder/nla-qwen36-27b-matryoshka",
            "tok_sub": "warmstart_av_lora", "merge_sub": None,
            "rl_sub": "rl_av_lora_iter400", "prefill": "", "stop": None,
            "sidecar": str(HERE / "sidecar_mat")},
    "std": {"repo": "ceselder/qwen3.6-27b-nla-L42",
            "tok_sub": "av_sft_lora", "merge_sub": "av_sft_lora",
            "rl_sub": "av_rl_lora_step400", "prefill": "<explanation>\n",
            "stop": "</explanation>", "sidecar": str(HERE / "sidecar_std")},
}


def _lines(text, cut=None):
    if cut and cut in text:
        text = text.split(cut, 1)[0]
    return [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]


def extract_last_tok_act(base, layers, tok, ctx_text, want_last):
    ids = tok.encode(ctx_text, add_special_tokens=False)
    assert tok.decode(ids) == ctx_text, "tokenize/decode round-trip drift"
    last = tok.decode(ids[-1:])
    assert last == want_last, f"final token is {last!r}, wanted {want_last!r}"
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    h = layers[LAYER].register_forward_hook(grab)
    try:
        with torch.inference_mode():
            base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
    finally:
        h.remove()
    v = grabbed["h"][0, -1].float().cpu().numpy()
    return ids, v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mat", "std"], required=True)
    ap.add_argument("--base", default=os.environ.get("BASE", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--rollouts", type=int, default=10)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    arm = ARMS[args.model]
    WORK.mkdir(parents=True, exist_ok=True)

    root = snapshot_download(arm["repo"], allow_patterns=[f"{arm['tok_sub']}/*",
                                                          f"{arm['rl_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(arm["sidecar"], tok)
    inj_char = cfg.injection_char

    print(f"[load] base {args.base} (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
    layers = resolve_decoder_layers(base)

    # ── 1+2: activations (clean base; cached, arm-independent) ────────────────
    ctx_ids, v_head = extract_last_tok_act(base, layers, tok, COUPLET_CTX, " head")
    _, v_moon = extract_last_tok_act(base, layers, tok, DONOR_CTX, " moon")
    np.save(WORK / "v_head.npy", v_head)
    np.save(WORK / "v_moon.npy", v_moon)
    json.dump({"ctx_ids": ctx_ids, "ctx_text": COUPLET_CTX},
              open(WORK / "context.json", "w"))
    print(f"[extract] pos={len(ctx_ids)-1} ||v_head||={np.linalg.norm(v_head):.1f} "
          f"||v_moon||={np.linalg.norm(v_moon):.1f} "
          f"cos={np.dot(v_head, v_moon)/np.linalg.norm(v_head)/np.linalg.norm(v_moon):.3f}",
          flush=True)

    # ── 3: baseline continuations (clean base; cached) ────────────────────────
    bpath = WORK / "baseline_cont.json"
    if not bpath.exists():
        torch.manual_seed(args.seed + 777)
        pt = torch.tensor([ctx_ids], device="cuda").repeat(10, 1)
        with torch.inference_mode():
            out = base.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                max_new_tokens=48, do_sample=True, temperature=1.0,
                                top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
        conts = [tok.decode(o[pt.shape[1]:], skip_special_tokens=True) for o in out]
        json.dump(conts, open(bpath, "w"))
        print("[baseline] sample:", conts[0][:120].replace("\n", " ⏎ "), flush=True)

    # ── build actor (arm-specific) ────────────────────────────────────────────
    if arm["merge_sub"]:
        base = PeftModel.from_pretrained(base, f"{root}/{arm['merge_sub']}").merge_and_unload()
    actor = PeftModel.from_pretrained(base, f"{root}/{arm['rl_sub']}").eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)

    content = cfg.actor_prompt_template.format(injection_char=inj_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], inj_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + arm["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    print(f"[prompt] len={len(prompt_ids)} tail={ptxt[-40:]!r}", flush=True)

    # ── 4: K T=1 explanations of v_head, one batched generate ────────────────
    t0 = time.time()
    torch.manual_seed(args.seed)
    pt = torch.tensor([prompt_ids], device="cuda").repeat(args.rollouts, 1)
    with torch.inference_mode():
        vref[0] = torch.tensor(np.stack([v_head] * args.rollouts),
                               dtype=torch.float32).cuda()
        try:
            out = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                 max_new_tokens=args.max_new, do_sample=True,
                                 temperature=1.0, top_p=1.0, top_k=0,
                                 pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
    expls = []
    for o in out:
        gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
        cut = gen.split(arm["stop"], 1)[0] if arm["stop"] and arm["stop"] in gen else gen
        expls.append({"raw": cut.strip(), "lines": _lines(gen, cut=arm["stop"])})
    print(f"[explain] {args.rollouts} rollouts in {time.time()-t0:.0f}s", flush=True)

    cjk = sum(bool(CJK.search(e["raw"])) for e in expls)
    payload = {"meta": {"model": args.model, "arm": arm["repo"], "base": args.base,
                        "layer": LAYER, "pos": len(ctx_ids) - 1,
                        "pos_below_min50": len(ctx_ids) - 1 < 50,
                        "act_norm": float(np.linalg.norm(v_head)),
                        "rollouts": args.rollouts, "cjk": cjk,
                        "gpu": torch.cuda.get_device_name(0)},
               "context": COUPLET_CTX, "explanations": expls}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(f"[write] {args.out} | CJK: {cjk}/{args.rollouts} (want 0)", flush=True)
    for i, e in enumerate(expls[:3]):
        print(f"--- rollout {i} ---\n{e['raw'][:500]}", flush=True)
    print(f"PHASE1_DONE_{args.model.upper()}", flush=True)


if __name__ == "__main__":
    main()
