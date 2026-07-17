"""GPU-side generation for the Suffix-Prediction eval (run ON the Vast box).

Reads data/manifest.json (built locally by build_manifest.py), extracts the
layer-42 activation at token t of each context's prefix, and produces K sampled
AV explanations per activation. NO critic / FVE here — the suffix eval grades
the explanation TEXT with an external LLM, so reconstruction is irrelevant.

Two arms, selected by --model, each in its OWN canonical inference protocol
(pinned from the deployed Spaces):
  mat : base + rl_av_lora_iter400,  NO prefill,   plain bullet lines
  std : base + av_sft_lora (merged) + av_rl_lora_step400,
        prefill '<explanation>\\n', decode cut at '</explanation>'

Activations are extracted from the CLEAN base (adapters absent/disabled) and are
therefore bit-identical across the two arms — the same 250 vectors feed both.

Usage on the box (WORK defaults to /workspace):
    python suffix_gen.py --model mat --out explanations_mat.json
    python suffix_gen.py --model std --out explanations_std.json
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
WORK = os.environ.get("WORK", "/workspace")
LAYER = 42
D = 5120
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")

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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mat", "std"], required=True)
    ap.add_argument("--manifest", default=str(HERE / "data" / "manifest.json"))
    ap.add_argument("--base", default=os.environ.get("BASE", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--av-batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-lines", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N contexts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    arm = ARMS[args.model]

    man = json.load(open(args.manifest))
    contexts = man["contexts"]
    if args.limit:
        contexts = contexts[: args.limit]
    print(f"[manifest] {len(contexts)} contexts, layer {man['meta']['layer']}", flush=True)

    root = snapshot_download(arm["repo"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(arm["sidecar"], tok)
    inj_char = cfg.injection_char

    # ── base (clean) ──────────────────────────────────────────────────────────
    print(f"[load] base {args.base} (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
    layers = resolve_decoder_layers(base)

    # ── stage 1: extract L42 at token t (clean base, one forward per ctx) ──────
    t0 = time.time()
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    h = layers[LAYER].register_forward_hook(grab)
    vecs, norms = np.empty((len(contexts), D), np.float32), []
    try:
        with torch.inference_mode():
            for k, c in enumerate(contexts):
                ids = c["prefix_ids"]
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                v = grabbed["h"][0, -1].float().cpu().numpy()   # last token == t
                vecs[k] = v
                norms.append(float(np.linalg.norm(v)))
    finally:
        h.remove()
    med = float(np.median(norms))
    outlier = [n > 3 * med for n in norms]
    print(f"[extract] {len(contexts)} vecs in {time.time()-t0:.0f}s | "
          f"norm median {med:.1f} | {sum(outlier)} outliers(>3x)", flush=True)

    # ── build actor (arm-specific) ────────────────────────────────────────────
    if arm["merge_sub"]:
        base = PeftModel.from_pretrained(base, f"{root}/{arm['merge_sub']}").merge_and_unload()
    actor = PeftModel.from_pretrained(base, f"{root}/{arm['rl_sub']}").eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)

    # fixed prompt (no per-context variation; only the injected vector changes)
    content = cfg.actor_prompt_template.format(injection_char=inj_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], inj_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + arm["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    print(f"[prompt] len={len(prompt_ids)} tail={ptxt[-40:]!r}", flush=True)

    # ── stage 2: K rollouts of AV explanations, batched T=1 ───────────────────
    t0 = time.time()
    # rollouts[k] -> list over contexts of explanation-text
    per_ctx = [[] for _ in contexts]
    with torch.inference_mode():
        for r in range(args.rollouts):
            torch.manual_seed(args.seed * 100 + r)
            for s in range(0, len(contexts), args.av_batch):
                chunk = vecs[s: s + args.av_batch]
                pt = base_pt.repeat(len(chunk), 1)
                vref[0] = torch.tensor(np.stack(chunk), dtype=torch.float32).cuda()
                try:
                    out = actor.generate(
                        input_ids=pt, attention_mask=torch.ones_like(pt),
                        max_new_tokens=args.max_new, do_sample=True,
                        temperature=1.0, top_p=1.0, top_k=0,
                        pad_token_id=tok.eos_token_id)
                finally:
                    vref[0] = None
                for j, o in enumerate(out):
                    gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                    lines = _lines(gen, cut=arm["stop"])[:args.n_lines]
                    per_ctx[s + j].append(lines)   # keep per-line for budget analysis
            print(f"  rollout {r+1}/{args.rollouts} done ({time.time()-t0:.0f}s)", flush=True)

    # ── save ──────────────────────────────────────────────────────────────────
    cjk_hits = 0
    entries = []
    for k, c in enumerate(contexts):
        expls = per_ctx[k]   # list over rollouts of line-lists
        cjk_hits += sum(bool(CJK.search("\n".join(e))) for e in expls)
        entries.append({"ci": c["ci"], "act_norm": norms[k],
                        "norm_outlier": outlier[k], "explanations": expls})
    payload = {"meta": {"model": args.model, "arm": arm["repo"],
                        "rollouts": args.rollouts, "base": args.base,
                        "n_contexts": len(contexts),
                        "norm_median": med, "n_outliers": int(sum(outlier)),
                        "cjk_explanations": cjk_hits,
                        "gpu": torch.cuda.get_device_name(0),
                        "prompt_tail": ptxt[-40:]},
               "entries": entries}
    Path(args.out).write_text(json.dumps(payload))
    print(f"[write] {args.out} | CJK-tainted explanations: {cjk_hits}/"
          f"{len(contexts)*args.rollouts} (smoke test: want ~0)", flush=True)
    print("sample[0]:", " / ".join(entries[0]["explanations"][0])[:400], flush=True)
    print("SUFFIX_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
