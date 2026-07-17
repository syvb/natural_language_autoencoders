"""GPU-side generation for the CoT-Hints eval (run ON the Vast box).

For each transcript in data/hints_manifest.json:
  1. extract the L42 activation at the LAST token ("Answer: (" of the
     unanswerable Q31) from the CLEAN base — shared by both arms;
  2. BEHAVIORAL manipulation check (base model, no adapters): sample the next
     token 8x at T=1 and record whether the answer letter equals the marked
     option — verifies the base actually learned the marker rule in-context;
  3. generate K AV-explanation rollouts per activation (per-arm protocol,
     identical to the suffix eval).

Usage:  python hints_gen.py --model mat --out results/hints_mat.json
        python hints_gen.py --model std --out results/hints_std.json --skip-behavioral
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
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")

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
    ap.add_argument("--manifest", default=str(HERE / "data" / "hints_manifest.json"))
    ap.add_argument("--base", default=os.environ.get("BASE", "Qwen/Qwen3.6-27B"))
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--av-batch", type=int, default=16)
    ap.add_argument("--behav-samples", type=int, default=8)
    ap.add_argument("--skip-behavioral", action="store_true")
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-lines", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    arm = ARMS[args.model]

    man = json.load(open(args.manifest))
    transcripts = man["transcripts"]
    if args.limit:
        transcripts = transcripts[: args.limit]
    print(f"[manifest] {len(transcripts)} transcripts", flush=True)

    root = snapshot_download(arm["repo"], allow_patterns=[f"{arm['tok_sub']}/*",
                                                          f"{arm['rl_sub']}/*",
                                                          "*.json", "*.yaml"])
    if arm["merge_sub"]:
        snapshot_download(arm["repo"], allow_patterns=[f"{arm['merge_sub']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{arm['tok_sub']}")
    cfg = load_nla_config(arm["sidecar"], tok)

    print(f"[load] base {args.base} (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
    layers = resolve_decoder_layers(base)

    # ── stage 1: extraction at the final token (clean base) ───────────────────
    t0 = time.time()
    grabbed = {}

    def grab(m, i, o):
        grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    all_ids = [tok.encode(t["text"], add_special_tokens=True) for t in transcripts]
    h = layers[LAYER].register_forward_hook(grab)
    vecs, norms = np.empty((len(transcripts), D), np.float32), []
    try:
        with torch.inference_mode():
            for k, ids in enumerate(all_ids):
                base(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                v = grabbed["h"][0, -1].float().cpu().numpy()
                vecs[k] = v
                norms.append(float(np.linalg.norm(v)))
    finally:
        h.remove()
    med = float(np.median(norms))
    print(f"[extract] {len(transcripts)} vecs in {time.time()-t0:.0f}s | "
          f"norm median {med:.1f}", flush=True)

    # ── stage 2: behavioral check on the clean base ───────────────────────────
    behav = {}
    if not args.skip_behavioral:
        t0 = time.time()
        torch.manual_seed(args.seed + 777)
        with torch.inference_mode():
            for k, t in enumerate(transcripts):
                ids = torch.tensor([all_ids[k]], device="cuda")
                out = base.generate(
                    input_ids=ids.repeat(args.behav_samples, 1),
                    attention_mask=torch.ones(args.behav_samples, ids.shape[1],
                                              dtype=torch.long, device="cuda"),
                    max_new_tokens=2, do_sample=True, temperature=1.0,
                    top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
                picks = [tok.decode(o[ids.shape[1]:]).strip()[:1].upper()
                         for o in out]
                behav[t["tid"]] = picks
        print(f"[behavioral] done in {time.time()-t0:.0f}s", flush=True)

    # ── stage 3: AV rollouts ──────────────────────────────────────────────────
    if arm["merge_sub"]:
        base = PeftModel.from_pretrained(base, f"{root}/{arm['merge_sub']}").merge_and_unload()
    actor = PeftModel.from_pretrained(base, f"{root}/{arm['rl_sub']}").eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}],
                             cfg.injection_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + arm["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    print(f"[prompt] len={len(prompt_ids)} tail={ptxt[-40:]!r}", flush=True)

    t0 = time.time()
    per = [[] for _ in transcripts]
    with torch.inference_mode():
        for r in range(args.rollouts):
            torch.manual_seed(args.seed * 100 + r)
            for s in range(0, len(transcripts), args.av_batch):
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
                    per[s + j].append(_lines(gen, cut=arm["stop"])[: args.n_lines])
            print(f"  rollout {r+1}/{args.rollouts} ({time.time()-t0:.0f}s)", flush=True)

    cjk = 0
    entries = []
    for k, t in enumerate(transcripts):
        expls = per[k]
        cjk += sum(bool(CJK.search("\n".join(e))) for e in expls)
        entries.append({"tid": t["tid"], "condition": t["condition"],
                        "final_marked_letter": t["final_marked_letter"],
                        "act_norm": norms[k],
                        "behav_picks": behav.get(t["tid"]),
                        "explanations": expls})
    payload = {"meta": {"model": args.model, "arm": arm["repo"],
                        "rollouts": args.rollouts, "base": args.base,
                        "n_transcripts": len(transcripts), "norm_median": med,
                        "cjk_explanations": cjk,
                        "gpu": torch.cuda.get_device_name(0)},
               "entries": entries}
    Path(args.out).write_text(json.dumps(payload))
    print(f"[write] {args.out} | CJK {cjk}/{len(transcripts)*args.rollouts}", flush=True)
    print("sample:", " / ".join(entries[0]["explanations"][0])[:300], flush=True)
    print("HINTS_GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
