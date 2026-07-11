"""Generate the WARM-START (pre-RL) matryoshka AV's explanations + FVE, to test
whether the RL is actually helping eval-awareness (and front-loading).

Identical to precompute_cache.py EXCEPT it swaps the two matryoshka adapters
for their warm-start (SFT) counterparts — same base, same bullet-format
sidecar/injection/prompt, same T=1 sampling — so any difference vs the RL
precache is attributable to the RL:
    AV     rl_av_lora_iter400  → warmstart_av_lora
    critic rl_critic_step400   → warmstart_ar_critic

Reuses the RL precache's exact token ids so positions line up 1:1. Adds SOLO
reconstruction (each line alone) alongside the cumulative prefixes, so the
buried-lede (first-line-alone FVE) can be compared too.

Both models resident on one H200 (≥110GB) — single process, no reload.
    python warmstart_gen.py --rl-precache space/precache.json --out warmstart_precache.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"
AV_SUBDIR = "warmstart_av_lora"       # pre-RL AV (was rl_av_lora_iter400)
TOK_SUBDIR = "warmstart_av_lora"
CRITIC_SUBDIR = "warmstart_ar_critic"  # pre-RL critic (was rl_critic_step400)
N_LINES = 10
MAX_NEW = 256
LAYER = 42
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
PREFILL = ""
HERE = Path(__file__).resolve().parent


def _normalize(v, scale):
    return v / v.norm().clamp_min(1e-12) * scale


def _lines(text):
    return [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rl-precache", default=str(HERE / "space" / "precache.json"))
    ap.add_argument("--mu", default=str(HERE / "space" / "mu.npy"))
    ap.add_argument("--out", default=str(HERE / "warmstart_precache.json"))
    ap.add_argument("--av-batch", type=int, default=16)
    ap.add_argument("--ar-batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"

    root = snapshot_download(MODEL_REPO)
    av_dir, tok_dir = f"{root}/{AV_SUBDIR}", f"{root}/{TOK_SUBDIR}"
    critic_dir = f"{root}/{CRITIC_SUBDIR}"

    tok = AutoTokenizer.from_pretrained(tok_dir)
    cfg = load_nla_config(str(HERE / "space"), tok)  # AV sidecar (prompt/injection)
    # critic template + scale from the WARM-START critic's OWN sidecar
    cmeta = yaml.safe_load(open(f"{critic_dir}/nla_meta.yaml"))
    critic_tpl = (cmeta.get("critic") or {}).get("critic_prompt_template") \
        or cmeta.get("critic_prompt_template") or cfg.critic_prompt_template
    mse_scale = float((cmeta.get("critic") or {}).get("mse_scale")
                      or cmeta.get("mse_scale") or cfg.mse_scale)
    assert "{explanation}" in critic_tpl, "warmstart critic template missing {explanation}"
    print(f"[critic] mse_scale={mse_scale:.3f} tpl_tail={critic_tpl[-40:]!r}", flush=True)
    mu = torch.tensor(np.load(args.mu), dtype=torch.float32)
    inj_char = cfg.injection_char

    content = cfg.actor_prompt_template.format(injection_char=inj_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], inj_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + PREFILL
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    print(f"[prompt] len={len(prompt_ids)}", flush=True)

    print("[load] base + warmstart_av_lora (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    actor = PeftModel.from_pretrained(base, av_dir).eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    base_layers = resolve_decoder_layers(actor.get_base_model())

    print("[load] warmstart_ar_critic (NLACriticModel)…", flush=True)
    critic = NLACriticModel.from_pretrained(
        critic_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()

    # reuse the RL precache's EXACT ids/pieces → identical positions
    rl = json.load(open(args.rl_precache))["entries"]
    docs = [{"text": e["text"], "ids": e["ids"], "pieces": e["pieces"]} for e in rl]
    flat = [(d_i, idx) for d_i, d in enumerate(docs) for idx in range(len(d["ids"]))]
    flat_pos = {(d_i, idx): n for n, (d_i, idx) in enumerate(flat)}
    print(f"{len(docs)} texts, {len(flat)} positions", flush=True)

    # stage 1: extract (raw base, adapter disabled) — one forward per text
    t0 = time.time()
    vecs = [None] * len(flat)
    with torch.inference_mode():
        for d_i, d in enumerate(docs):
            grabbed = {}

            def grab(m, i, o):
                grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

            h = base_layers[LAYER].register_forward_hook(grab)
            try:
                with actor.disable_adapter():
                    actor(input_ids=torch.tensor([d["ids"]], device="cuda"), use_cache=False)
            finally:
                h.remove()
            hs = grabbed["h"][0].float().cpu().numpy()
            for idx in range(len(d["ids"])):
                vecs[flat_pos[(d_i, idx)]] = hs[idx]
    print(f"extract done in {time.time() - t0:.0f}s", flush=True)

    # stage 2: warm-start AV verbalization (batched T=1)
    t0 = time.time()
    torch.manual_seed(args.seed)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    all_lines = []
    with torch.inference_mode():
        for s in range(0, len(flat), args.av_batch):
            chunk = vecs[s: s + args.av_batch]
            pt = base_pt.repeat(len(chunk), 1)
            vref[0] = torch.tensor(np.stack(chunk), dtype=torch.float32).cuda()
            try:
                out = actor.generate(
                    input_ids=pt, attention_mask=torch.ones_like(pt),
                    max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                    top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
            finally:
                vref[0] = None
            for o in out:
                gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                all_lines.append(_lines(gen)[:N_LINES])
            if (s // args.av_batch) % 20 == 0:
                print(f"  verbalize {min(s + args.av_batch, len(flat))}/{len(flat)}", flush=True)
    print(f"verbalize done in {time.time() - t0:.0f}s; sample={all_lines[0]}", flush=True)

    # stage 3: critic — cumulative prefixes (fve) AND each line solo
    t0 = time.time()
    pad = tok.eos_token_id

    @torch.inference_mode()
    def reconstruct(texts):
        idlists = [tok.encode(critic_tpl.format(explanation=e),
                              add_special_tokens=False)[:1024] for e in texts]
        outs = []
        for i in range(0, len(idlists), args.ar_batch):
            ck = idlists[i: i + args.ar_batch]
            m = max(len(x) for x in ck)
            bx = torch.full((len(ck), m), pad, dtype=torch.long, device="cuda")
            at = torch.zeros((len(ck), m), dtype=torch.long, device="cuda")
            for r, q in enumerate(ck):
                bx[r, :len(q)] = torch.tensor(q, dtype=torch.long)
                at[r, :len(q)] = 1
            outs.append(critic_predict(critic, bx, at, mse_scale).float().cpu())
        return torch.cat(outs, 0) if outs else torch.zeros(0)

    texts, spans = [], []
    for lines in all_lines:
        start = len(texts)
        for k in range(1, len(lines) + 1):          # cumulative prefixes
            texts.append("\n".join(lines[:k]))
        for ln in lines:                             # solo lines
            texts.append(ln)
        spans.append((start, len(lines)))
    preds = reconstruct(texts)

    results = []
    for n, (lines, (start, nl)) in enumerate(zip(all_lines, spans)):
        if not lines:
            results.append(None)
            continue
        gold_n = _normalize(torch.tensor(vecs[n], dtype=torch.float32), mse_scale)
        denom = ((gold_n - mu) ** 2).mean().item()
        fve, cos, solo = [], [], []
        for j in range(nl):                          # cumulative
            pn = _normalize(preds[start + j], mse_scale)
            fve.append(round(1.0 - ((pn - gold_n) ** 2).mean().item() / denom, 6))
            cos.append(round(float(pn @ gold_n / (pn.norm() * gold_n.norm())), 6))
        for j in range(nl):                          # solo
            pn = _normalize(preds[start + nl + j], mse_scale)
            solo.append(round(1.0 - ((pn - gold_n) ** 2).mean().item() / denom, 6))
        results.append({"lines": lines, "fve": fve, "cos": cos, "solo": solo})
    print(f"score done in {time.time() - t0:.0f}s", flush=True)

    entries, pos = [], 0
    for d in docs:
        n = len(d["ids"])
        entries.append({"text": d["text"], "ids": d["ids"], "pieces": d["pieces"],
                        "results": results[pos: pos + n]})
        pos += n
    payload = {"meta": {"model_repo": MODEL_REPO, "av": AV_SUBDIR, "critic": CRITIC_SUBDIR,
                        "n_lines": N_LINES, "gpu": torch.cuda.get_device_name(0),
                        "generated_unix": int(time.time())},
               "entries": entries}
    Path(args.out).write_text(json.dumps(payload))
    n_ok = sum(r is not None for r in results)
    print(f"wrote {args.out}: {n_ok}/{len(flat)} positions "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
