"""Precompute the Space's per-token analysis for every default text → precache.json.

The Space serves default-text clicks from this file instantly (no ZeroGPU call).
Mirrors app.py's matryoshka pipeline EXACTLY (same model, injection, no-prefill
prompt, T=1 sampling, feature-line parse, FVE-vs-mu formula) — but batched:
  1. EXTRACT   one full-text forward per text (causal ⇒ the L42 output at each
               position equals a per-prefix extraction at that position).
  2. VERBALIZE the AV samples one decode per position at T=1, karvonen-injected
               in batches (max_new=256, first N_LINES feature lines kept).
  3. RECONSTRUCT critic scores cumulative line prefixes; FVE vs mu.npy.

Needs ~100GB VRAM (actor + critic resident — an H200). Run on the box:
    python precompute_cache.py --out precache.json
"""
import argparse
import json
import os
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

# ── constants (KEEP IN SYNC with app.py) ─────────────────────────────────────
MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"
RL_SUBDIR = "rl_av_lora_iter400"
TOK_SUBDIR = "warmstart_av_lora"
CRITIC_SUBDIR = "rl_critic_step400"
N_LINES = 10
MAX_TEXT_TOKENS = 2560  # KEEP IN SYNC with app.py + build_default_texts.py
MAX_NEW = 256
LAYER = 42
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
PREFILL = ""
HERE = Path(__file__).resolve().parent
def _normalize(v, scale):
    return v / v.norm().clamp_min(1e-12) * scale


def _lines(text):
    """The model's feature lines: split on newlines, whitespace-strip, drop
    empties — for BOTH the critic and the cached display (the matryoshka emits
    plain lines, no markers). Mirrors app._lines(streaming=False)."""
    return [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", default=str(HERE / "default_texts.json"))
    ap.add_argument("--mu", default=str(HERE / "mu.npy"))
    ap.add_argument("--out", default=str(HERE / "precache.json"))
    ap.add_argument("--av-batch", type=int, default=16)
    ap.add_argument("--ar-batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"

    root = snapshot_download(MODEL_REPO)
    rl_dir, tok_dir = f"{root}/{RL_SUBDIR}", f"{root}/{TOK_SUBDIR}"
    critic_dir = f"{root}/{CRITIC_SUBDIR}"

    tok = AutoTokenizer.from_pretrained(tok_dir)
    cfg = load_nla_config(str(HERE), tok)
    mse_scale = float(cfg.mse_scale)
    critic_tpl = cfg.critic_prompt_template
    mu = torch.tensor(np.load(args.mu), dtype=torch.float32)
    texts = json.load(open(args.texts))
    inj_char = cfg.injection_char

    # fixed prompt (matches app.py) — no prefill
    content = cfg.actor_prompt_template.format(injection_char=inj_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], inj_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + PREFILL
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    print(f"[prompt] len={len(prompt_ids)} tail={ptxt[-30:]!r}", flush=True)

    # ── models (both resident; single H200) ──────────────────────────────────
    print("[load] base + matryoshka rl_av_lora_iter400 (bf16)…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    actor = PeftModel.from_pretrained(base, rl_dir).eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    base_layers = resolve_decoder_layers(actor.get_base_model())

    print("[load] critic rl_critic_step400 (NLACriticModel)…", flush=True)
    critic = NLACriticModel.from_pretrained(
        critic_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()

    # ── tokenize the default texts ────────────────────────────────────────────
    docs = []
    for text in texts:
        ids = tok(text.strip(), add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
        docs.append({"text": text, "ids": ids,
                     "pieces": tok.batch_decode([[t] for t in ids])})
    flat = [(d_i, idx) for d_i, d in enumerate(docs) for idx in range(len(d["ids"]))]
    print(f"{len(docs)} texts, {len(flat)} positions", flush=True)

    # ── stage 1: extract every position — ONE forward per text (causal) ───────
    t0 = time.time()
    vecs = [None] * len(flat)
    flat_pos = {(d_i, idx): n for n, (d_i, idx) in enumerate(flat)}
    with torch.inference_mode():
        for d_i, d in enumerate(docs):
            grabbed = {}

            def grab(m, i, o):
                grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

            h = base_layers[LAYER].register_forward_hook(grab)
            try:
                with actor.disable_adapter():
                    actor(input_ids=torch.tensor([d["ids"]], device="cuda"),
                          use_cache=False)
            finally:
                h.remove()
            hs = grabbed["h"][0].float().cpu().numpy()   # [seq, d] all positions
            for idx in range(len(d["ids"])):
                vecs[flat_pos[(d_i, idx)]] = hs[idx]
    print(f"extract done in {time.time() - t0:.0f}s", flush=True)

    # ── stage 2: AV verbalization, batched T=1 sampling ───────────────────────
    t0 = time.time()
    torch.manual_seed(args.seed)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    all_lines = []   # per-position: the model's feature lines
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
            for o in out:  # inputs=input_ids ⇒ o includes the prompt; slice it
                gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                all_lines.append(_lines(gen)[:N_LINES])
            print(f"  verbalize {min(s + args.av_batch, len(flat))}/{len(flat)}",
                  flush=True)
    print(f"verbalize done in {time.time() - t0:.0f}s", flush=True)
    print(f"[sample] lines[0]={all_lines[0] if all_lines else None}", flush=True)

    # ── stage 3: critic FVE per cumulative line prefix ────────────────────────
    t0 = time.time()
    pad = tok.eos_token_id

    @torch.inference_mode()
    def reconstruct(prefix_texts):
        idlists = [tok.encode(critic_tpl.format(explanation=e),
                              add_special_tokens=False)[:1024] for e in prefix_texts]
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
        return torch.cat(outs, 0)

    # build the full list of cumulative prefixes across all positions, score once
    prefix_texts, spans = [], []
    for lines in all_lines:
        start = len(prefix_texts)
        for k in range(1, len(lines) + 1):
            prefix_texts.append("\n".join(lines[:k]))
        spans.append((start, len(prefix_texts)))
    preds = reconstruct(prefix_texts) if prefix_texts else torch.zeros(0)

    results = []
    for n, (lines, (a, b)) in enumerate(zip(all_lines, spans)):
        if not lines:
            results.append(None)
            continue
        gold_n = _normalize(torch.tensor(vecs[n], dtype=torch.float32), mse_scale)
        denom = ((gold_n - mu) ** 2).mean().item()
        fve, cos = [], []
        for pred in preds[a:b]:
            pred_n = _normalize(pred, mse_scale)
            fve.append(round(1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom, 6))
            cos.append(round(float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm())), 6))
        results.append({"lines": lines, "fve": fve, "cos": cos})
    print(f"score done in {time.time() - t0:.0f}s", flush=True)

    entries, pos = [], 0
    for d in docs:
        n = len(d["ids"])
        entries.append({"text": d["text"], "ids": d["ids"], "pieces": d["pieces"],
                        "results": results[pos: pos + n]})
        pos += n
    payload = {"meta": {"model_repo": MODEL_REPO, "n_lines": N_LINES,
                        "gpu": torch.cuda.get_device_name(0),
                        "generated_unix": int(time.time())},
               "entries": entries}
    Path(args.out).write_text(json.dumps(payload))
    n_ok = sum(r is not None for r in results)
    print(f"wrote {args.out}: {n_ok}/{len(flat)} positions "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
