"""Cost-optimized precompute of the STANDARD (L42, <explanation>-format) Space's
per-token analysis for a set of texts. Same pipeline + output format as
space_std/precompute_cache.py (mirrors space_std/app.py EXACTLY), with the same
three cost changes as the matryoshka precompute_weirdchat.py:

  1. STAGED LOAD — free the actor (base + SFT + RL adapters) before loading the
     critic so only ONE 27B is resident → fits an 80GB H100 (~$1/hr) instead of
     needing an H200. Activations are already on CPU after extraction.
  2. STOP AT </explanation> — the standard model reliably closes the tag and
     EOSes, so generation is naturally short; `stop_strings=[STOP_STR]` bounds
     the rare rambler per-sequence (built-in, efficient) — no full max_new.
  3. RESUMABLE — checkpoint lines+vecs (stacked float32) after the expensive
     verbalize stage; a crash/restart skips to reconstruct. Atomic final write.

Run:  python precompute_weirdchat.py --texts weirdchat_texts.json --out precache_weirdchat.json
"""
import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

# ── constants (KEEP IN SYNC with space_std/app.py / precompute_cache.py) ─────
MODEL_REPO = "ceselder/qwen3.6-27b-nla-L42"
BASE_ID = "Qwen/Qwen3.6-27B"
SFT_SUBDIR = "av_sft_lora"           # warmstart LoRA (also carries the tokenizer)
RL_SUBDIR = "av_rl_lora_step400"
CRITIC_SUBDIR = "rl_critic_step400"
N_LINES = 10
MAX_TEXT_TOKENS = 5120
MAX_NEW = 256
LAYER = 42
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
PREFILL = "<explanation>\n"
STOP_STR = "</explanation>"
HERE = Path(__file__).resolve().parent


def _normalize(v, scale):
    return v / v.norm().clamp_min(1e-12) * scale


def _lines(text):
    """Body of <explanation>…</explanation> as non-empty lines (the 2-3 snippets).
    The generation is decoded WITHOUT the prompt (which carries the PREFILL tag),
    so it has no opening <explanation>; split defensively then cut at </explanation>.
    Mirrors space_std/app._lines(streaming=False)."""
    after = text.split("<explanation>", 1)[-1].split(STOP_STR, 1)[0]
    return [ln.strip() for ln in after.split("\n") if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", default=str(HERE / "weirdchat_texts.json"))
    ap.add_argument("--mu", default=str(HERE / "mu.npy"))
    ap.add_argument("--out", default=str(HERE / "precache_weirdchat.json"))
    ap.add_argument("--av-batch", type=int, default=24)
    ap.add_argument("--ar-batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    ckpt = args.out + ".stage2.npz"

    root = snapshot_download(
        MODEL_REPO,
        allow_patterns=[f"{SFT_SUBDIR}/*", f"{RL_SUBDIR}/*", f"{CRITIC_SUBDIR}/*"],
    )
    sft_dir, rl_dir = f"{root}/{SFT_SUBDIR}", f"{root}/{RL_SUBDIR}"
    critic_dir = f"{root}/{CRITIC_SUBDIR}"

    tok = AutoTokenizer.from_pretrained(sft_dir)
    cfg = load_nla_config(str(HERE), tok)
    mse_scale = float(cfg.mse_scale)
    critic_tpl = cfg.critic_prompt_template
    mu = torch.tensor(np.load(args.mu), dtype=torch.float32)
    texts = json.load(open(args.texts))
    inj_char = cfg.injection_char

    content = cfg.actor_prompt_template.format(injection_char=inj_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], inj_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + PREFILL
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    print(f"[prompt] len={len(prompt_ids)} tail={ptxt[-30:]!r}", flush=True)

    docs = []
    for text in texts:
        ids = tok(text.strip(), add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
        docs.append({"text": text, "ids": ids,
                     "pieces": tok.batch_decode([[t] for t in ids])})
    flat = [(d_i, idx) for d_i, d in enumerate(docs) for idx in range(len(d["ids"]))]
    print(f"{len(docs)} texts, {len(flat)} positions", flush=True)

    if os.path.exists(ckpt):
        print(f"[resume] loading stage-2 checkpoint {ckpt}", flush=True)
        z = np.load(ckpt, allow_pickle=True)
        vecs = [np.asarray(v, dtype=np.float32) for v in z["vecs"]]
        all_lines = [list(x) for x in z["all_lines"]]
    else:
        print("[load] base + av_sft_lora + av_rl_lora_step400 (bf16)…", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map={"": 0})
        actor = PeftModel.from_pretrained(base, sft_dir, adapter_name="sft")
        actor.load_adapter(rl_dir, adapter_name="rl")
        actor.base_model.set_adapter(["sft", "rl"])
        actor = actor.eval()
        vref = [None]
        register_karvonen_hook(actor, vref, cfg.injection_token_id,
                               cfg.injection_left_neighbor_id,
                               cfg.injection_right_neighbor_id, layer_idx=1)
        base_layers = resolve_decoder_layers(actor.get_base_model())

        # stage 1: extract L42 output at every position (raw base, adapters off)
        t0 = time.time()
        vecs = [None] * len(flat)
        fpos = {(d_i, idx): n for n, (d_i, idx) in enumerate(flat)}
        with torch.inference_mode():
            for d_i, d in enumerate(docs):
                grabbed = {}
                h = base_layers[LAYER].register_forward_hook(
                    lambda m, i, o: grabbed.__setitem__(
                        "h", (o[0] if isinstance(o, tuple) else o).detach()))
                try:
                    with actor.disable_adapter():
                        actor(input_ids=torch.tensor([d["ids"]], device="cuda"),
                              use_cache=False)
                finally:
                    h.remove()
                hs = grabbed["h"][0].float().cpu().numpy()
                for idx in range(len(d["ids"])):
                    vecs[fpos[(d_i, idx)]] = hs[idx]
        print(f"extract done in {time.time() - t0:.0f}s", flush=True)

        # stage 2: AV verbalize, batched T=1, STOP at </explanation> per-sequence
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
                        top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id,
                        stop_strings=[STOP_STR], tokenizer=tok)
                finally:
                    vref[0] = None
                for o in out:
                    gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                    all_lines.append(_lines(gen)[:N_LINES])
                if (s // args.av_batch) % 10 == 0:
                    print(f"  verbalize {min(s + args.av_batch, len(flat))}/{len(flat)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        print(f"verbalize done in {time.time() - t0:.0f}s", flush=True)
        print(f"[sample] lines[0]={all_lines[0] if all_lines else None}", flush=True)

        np.savez(ckpt, vecs=np.stack(vecs).astype(np.float32),
                 all_lines=np.array(all_lines, dtype=object))
        print(f"[ckpt] wrote {ckpt}", flush=True)

        # free the actor (base + adapters); base_layers/hooks/vref hold refs to it
        vref[0] = None
        del actor, base, base_layers, vref
        gc.collect()
        torch.cuda.empty_cache()

    print("[load] critic rl_critic_step400 (NLACriticModel)…", flush=True)
    critic = NLACriticModel.from_pretrained(
        critic_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()
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

    t0 = time.time()
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
        gold_n = _normalize(torch.tensor(np.asarray(vecs[n], dtype=np.float32)), mse_scale)
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
    tmp = args.out + ".tmp"
    Path(tmp).write_text(json.dumps(payload))
    os.replace(tmp, args.out)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    n_ok = sum(r is not None for r in results)
    print(f"wrote {args.out}: {n_ok}/{len(flat)} positions "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)", flush=True)
    print("PRECOMPUTE_DONE", flush=True)


if __name__ == "__main__":
    main()
