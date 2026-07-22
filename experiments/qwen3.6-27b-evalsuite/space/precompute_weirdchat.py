"""Cost-optimized precompute of the Space's per-token analysis for a set of texts.

Same pipeline + output format as precompute_cache.py (mirrors app.py's matryoshka
path EXACTLY), with three changes that cut GPU cost so this runs on a single
80GB H100 (~$0.5/hr) overnight instead of an H200:

  1. STAGED LOAD — the pipeline is extract→verbalize (actor) then reconstruct
     (critic). Activations are already moved to CPU after extraction, so we free
     the actor before loading the critic → only ONE 27B is resident at a time,
     which fits 80GB (vs ~100GB for both). ~7x cheaper GPU/hr.
  2. EARLY STOP — batched, per-sequence _StopAfterLines (ported from app.py):
     decoding halts each sample once it has N_LINES complete feature lines, so we
     don't burn the full 256-token budget (the matryoshka rarely EOSes but its
     lines are short → ~40% fewer decoded tokens, identical kept output).
  3. RESUMABLE — after the expensive verbalize stage the lines+vecs are
     checkpointed; a crash/restart skips straight to reconstruct. The final
     write is atomic.

Run on the box (needs ~/.hf_token; the qwen3_5 fla env per the box recipe):
    python precompute_weirdchat.py --texts weirdchat_texts.json --out precache_weirdchat.json
"""
import argparse
import gc
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
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList)

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

# ── constants (KEEP IN SYNC with app.py / precompute_cache.py) ───────────────
MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"
RL_SUBDIR = "rl_av_lora_iter400"
TOK_SUBDIR = "warmstart_av_lora"
CRITIC_SUBDIR = "rl_critic_step400"
N_LINES = 10
MAX_TEXT_TOKENS = 5120
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


def _lines_streaming(text):
    # count only COMPLETE (newline-terminated) lines — drop the trailing partial
    return [ln.strip() for ln in text.split("\n")[:-1] if ln.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", default=str(HERE / "weirdchat_texts.json"))
    ap.add_argument("--mu", default=str(HERE / "mu.npy"))
    ap.add_argument("--out", default=str(HERE / "precache_weirdchat.json"))
    ap.add_argument("--av-batch", type=int, default=12)
    ap.add_argument("--ar-batch", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    ckpt = args.out + ".stage2.npz"

    root = snapshot_download(
        MODEL_REPO,
        allow_patterns=[f"{RL_SUBDIR}/*", f"{TOK_SUBDIR}/*", f"{CRITIC_SUBDIR}/*"],
    )
    rl_dir, tok_dir = f"{root}/{RL_SUBDIR}", f"{root}/{TOK_SUBDIR}"
    critic_dir = f"{root}/{CRITIC_SUBDIR}"

    tok = AutoTokenizer.from_pretrained(tok_dir)
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
    prompt_len = len(prompt_ids)
    print(f"[prompt] len={prompt_len}", flush=True)

    # tokenize input texts (once; needed for output + resume key)
    docs = []
    for text in texts:
        ids = tok(text.strip(), add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
        docs.append({"text": text, "ids": ids,
                     "pieces": tok.batch_decode([[t] for t in ids])})
    flat = [(d_i, idx) for d_i, d in enumerate(docs) for idx in range(len(d["ids"]))]
    print(f"{len(docs)} texts, {len(flat)} positions", flush=True)

    class _StopAfterLines(StoppingCriteria):
        """Per-sequence stop once a row has N_LINES complete lines. Returns a
        BoolTensor[batch] (transformers>=4.4x per-row semantics) so early rows
        stop while the rest keep going — real per-sample early-exit, not
        batch-gated."""

        def __init__(self, prompt_len, batch):
            self.prompt_len = prompt_len
            self.batch = batch

        def __call__(self, input_ids, scores, **kw):
            gens = tok.batch_decode(input_ids[:, self.prompt_len:],
                                    skip_special_tokens=True)
            return torch.tensor([len(_lines_streaming(g)) >= N_LINES for g in gens],
                                device=input_ids.device)

    # ── resume: if stage-2 checkpoint exists, skip extract+verbalize ──────────
    if os.path.exists(ckpt):
        print(f"[resume] loading stage-2 checkpoint {ckpt}", flush=True)
        z = np.load(ckpt, allow_pickle=True)
        vecs = list(z["vecs"])
        all_lines = list(z["all_lines"])
    else:
        # ── load ACTOR (base + matryoshka LoRA) ───────────────────────────────
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

        # stage 1: extract L42 output at every position (one forward per text)
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

        # stage 2: AV verbalize, batched T=1, EARLY-STOP after N_LINES
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
                        stopping_criteria=StoppingCriteriaList(
                            [_StopAfterLines(pt.shape[1], len(chunk))]))
                finally:
                    vref[0] = None
                for o in out:
                    gen = tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                    all_lines.append(_lines(gen)[:N_LINES])
                if (s // args.av_batch) % 10 == 0:
                    print(f"  verbalize {min(s + args.av_batch, len(flat))}/{len(flat)} "
                          f"({time.time() - t0:.0f}s)", flush=True)
        print(f"verbalize done in {time.time() - t0:.0f}s", flush=True)

        # checkpoint the expensive stage so a stage-3 crash doesn't redo it
        np.savez(ckpt, vecs=np.array(vecs, dtype=object),
                 all_lines=np.array(all_lines, dtype=object))
        print(f"[ckpt] wrote {ckpt}", flush=True)

        del actor, base
        gc.collect()
        torch.cuda.empty_cache()

    # ── load CRITIC (actor freed) ─────────────────────────────────────────────
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

    # stage 3: critic FVE per cumulative line prefix
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
