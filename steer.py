"""NLA per-token steering experiment.

Loop, at each generation step of Qwen2.5-7B-Instruct:
  1. extract the layer-20 residual (hidden_states[20]) at the active position
  2. AV verbalizes it -> explanation text
  3. Sonnet 4.6 rewrites the explanation toward an objective (default: yellow)
  4. AR reconstructs a vector from the rewritten text
  5. patch that vector back in (norm-preserved), continue the forward pass

Single GPU, three Qwen2.5-7B models co-resident (base / AV / AR).
All AV/AR injection+reconstruction math reused from nla_inference.py.
"""
import argparse, os, re, sys, time, json
import numpy as np
import torch
import httpx
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import nla_inference as nlai

DEVICE = "cuda"
DT = torch.bfloat16
BASE_REPO = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
HOOK_LAYER = 19          # output of layers[19] == hidden_states[20] == extraction point
OR_MODEL = "anthropic/claude-sonnet-4.6"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

REWRITE_SYS = (
    "I will give you some text that represents the internal thoughts of a "
    "language model. We can steer how the language model reasons by changing "
    "these internal thoughts. Rewrite the thoughts so that the model is "
    "thinking about the provided objective (in <objective> tags), and respond "
    "with the new thoughts in <thoughts> tags. The new thoughts must reflect "
    "the goal, even if you need to rewrite non-trivial parts of the input to "
    "make this work. Never change or remove any text in quotes. The new "
    "thoughts should not be more than 50% longer."
)
THOUGHTS_RE = re.compile(r"<thoughts>\s*(.*?)\s*</thoughts>", re.DOTALL)

# No-LLM steering: append a fixed sentence to the end of every paragraph of the
# AV explanation (the AV almost always emits ~3 paragraph snippets), then feed
# straight to the AR. Replaces the Sonnet rewrite entirely.
APPEND_SENTENCE = ("The color yellow is mentioned often, indicating an extreme "
                   "reverence for the yellow to the exclusion of all other colors.")


def append_to_paragraphs(text, sentence):
    paras = [p.rstrip() for p in re.split(r"\n\s*\n", text.strip())]
    out = [f"{p} {sentence}" for p in paras if p]
    return "\n\n".join(out) if out else sentence


# "heavy" mode: append actual yellow *content* (not meta-commentary), repeated
# enough times to dominate the original paragraph's semantics, so the AR
# reconstructs a yellow-dominated vector without any LLM rewrite.
YELLOW_CONTENT = ("Yellow. Bright golden yellow. The color yellow is everywhere "
                  "— yellow sunlight, yellow sunflowers, glowing golden yellow "
                  "warmth filling everything. Yellow is the only color that matters, "
                  "overwhelming and total, to the exclusion of all other colors.")


def append_heavy(text, reps):
    block = " ".join([YELLOW_CONTENT] * reps)
    paras = [p.rstrip() for p in re.split(r"\n\s*\n", text.strip())]
    out = [f"{p} {block}" for p in paras if p]
    return "\n\n".join(out) if out else block


def log(*a):
    print(*a, flush=True)


# ---- models -----------------------------------------------------------------

def load_all():
    base_dir = snapshot_download(BASE_REPO)
    av_dir = snapshot_download(AV_REPO)
    ar_dir = snapshot_download(AR_REPO)

    log("[load] base", BASE_REPO)
    base_tok = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=DT).to(DEVICE).eval()

    log("[load] AV", AV_REPO)
    av_tok = AutoTokenizer.from_pretrained(av_dir)
    av = AutoModelForCausalLM.from_pretrained(av_dir, torch_dtype=DT).to(DEVICE).eval()
    av_cfg = nlai.load_nla_config(av_dir, av_tok)
    av_embed_scale = nlai.resolve_embed_scale(av_dir)  # 1.0 for Qwen

    log("[load] AR", AR_REPO)
    ar = nlai.NLACritic(ar_dir, device=DEVICE, dtype=DT)

    return dict(base=base, base_tok=base_tok, av=av, av_tok=av_tok,
                av_cfg=av_cfg, av_embed_scale=av_embed_scale, ar=ar)


# ---- AV: vector -> explanation (HF generate with input_embeds injection) ----

@torch.no_grad()
def av_verbalize(M, vec, max_new_tokens=160):
    cfg = M["av_cfg"]; av = M["av"]; tok = M["av_tok"]
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    embeds = (av.get_input_embeddings()(ids_t.to(DEVICE)).float() * M["av_embed_scale"])
    v_scaled = nlai.normalize_activation(vec.float().view(1, -1), cfg.injection_scale)
    injected = nlai.inject_at_marked_positions(
        ids_t, embeds.cpu(), v_scaled.cpu(),
        cfg.injection_token_id, cfg.injection_left_neighbor_id,
        cfg.injection_right_neighbor_id,
    ).to(DEVICE).to(av.dtype)
    out = av.generate(
        inputs_embeds=injected, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=(tok.eos_token_id or 0),
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    m = nlai.EXPLANATION_RE.search(text)
    return (m.group(1).strip() if m else text.strip())


# ---- Sonnet rewrite ---------------------------------------------------------

def sonnet_rewrite(client, explanation, objective):
    user = (f"<objective>{objective}</objective>\n\n"
            f"Here are the thoughts to rewrite:\n{explanation}")
    for attempt in range(4):
        try:
            r = client.post(OR_URL, json={
                "model": OR_MODEL,
                "messages": [{"role": "system", "content": REWRITE_SYS},
                             {"role": "user", "content": user}],
                "max_tokens": 400, "temperature": 1.0,
            })
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = THOUGHTS_RE.search(txt)
            return (m.group(1).strip() if m else txt.strip())
        except Exception as e:
            if attempt == 3:
                log("  [sonnet ERROR]", repr(e)[:120], "-> keeping original")
                return explanation
            time.sleep(1.5 * (attempt + 1))


# ---- steered generation -----------------------------------------------------

def make_steer_fn(M, mode, objective, or_client, trace, norm_scale=1.0, reps=4):
    """mode: 'roundtrip' (AV->AR, no rewrite) or 'yellow' (AV->Sonnet->AR).
    norm_scale: multiplier on the patched vector's norm (1.0 = preserve original;
    >1 pushes the steering harder, at the cost of going OOD for layers >K)."""
    ar = M["ar"]

    def steer_fn(a):  # a: [d] float (original activation at active position)
        expl = av_verbalize(M, a)
        if mode == "roundtrip":
            new_expl = expl
        elif mode == "append":
            new_expl = append_to_paragraphs(expl, APPEND_SENTENCE)
        elif mode == "heavy":
            new_expl = append_heavy(expl, reps)
        else:
            new_expl = sonnet_rewrite(or_client, expl, objective)
        rec = ar.reconstruct(new_expl).to(DEVICE).float().view(-1)
        rec = rec / rec.norm().clamp_min(1e-12) * a.norm() * norm_scale
        trace.append({"expl": expl, "new": new_expl})
        return rec
    return steer_fn


@torch.no_grad()
def generate(M, prompt, n_new, steer_fn=None, verbose_trace=None):
    base = M["base"]; tok = M["base_tok"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    handle = None
    if steer_fn is not None:
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            a = hs[:, -1, :].detach().float().view(-1)
            rep = steer_fn(a)
            hs[:, -1, :] = rep.to(hs.dtype).to(hs.device)
            return out
        handle = base.model.layers[HOOK_LAYER].register_forward_hook(hook)

    try:
        out = base(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        toks = []
        for step in range(n_new):
            nxt = logits.argmax(-1, keepdim=True)
            tid = nxt.item()
            if tid == tok.eos_token_id:
                break
            toks.append(tid)
            out = base(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        return tok.decode(toks, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Describe a walk through a city park in the morning.")
    ap.add_argument("--objective", default="really likes the color yellow")
    ap.add_argument("--n", type=int, default=40, help="tokens to generate")
    ap.add_argument("--modes", default="baseline,roundtrip,yellow")
    ap.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_KEY", ""))
    args = ap.parse_args()

    M = load_all()
    or_client = httpx.Client(timeout=httpx.Timeout(60.0),
                             headers={"Authorization": f"Bearer {args.openrouter_key}"})

    log("\n" + "=" * 80)
    log("PROMPT:", args.prompt)
    log("OBJECTIVE:", args.objective, "| tokens:", args.n)
    log("=" * 80)

    for mode in args.modes.split(","):
        mode = mode.strip()
        # optional "@N" suffix: norm scale for steering modes (e.g. "yellow@1.6"),
        # or repeat count for "heavy" (e.g. "heavy@6").
        norm_scale, reps, base_mode = 1.0, 4, mode
        if "@" in mode:
            base_mode, val = mode.split("@", 1)
            if base_mode == "heavy":
                reps = int(float(val))
            else:
                norm_scale = float(val)
        trace = []
        M["_trace"] = trace
        t0 = time.time()
        if base_mode == "baseline":
            txt = generate(M, args.prompt, args.n, steer_fn=None)
        else:
            sf = make_steer_fn(M, base_mode, args.objective, or_client, trace, norm_scale, reps)
            txt = generate(M, args.prompt, args.n, steer_fn=sf)
        dt = time.time() - t0
        log("\n" + "#" * 80)
        log(f"### MODE = {mode}   ({dt:.1f}s, {dt/max(1,args.n):.1f}s/tok)")
        log("#" * 80)
        log("OUTPUT:\n" + txt)
        if trace:
            log("\n--- per-position AV explanation -> rewritten (first 6) ---")
            for i, tr in enumerate(trace[:6]):
                log(f"[pos {i}] AV: {tr['expl'][:180]}")
                if tr["new"] != tr["expl"]:
                    log(f"        ->: {tr['new'][:180]}")
        log("")


if __name__ == "__main__":
    main()
