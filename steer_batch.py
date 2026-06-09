"""Batched lock-step NLA steering — generate B samples in parallel.

Same per-token loop as steer.py (AV verbalize -> Sonnet rewrite -> AR
reconstruct -> norm-preserved patch), but processes a whole batch of B samples
in lock-step so the expensive parts amortize:
  - ONE batched AV generate for all B activations (identical prompt, one vector
    injected per row -> no padding)
  - all B Sonnet rewrites fired CONCURRENTLY (thread pool)
  - AR reconstruct + base-model step run on the [B, d] batch at once

Divergence across the B samples comes from base-model sampling + the stochastic
(temperature 1.0) Sonnet rewrites, so identical-prompt samples spread out.

Reuses steer.py for model loading and the Sonnet call.
"""
import argparse, os, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import httpx

import nla_inference as nlai
import steer as S

DEVICE = "cuda"
HOOK_LAYER = S.HOOK_LAYER


def log(*a):
    print(*a, flush=True)


# ---- batched AV: [B,d] activations -> B explanation strings ------------------

@torch.no_grad()
def av_verbalize_batch(M, A, max_new_tokens=160):
    cfg = M["av_cfg"]; av = M["av"]; tok = M["av_tok"]
    B = A.shape[0]
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    T = len(input_ids)
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).expand(B, -1)  # [B,T]
    embeds = (av.get_input_embeddings()(ids_t.to(DEVICE)).float() * M["av_embed_scale"])
    # one injection site per row, row-major iteration assigns vectors[b] -> row b
    v_scaled = nlai.normalize_activation(A.float(), cfg.injection_scale)  # [B,d]
    injected = nlai.inject_at_marked_positions(
        ids_t, embeds.cpu(), v_scaled.cpu(),
        cfg.injection_token_id, cfg.injection_left_neighbor_id,
        cfg.injection_right_neighbor_id,
    ).to(DEVICE).to(av.dtype)
    attn = torch.ones(B, T, dtype=torch.long, device=DEVICE)
    out = av.generate(
        inputs_embeds=injected, attention_mask=attn,
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=(tok.eos_token_id or 0),
    )
    texts = tok.batch_decode(out, skip_special_tokens=True)
    expls = []
    for t in texts:
        m = nlai.EXPLANATION_RE.search(t)
        expls.append(m.group(1).strip() if m else t.strip())
    return expls


# ---- batched AR: B texts -> [B,d] (loop; AR forward is cheap) ----------------

@torch.no_grad()
def ar_reconstruct_batch(M, texts):
    recs = [M["ar"].reconstruct(t).float().view(-1) for t in texts]
    return torch.stack(recs, 0).to(DEVICE)  # [B,d]


# ---- concurrent Sonnet rewrites ---------------------------------------------

def sonnet_rewrite_batch(client, pool, expls, objective):
    return list(pool.map(lambda e: S.sonnet_rewrite(client, e, objective), expls))


# ---- sampling ----------------------------------------------------------------

def sample_next(logits, temperature, top_p):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, si = torch.sort(probs, descending=True, dim=-1)
        cum = sp.cumsum(-1)
        sp[(cum - sp) > top_p] = 0.0
        sp = sp / sp.sum(-1, keepdim=True)
        choice = torch.multinomial(sp, 1)
        return si.gather(-1, choice)
    return torch.multinomial(probs, 1)


# ---- batched steered generation ---------------------------------------------

@torch.no_grad()
def generate_batch(M, prompt, B, n_new, steer_fn_batch, temperature, top_p):
    base = M["base"]; tok = M["base_tok"]
    ids0 = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, return_tensors="pt")
    ids = ids0.expand(B, -1).to(DEVICE)  # [B,T0]

    def hook(mod, inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        A = hs[:, -1, :].detach().float()       # [B,d]
        rep = steer_fn_batch(A)                  # [B,d]
        hs[:, -1, :] = rep.to(hs.dtype).to(hs.device)
        return out
    handle = base.model.layers[HOOK_LAYER].register_forward_hook(hook)

    gen = [[] for _ in range(B)]
    try:
        attn = torch.ones_like(ids)
        out = base(input_ids=ids, attention_mask=attn, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        ones = torch.ones(B, 1, dtype=attn.dtype, device=DEVICE)
        for step in range(n_new):
            nxt = sample_next(logits, temperature, top_p)   # [B,1]
            for b in range(B):
                gen[b].append(nxt[b, 0].item())
            attn = torch.cat([attn, ones], dim=1)
            out = base(input_ids=nxt, attention_mask=attn,
                       past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            log(f"  [step {step+1}/{n_new}]")
    finally:
        handle.remove()

    eos = tok.eos_token_id
    texts = []
    for g in gen:
        cut = g.index(eos) if eos in g else len(g)
        texts.append(tok.decode(g[:cut], skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Explain how a bicycle works.")
    ap.add_argument("--objective",
                    default="extreme over-the-top love for the color yellow, "
                            "to the exclusion of all other colors")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_KEY", ""))
    ap.add_argument("--out", default="/root/samples.txt")
    args = ap.parse_args()

    M = S.load_all()
    client = httpx.Client(timeout=httpx.Timeout(60.0),
                          headers={"Authorization": f"Bearer {args.openrouter_key}"})
    pool = ThreadPoolExecutor(max_workers=max(4, args.samples))

    trace = {"n": 0}

    def steer_fn_batch(A):  # [B,d] -> [B,d]
        expls = av_verbalize_batch(M, A)
        news = sonnet_rewrite_batch(client, pool, expls, args.objective)
        recs = ar_reconstruct_batch(M, news)
        recs = recs / recs.norm(dim=-1, keepdim=True).clamp_min(1e-12) \
               * A.norm(dim=-1, keepdim=True)
        if trace["n"] < 3:  # log first few steps, sample 0
            log(f"    [trace step {trace['n']}] AV[0]: {expls[0][:140]}")
            log(f"                       ->[0]: {news[0][:140]}")
        trace["n"] += 1
        return recs

    log("=" * 80)
    log(f"PROMPT: {args.prompt}")
    log(f"OBJECTIVE: {args.objective}")
    log(f"samples={args.samples} tokens={args.n} temp={args.temperature} top_p={args.top_p}")
    log("=" * 80)

    t0 = time.time()
    texts = generate_batch(M, args.prompt, args.samples, args.n,
                           steer_fn_batch, args.temperature, args.top_p)
    dt = time.time() - t0
    log(f"\n### DONE {args.samples} samples x {args.n} tok in {dt:.1f}s "
        f"({dt/args.n:.1f}s/step, {dt/(args.samples*args.n):.2f}s/sample-tok)")

    with open(args.out, "w") as f:
        for i, t in enumerate(texts):
            block = f"\n===== SAMPLE {i} " + "=" * 60 + f"\n{t}\n"
            log(block)
            f.write(block)
    log(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
