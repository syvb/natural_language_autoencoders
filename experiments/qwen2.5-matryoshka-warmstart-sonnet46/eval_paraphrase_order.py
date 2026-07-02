"""Faithfulness eval #1 (paraphrase robustness) + #4 (order optimality) for an AV/AR pair.

For N held-out samples (same selection as sweep_fve.py):
  1. AV generates the explanation (greedy, 256 tokens); split into lines.
  2. Each line is PARAPHRASED by an external LLM (meaning preserved, wording
     maximally changed; line count + order preserved) — if round-trip FVE
     survives paraphrase, the pair communicates in semantics; if it collapses,
     the words were (partly) a co-adapted code.
  3. AR reconstructs under several conditions, set-level FVE per prefix length
     k = 1..K lines (full text when a sample has < k lines) and at full length:
       orig        model's own text and order
       para        line-wise paraphrase, model order
       reversed    model's lines, reversed order
       random      model's lines, shuffled (mean of 3 seeds)
       crossfloor  lines swapped in from OTHER samples (destruction floor)
       oracle      per-sample GREEDY best ordering of the model's own lines,
                   chosen by per-sample reconstruction error (depth K) — how
                   close is the model's order to optimal front-loading?

Env: AV_DIR, AR_DIR, EVAL, OUT (json), OPENROUTER_KEY_FILE (default
/root/.openrouter_key), N (default 100), K (default 10), DO_ORACLE (default 1).
Stages are resumable: explanations/paraphrases are cached inside OUT.
"""
import json
import os
import random
import re
import sys
import urllib.request

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.injection import inject_at_marked_positions
from nla.schema import INJECT_PLACEHOLDER, extract_explanation_open, normalize_activation
from nla_inference import NLACritic

AV_DIR = os.environ["AV_DIR"]
AR_DIR = os.environ["AR_DIR"]
EVAL = os.environ.get("EVAL", "/workspace/out/av_eval_v3.parquet")
OUT = os.environ.get("OUT", "/workspace/para_order.json")
KEYFILE = os.environ.get("OPENROUTER_KEY_FILE", "/root/.openrouter_key")
N = int(os.environ.get("N", "100"))
K = int(os.environ.get("K", "10"))
DO_ORACLE = os.environ.get("DO_ORACLE", "1") == "1"
MAXL = 20      # cap lines considered per sample (bounds oracle cost)
dev = "cuda"

state = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(state, open(OUT, "w"))


# ---------- stage 1: generate explanations ----------
t = pq.read_table(EVAL)
docs = t.column("doc_id").to_pylist()
seen = {}
for idx, d in enumerate(docs):
    if d not in seen:
        seen[d] = idx
first = list(seen.values())
step = max(1, len(first) // N)
sel = [first[min(i * step, len(first) - 1)] for i in range(N)]
vecs = [np.asarray(t.column("activation_vector")[i].as_py(), dtype=np.float32) for i in sel]

if "expls" not in state:
    import yaml
    meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
    T = meta["tokens"]
    inj_id, left, right, inj_char = (T["injection_token_id"], T["injection_left_neighbor_id"],
                                     T["injection_right_neighbor_id"], T["injection_char"])
    inj_scale = meta["extraction"]["injection_scale"]
    tok = AutoTokenizer.from_pretrained(AV_DIR); tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    av = AutoModelForCausalLM.from_pretrained(AV_DIR, dtype=torch.bfloat16).to(dev).eval()
    emb = av.get_input_embeddings()
    prompts = [t.column("prompt")[i].as_py() for i in sel]

    @torch.no_grad()
    def av_generate(bp, bv):
        seqs = []
        for p in bp:
            msgs = [{**m, "content": m["content"].replace(INJECT_PLACEHOLDER, inj_char)} for m in p]
            seqs.append(tok.apply_chat_template(msgs, add_generation_prompt=True))
        m = max(len(s) for s in seqs); pad = tok.pad_token_id
        inp = np.full((len(seqs), m), pad, dtype=np.int64); att = np.zeros((len(seqs), m), dtype=np.int64)
        for k, s in enumerate(seqs):
            inp[k, m - len(s):] = s; att[k, m - len(s):] = 1
        inp = torch.tensor(inp, device=dev); att = torch.tensor(att, device=dev)
        e = emb(inp)
        V = torch.stack([normalize_activation(torch.tensor(v).view(1, -1), inj_scale)[0] for v in bv])
        out = av.generate(inputs_embeds=inject_at_marked_positions(inp, e, V, inj_id, left, right),
                          attention_mask=att, max_new_tokens=256, do_sample=False, pad_token_id=pad)
        return [tok.decode(o, skip_special_tokens=True) for o in out]

    expls = []
    for s in range(0, len(sel), 16):
        for txt in av_generate(prompts[s:s+16], vecs[s:s+16]):
            expls.append(extract_explanation_open(txt) or txt)
        print(f"  AV gen {min(s+16, len(sel))}/{len(sel)}", flush=True)
    state["expls"] = expls
    save()
    del av, emb
    torch.cuda.empty_cache()
print("explanations ready", flush=True)

lines_per = [[ln.strip() for ln in re.split(r"\n+", e.strip()) if ln.strip()][:MAXL]
             for e in state["expls"]]

# ---------- stage 2: line-wise paraphrase ----------
if "paras" not in state:
    KEY = open(KEYFILE).read().strip()
    URL = "https://openrouter.ai/api/v1/chat/completions"
    PROMPT = """Paraphrase each numbered line below. Preserve each line's MEANING precisely but change the wording as much as possible (synonyms, different syntax, reordered clauses). Keep exactly {n} lines, same order, one paraphrase per input line. Do not add or merge content.

LINES:
{listing}

Answer with JSON only: {{"lines": ["...", ...]}} with exactly {n} strings."""

    def paraphrase(lines):
        listing = "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
        body = json.dumps({"model": "anthropic/claude-haiku-4.5", "temperature": 0.7,
                           "messages": [{"role": "user", "content": PROMPT.format(n=len(lines), listing=listing)}]}).encode()
        req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        for _ in range(4):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=120))
                txt = r["choices"][0]["message"]["content"]
                out = json.loads(txt[txt.find("{"):txt.rfind("}") + 1])["lines"]
                if len(out) == len(lines) and all(isinstance(x, str) and x.strip() for x in out):
                    return [x.strip() for x in out]
            except Exception:
                continue
        return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(paraphrase, lines_per))
    fails = sum(1 for r in results if r is None)
    state["paras"] = [r if r is not None else lp for r, lp in zip(results, lines_per)]
    state["para_fail_count"] = fails
    save()
    print(f"paraphrases ready ({fails} failures kept original)", flush=True)
paras_per = [p[:MAXL] for p in state["paras"]]

# ---------- stage 3: reconstructions ----------
critic = NLACritic(AR_DIR, device=dev)
ms = critic.mse_scale
G = torch.stack([torch.tensor(v) for v in vecs]).float()
Gn = G / G.norm(dim=-1, keepdim=True) * ms
mu = Gn.mean(0)
den = ((Gn - mu) ** 2).mean().item()
ctok = critic.tokenizer
ctok.padding_side = "right"
if ctok.pad_token_id is None:
    ctok.pad_token = ctok.eos_token


@torch.no_grad()
def recon(texts, bs=48):
    outs = []
    for i in range(0, len(texts), bs):
        prompts = [critic.template.format(explanation=e) for e in texts[i:i+bs]]
        enc = ctok(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
        ids = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev)
        hs = critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
        h = hs[torch.arange(hs.size(0)), am.sum(1) - 1]
        outs.append(critic.value_head(h).float().cpu())
    return torch.cat(outs, 0)


def set_fve(P):
    Pn = P / P.norm(dim=-1, keepdim=True) * ms
    return 1 - ((Pn - Gn) ** 2).mean().item() / den


def per_sample_err(P):
    Pn = P / P.norm(dim=-1, keepdim=True) * ms
    return ((Pn - Gn) ** 2).mean(dim=1)  # [N]


def prefix_curve(order_lines):
    """order_lines: list per sample of ordered lines. FVE at k=1..K + full."""
    curve = {}
    for k in range(1, K + 1):
        texts = ["\n".join(lp[:k]) for lp in order_lines]
        curve[k] = set_fve(recon(texts))
    curve["full"] = set_fve(recon(["\n".join(lp) for lp in order_lines]))
    return curve


results = state.setdefault("results", {})

if "orig" not in results:
    results["orig"] = prefix_curve(lines_per); save()
if "para" not in results:
    results["para"] = prefix_curve(paras_per); save()
if "reversed" not in results:
    results["reversed"] = prefix_curve([list(reversed(lp)) for lp in lines_per]); save()
if "random" not in results:
    accs = []
    for seed in range(3):
        rng = random.Random(seed)
        shuf = [rng.sample(lp, len(lp)) for lp in lines_per]
        accs.append(prefix_curve(shuf))
    results["random"] = {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}
    save()
if "crossfloor" not in results:
    # replace sample i's lines with sample (i+1)%N's (count-matched by cycling)
    cross = []
    for i, lp in enumerate(lines_per):
        src = lines_per[(i + 1) % len(lines_per)]
        cross.append([src[j % len(src)] for j in range(len(lp))])
    results["crossfloor"] = prefix_curve(cross); save()
print("core conditions done", flush=True)

if DO_ORACLE and "oracle" not in results:
    # depth-synchronized greedy across samples: at each depth, batch every
    # sample's candidate prefixes, pick the line minimizing per-sample error.
    chosen = [[] for _ in lines_per]
    remaining = [list(range(len(lp))) for lp in lines_per]
    oracle_curve = {}
    for depth in range(1, K + 1):
        cand_texts, cand_meta = [], []
        for i, lp in enumerate(lines_per):
            if not remaining[i]:
                continue
            base = [lp[j] for j in chosen[i]]
            for j in remaining[i]:
                cand_texts.append("\n".join(base + [lp[j]]))
                cand_meta.append((i, j))
        if not cand_texts:
            break
        P = recon(cand_texts)
        errs = per_sample_err_for_candidates = ((P / P.norm(dim=-1, keepdim=True) * ms
                 - torch.stack([Gn[i] for i, _ in cand_meta])) ** 2).mean(dim=1)
        best = {}
        for (i, j), e in zip(cand_meta, errs.tolist()):
            if i not in best or e < best[i][0]:
                best[i] = (e, j)
        for i, (_, j) in best.items():
            chosen[i].append(j); remaining[i].remove(j)
        texts = ["\n".join([lp[j] for j in chosen[i]]) if chosen[i] else "\n".join(lp)
                 for i, lp in enumerate(lines_per)]
        oracle_curve[depth] = set_fve(recon(texts))
        print(f"  oracle depth {depth}: FVE={oracle_curve[depth]:.4f}", flush=True)
    results["oracle"] = oracle_curve
    save()

# ---------- summary ----------
print("\n=== curves (FVE by prefix lines) ===")
conds = [c for c in ("orig", "para", "oracle", "random", "reversed", "crossfloor") if c in results]
hdr = "k    " + "  ".join(f"{c:>10s}" for c in conds)
print(hdr)
for k in list(range(1, K + 1)) + ["full"]:
    row = f"{str(k):4s} "
    for c in conds:
        v = results[c].get(str(k), results[c].get(k))
        row += f"  {v:>10.4f}" if v is not None else f"  {'—':>10s}"
    print(row)
of, pf = results["orig"]["full"], results["para"]["full"]
print(f"\nparaphrase retention (full): {pf:.4f}/{of:.4f} = {pf/of:.3f}")
print("EVAL_PARA_ORDER_DONE", flush=True)
