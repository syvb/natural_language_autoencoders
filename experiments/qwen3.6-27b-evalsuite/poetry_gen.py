"""Generalization of the Planning-in-Poetry result across couplets.

Screens 12 spontaneous couplets ("Write a rhyming couplet." + a first line
ending in a rhyme anchor), keeps up to 6 where the base model concentrates
>=40% of second lines on ONE plan word, then runs the validated pipeline on
each: per-token AV explanations (n=18 at the line break for the observational
plan-mention metric), critic re-encodings of original and EDITED explanations
(each couplet's (plan, anchor) pair swapped round-robin to another selected
couplet's pair), and whole-line steering at L42:

  nopatch | mrec_{m}_orig_all (control) | mrec_{m}_edit_all (payoff) |
  mrec_{m}_edit_nl (single-token spot check, expected inert)

Phases (one 27B at a time):
  python poetry_gen.py screen
  python poetry_gen.py av mat|std
  python poetry_gen.py mencode mat|std
  python poetry_gen.py msteer
"""
import argparse, json, re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook

from poetry_steer import (AV_BATCH, BASE_ID, DEV, LAYER, MODELS, N_SAMP,
                          PRECLOSED, THINK_OPEN, Base, chat_prompt, edit_text,
                          last_words, parse_units)

USER = "Write a rhyming couplet."
AV_SAMPLES = 2         # per couplet token
AV_STEER_SAMPLES = 16  # extra at the line-break token (observational metric)
MIN_PLAN = 7           # plan-word concentration >= 7/25 to select
                       # (>=10 left only 2/12 couplets — concentrated plans
                       # are the exception; see pg_meta.json "screen")
MAX_SEL = 6

CANDS = [
    ("mouse",  "The old grey cat had spied a mouse,"),
    ("goat",   "The farmer fed his hungry goat,"),
    ("log",    "A tiny frog sat on a log,"),
    ("bread",  "The baker baked a loaf of bread,"),
    ("gleam",  "At night the stars began to gleam,"),
    ("bone",   "The little dog dug up a bone,"),
    ("tree",   "He climbed up high into a tree,"),
    ("sea",    "She sailed her boat across the sea,"),
    ("cheese", "The mouse crept out to find some cheese,"),
    ("line",   "The fisherman cast out his line,"),
    ("coat",   "He put on his hat and grabbed his coat,"),
    ("rain",   "The children danced out in the rain,"),
]


def phase_screen():
    base = Base()
    tok = base.tok
    rows = []
    for name, line1 in CANDS:
        ptxt = chat_prompt(USER, line1)
        ids = tok.encode(ptxt, add_special_tokens=False)
        pre = ptxt[: ptxt.index(line1)]
        cstart = len(tok.encode(pre, add_special_tokens=False))
        assert tok.decode(ids[cstart:]) == line1 + "\n", repr(tok.decode(ids[cstart:]))
        gens = base.gen(ids, N_SAMP, 0, None)
        tally = Counter(last_words(gens))
        anchor = re.sub(r"[^a-z]", "", line1.split()[-1].lower())
        plan, cnt = None, 0
        for w, c in tally.most_common():
            if len(w) >= 3 and w != anchor and w.isalpha():
                plan, cnt = w, c
                break
        rows.append(dict(name=name, line1=line1, ids=ids, cstart=cstart,
                         anchor=anchor, plan=plan, plan_n=cnt,
                         pieces=[tok.decode([i]) for i in ids[cstart:]],
                         tally=dict(tally.most_common()), gens=gens))
        print(f"[screen {name}] anchor={anchor} plan={plan} {cnt}/{N_SAMP} "
              f"tally={dict(tally.most_common(5))}", flush=True)
    sel = sorted([r for r in rows if r["plan"] and r["plan_n"] >= MIN_PLAN],
                 key=lambda r: -r["plan_n"])[:MAX_SEL]
    assert len(sel) >= 3, f"only {len(sel)} couplets passed the screen"
    for i, r in enumerate(sel):
        p = sel[(i + 1) % len(sel)]
        subs = []
        for a, b in ((r["plan"], p["plan"]), (r["anchor"], p["anchor"])):
            if a == b:
                continue
            subs += [(rf"\b{a}s\b", b + "s"),
                     (rf"\b{a.capitalize()}s\b", b.capitalize() + "s"),
                     (rf"\b{a}\b", b),
                     (rf"\b{a.capitalize()}\b", b.capitalize())]
        r["partner"] = p["name"]
        r["subs"] = subs
        r["plan_re"] = rf"\b{r['plan']}"
        r["targets"] = [p["plan"], p["plan"] + "s"]
        print(f"[select {r['name']}] plan={r['plan']} ({r['plan_n']}/{N_SAMP}) "
              f"edit -> {p['plan']}/{p['anchor']} (partner {p['name']})", flush=True)
    acts = {r["name"]: base.extract_all(r["ids"])[r["cstart"]:].numpy() for r in sel}
    np.savez_compressed("pg_acts.npz", **acts)
    KEEP = ("name", "line1", "ids", "cstart", "pieces", "anchor", "plan",
            "plan_n", "partner", "subs", "plan_re", "targets", "tally")
    json.dump({"user": USER,
               "selected": [{k: r[k] for k in KEEP} for r in sel],
               "screen": [{k: r[k] for k in
                           ("name", "line1", "anchor", "plan", "plan_n", "tally", "gens")}
                          for r in rows]},
              open("pg_meta.json", "w"), indent=1)
    print(f"[screen] saved; {len(sel)} couplets selected", flush=True)


def load_selected():
    return json.load(open("pg_meta.json"))["selected"]


def phase_av(model):
    C = MODELS[model]
    sel = load_selected()
    allacts = np.load("pg_acts.npz")
    pats = sorted({f"{p}/*" for p in C["av"] + [C["tok"]]})
    root = snapshot_download(C["repo"], allow_patterns=pats)
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok']}")
    cfg = load_nla_config(C["space"], tok)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=DEV).eval()
    if len(C["av"]) == 2:
        base = PeftModel.from_pretrained(base, f"{root}/{C['av'][0]}").merge_and_unload()
    actor = PeftModel.from_pretrained(base, f"{root}/{C['av'][-1]}").eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], cfg.injection_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + C["prefill"]
    pids = tok.encode(ptxt, add_special_tokens=False)
    print(f"[av {model}] prompt len {len(pids)}", flush=True)

    @torch.no_grad()
    def av_batch(vecs, seed):
        n = len(vecs)
        pt = torch.tensor([pids], dtype=torch.long, device=DEV).repeat(n, 1)
        vref[0] = torch.tensor(np.stack(vecs), dtype=torch.float32).cuda()
        torch.manual_seed(seed)
        try:
            out = actor.generate(
                input_ids=pt, attention_mask=torch.ones_like(pt),
                max_new_tokens=256, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
        finally:
            vref[0] = None
        return [C["prefill"] + tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                for o in out]

    result = {}
    for r in sel:
        acts = allacts[r["name"]]
        ncp = acts.shape[0]
        jobs = []
        for i in range(ncp):
            n = AV_SAMPLES + (AV_STEER_SAMPLES if i == ncp - 1 else 0)
            jobs += [i] * n
        outs = {i: [] for i in range(ncp)}
        for s in range(0, len(jobs), AV_BATCH):
            chunk = jobs[s:s + AV_BATCH]
            for i, g in zip(chunk, av_batch([acts[i] for i in chunk], seed=s)):
                outs[i].append(g)
        nl = outs[ncp - 1]
        nmention = sum(bool(re.search(r["plan_re"], g, re.I)) for g in nl)
        print(f"[av {model} {r['name']}] line-break plan mentions "
              f"{nmention}/{len(nl)}", flush=True)
        result[r["name"]] = {"gens": {str(i): outs[i] for i in outs},
                             "nl_mention": nmention, "nl_n": len(nl)}
    json.dump({"model": model, "tags": result},
              open(f"pg_av_{model}.json", "w"), indent=1)
    print(f"[av {model}] saved", flush=True)


def phase_mencode(model):
    C = MODELS[model]
    sel = load_selected()
    av = json.load(open(f"pg_av_{model}.json"))["tags"]
    root = snapshot_download(C["repo"], allow_patterns=[f"{C['tok']}/*", f"{C['critic']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok']}")
    cfg = load_nla_config(C["space"], tok)
    SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    critic = NLACriticModel.from_pretrained(
        f"{root}/{C['critic']}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(DEV).eval()

    @torch.inference_mode()
    def enc(text):
        ids = tok.encode(TPL.format(explanation=text), add_special_tokens=False)[:1024]
        bx = torch.tensor([ids], device=DEV)
        return critic_predict(critic, bx, torch.ones_like(bx), SCALE)[0].float().cpu()

    out = {}
    for r in sel:
        gens = av[r["name"]]["gens"]
        P = len(r["pieces"])
        subs = [tuple(s) for s in r["subs"]]

        def score(g):
            nsub = sum(len(re.findall(p, g)) for p, _ in subs)
            return (1 if re.search(r["plan_re"], g, re.I) else 0, nsub)
        vo, ve, nedit = [], [], 0
        for i in range(P):
            s = max(gens[str(i)], key=score)
            units = parse_units(model, s)
            units_edit = [edit_text(u, subs) for u in units]
            nedit += units != units_edit
            vo.append(enc("\n".join(units)))
            ve.append(enc("\n".join(units_edit)))
        out[r["name"]] = {"orig": torch.stack(vo), "edit": torch.stack(ve)}
        cos = torch.nn.functional.cosine_similarity(out[r["name"]]["orig"],
                                                    out[r["name"]]["edit"], dim=-1)
        print(f"[menc {model} {r['name']}] P={P} edited@{nedit}/{P} "
              f"cos(orig,edit) min={cos.min():.4f} mean={cos.mean():.4f}", flush=True)
    torch.save(out, f"pg_mvecs_{model}.pt")
    print(f"[menc {model}] saved", flush=True)


def phase_msteer():
    sel = load_selected()
    mv = {m: torch.load(f"pg_mvecs_{m}.pt", weights_only=False) for m in MODELS}
    base = Base()
    res = {}
    for r in sel:
        ids, cstart = r["ids"], r["cstart"]
        pos = len(ids) - 1
        R = {"pos": pos, "n": N_SAMP, "plan": r["plan"], "targets": r["targets"],
             "partner": r["partner"], "arms": {}}

        def run(name, mode):
            gens = base.gen(ids, N_SAMP, 0, mode)
            R["arms"][name] = gens
            ws = last_words(gens)
            tt = sum(w in r["targets"] for w in ws)
            pp = sum(w == r["plan"] for w in ws)
            print(f"[msteer {r['name']} {name}] plan={pp}/{N_SAMP} "
                  f"target={tt}/{N_SAMP} {dict(Counter(ws).most_common(4))}", flush=True)

        run("nopatch", None)
        for m in MODELS:
            V = mv[m][r["name"]]
            run(f"mrec_{m}_orig_all", {LAYER: ("patch_multi", cstart, V["orig"])})
            run(f"mrec_{m}_edit_all", {LAYER: ("patch_multi", cstart, V["edit"])})
            ve = V["edit"][-1]
            run(f"mrec_{m}_edit_nl", {LAYER: ("patch", pos, ve / ve.norm())})
        res[r["name"]] = R
    json.dump(res, open("pg_steer.json", "w"), indent=1)
    print("[msteer] saved pg_steer.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    sub.add_parser("screen")
    pa = sub.add_parser("av"); pa.add_argument("model", choices=list(MODELS))
    pm = sub.add_parser("mencode"); pm.add_argument("model", choices=list(MODELS))
    sub.add_parser("msteer")
    a = ap.parse_args()
    if a.phase == "screen":
        phase_screen()
    elif a.phase == "av":
        phase_av(a.model)
    elif a.phase == "mencode":
        phase_mencode(a.model)
    else:
        phase_msteer()
