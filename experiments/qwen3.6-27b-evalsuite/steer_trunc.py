"""Truncated-steering behavioural fidelity: matryoshka vs standard, head to head.

The matryoshka steering claim (what the Space demo leans on): KEEP ONLY THE
FIRST k UNITS of the explanation, re-encode them with the critic, patch the
reconstruction back at the token, and the continuation still steers the way the
FULL explanation would. For the standard NLA a truncated `<explanation>` prefix
re-encodes OOD (its early sentences are throat-clearing) so the truncated steer
misfires until the explanation is nearly complete.

Positions come straight from the precache (space/ + space_std/), so we reuse the
already-validated verbalizations for BOTH models at the SAME positions/texts —
no AV re-generation, no prefill/stop-protocol risk. Only base + the two critics
are needed. Per selected (entry, pos):

  extract   v = raw-base L42 activation at pos (true activation)
  encode    critic re-encodes v̂(first-k) for k in K_LIST and v̂(all); norm-match
  continue  patch each v̂_k into L42 at pos, regenerate the text seed-matched

Fidelity(k) is judged offline (nex-n2-mini): does the k-truncated-steer
continuation match the all-units-steer continuation? cos(v̂_k, v̂_all) and
FVE(v̂_k, v) are logged as no-judge proxies.

Phased so one 80GB card holds one 27B at a time:
  python steer_trunc.py extract
  python steer_trunc.py encode mat ; python steer_trunc.py encode std
  python steer_trunc.py continue
"""
import argparse, json, os, re
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import critic_predict
from nla.utils.arch_adapters import resolve_decoder_layers

BASE_ID = "Qwen/Qwen3.6-27B"
LAYER = 42
DEV = "cuda"
SEEDS = [0, 1]
CONT_TOKENS = 60
K_LIST = [1, 2, 3, 0]          # 0 == all units
N_PER_ENTRY = 9                # positions sampled per precache entry
MIN_POS = 60
MODELS = {
    "mat": dict(repo="ceselder/nla-qwen36-27b-matryoshka", tok="warmstart_av_lora",
                critic="rl_critic_step400", space="space", unit="line"),
    "std": dict(repo="ceselder/qwen3.6-27b-nla-L42", tok="av_sft_lora",
                critic="rl_critic_step400", space="space_std", unit="sentence"),
}


def sent_units(line):
    bounds = []
    for m in re.finditer(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])', line):
        if (line[:m.end()].count('"') + line[:m.end()].count('“') + line[:m.end()].count('”')) % 2 == 0:
            bounds.append(m.end())
    units, prev = [], 0
    for b in bounds:
        units.append(line[prev:b]); prev = b
    units.append(line[prev:])
    return [u.strip() for u in units if u.strip()]


def units_of(model, lines):
    if MODELS[model]["unit"] == "line":
        return [l.strip() for l in lines if l.strip()]
    return [u for ln in lines for u in sent_units(ln)]


def select_positions():
    """Deterministic spread of positions per entry with enough units in BOTH
    models (so k=1,2,3 truncation is meaningful). Returns [(ei,pos)]."""
    mat = json.load(open("space/precache.json"))["entries"]
    std = json.load(open("space_std/precache.json"))["entries"]
    picks = []
    for ei in range(len(mat)):
        cand = []
        for pos, r in enumerate(mat[ei]["results"]):
            if pos < MIN_POS or not r or not r.get("lines"):
                continue
            rs = std[ei]["results"][pos]
            if not rs or not rs.get("lines"):
                continue
            n_mat = len(units_of("mat", r["lines"]))
            n_std = len(units_of("std", rs["lines"]))
            if n_mat >= 4 and n_std >= 3:
                cand.append(pos)
        if not cand:
            continue
        step = max(1, len(cand) // N_PER_ENTRY)
        picks += [(ei, cand[i]) for i in range(0, len(cand), step)][:N_PER_ENTRY]
    return picks, mat, std


class _StopForward(Exception):
    pass


class Base:
    def __init__(self):
        print("[base] loading raw Qwen3.6-27B…", flush=True)
        self.m = AutoModelForCausalLM.from_pretrained(
            BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=DEV).eval()
        self.tok = AutoTokenizer.from_pretrained(BASE_ID)
        self.layers = resolve_decoder_layers(self.m)
        self.patch = [None]

        def hook(module, inputs, output):
            pr = self.patch[0]
            if pr is None:
                return
            pos, vec = pr
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == pos + 1:
                h[:, pos, :] = vec.to(h.dtype)
            return output
        self.layers[LAYER].register_forward_hook(hook)

    @torch.inference_mode()
    def extract(self, ids):
        grabbed = {}

        def grab(module, inputs, output):
            grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
            raise _StopForward
        h = self.layers[LAYER].register_forward_hook(grab)
        try:
            try:
                self.m(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
            except _StopForward:
                pass
            return grabbed["h"][0, -1].float().cpu()
        finally:
            h.remove()

    @torch.inference_mode()
    def cont(self, ids, seed, vec):
        t = torch.tensor([ids], device=DEV)
        self.patch[0] = None if vec is None else (len(ids) - 1, vec.to(DEV))
        torch.manual_seed(seed)
        try:
            out = self.m.generate(
                input_ids=t, attention_mask=torch.ones_like(t),
                max_new_tokens=CONT_TOKENS, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, pad_token_id=self.tok.eos_token_id)
        finally:
            self.patch[0] = None
        return self.tok.decode(out[0, len(ids):], skip_special_tokens=True)


def key(ei, pos):
    return f"{ei}_{pos}"


def phase_extract():
    picks, mat, _ = select_positions()
    print(f"[select] {len(picks)} positions: {picks}", flush=True)
    base = Base()
    acts, meta = {}, {}
    for ei, pos in picks:
        ids = mat[ei]["ids"][: pos + 1]
        acts[key(ei, pos)] = base.extract(ids).numpy()
        ctx = "".join(mat[ei]["pieces"][max(0, pos - 24): pos + 1])[-160:]
        nopatch = [base.cont(ids, s, None) for s in SEEDS]
        meta[key(ei, pos)] = dict(ei=ei, pos=pos, ids=ids, ctx=ctx, nopatch=nopatch)
        print(f"[extract {key(ei,pos)}] ctx=…{ctx[-50:]!r} -> {nopatch[0][:50]!r}", flush=True)
    np.savez_compressed("st_acts.npz", **acts)
    json.dump(meta, open("st_meta.json", "w"))
    print("[saved] st_acts.npz st_meta.json", flush=True)


def phase_encode(model):
    C = MODELS[model]
    acts = {k: np.load("st_acts.npz")[k] for k in np.load("st_acts.npz").files}
    pre = json.load(open(f"{C['space']}/precache.json"))["entries"]
    root = snapshot_download(C["repo"], allow_patterns=[f"{C['tok']}/*", f"{C['critic']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok']}")
    cfg = load_nla_config(C["space"], tok)
    SCALE = float(cfg.mse_scale); TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(Path(C["space"]) / "mu.npy"), dtype=torch.float32)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{C['critic']}", torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    critic.to(DEV).eval()

    @torch.inference_mode()
    def enc(text):
        ids = tok.encode(TPL.format(explanation=text), add_special_tokens=False)[:1024]
        bx = torch.tensor([ids], device=DEV)
        return critic_predict(critic, bx, torch.ones_like(bx), SCALE)[0].float().cpu()

    out = {}
    for k in sorted(acts):
        ei, pos = map(int, k.split("_"))
        lines = pre[ei]["results"][pos]["lines"]
        units = units_of(model, lines)
        v = torch.tensor(acts[k], dtype=torch.float32)
        vn = v.norm().clamp_min(1e-12)
        gn = v / vn * SCALE
        denom = ((gn - MU) ** 2).mean().item()
        p_all = enc("\n".join(units))
        vall = p_all / p_all.norm().clamp_min(1e-12) * vn
        rec = {}
        for kk in K_LIST:
            us = units if kk == 0 else units[:kk]
            p = enc("\n".join(us))
            vk = p / p.norm().clamp_min(1e-12) * vn
            pn = p / p.norm().clamp_min(1e-12) * SCALE
            fve = 1.0 - ((pn - gn) ** 2).mean().item() / denom
            cos_all = float(vk @ vall / (vk.norm() * vall.norm()))
            rec[str(kk)] = dict(vec=vk, cos_all=round(cos_all, 4), fve=round(fve, 4))
        out[k] = dict(n_units=len(units), units=units, rec=rec)
        print(f"[enc {model} {k}] units={len(units)} fve1={rec['1']['fve']} "
              f"fveAll={rec['0']['fve']} cos1={rec['1']['cos_all']}", flush=True)
    torch.save(out, f"st_vecs_{model}.pt")
    print(f"[saved] st_vecs_{model}.pt", flush=True)


def phase_continue():
    meta = json.load(open("st_meta.json"))
    enc = {m: torch.load(f"st_vecs_{m}.pt", weights_only=False) for m in MODELS}
    base = Base()
    results = []
    for k in sorted(meta, key=lambda z: (int(z.split("_")[0]), int(z.split("_")[1]))):
        md = meta[k]
        row = {"key": k, "ei": md["ei"], "pos": md["pos"], "ctx": md["ctx"],
               "nopatch": md["nopatch"], "models": {}}
        ids = md["ids"]
        for m in MODELS:
            e = enc[m][k]
            arms = {}
            for kk in K_LIST:
                r = e["rec"][str(kk)]
                arms[str(kk)] = dict(cos_all=r["cos_all"], fve=r["fve"],
                                     gen=[base.cont(ids, s, r["vec"]) for s in SEEDS])
            row["models"][m] = dict(n_units=e["n_units"], units=e["units"], arms=arms)
            print(f"[cont {m} {k}] k1={arms['1']['gen'][0][:45]!r} "
                  f"kAll={arms['0']['gen'][0][:45]!r}", flush=True)
        results.append(row)
    json.dump({"K_LIST": K_LIST, "seeds": SEEDS, "results": results},
              open("st_surgery.json", "w"), indent=1)
    print("[saved] st_surgery.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    sub.add_parser("extract")
    pe = sub.add_parser("encode"); pe.add_argument("model", choices=list(MODELS))
    sub.add_parser("continue")
    a = ap.parse_args()
    if a.phase == "extract": phase_extract()
    elif a.phase == "encode": phase_encode(a.model)
    else: phase_continue()
