"""Planning-in-Poetry reproduction on Qwen3.6-27B: matryoshka vs standard NLA.

Reproduces the paper's case study (rhyming-couplet planning, Opus 4.6 there)
on Qwen3.6-27B with both 27B NLAs, on TWO couplets chosen by screening
(screen_couplets.py, N=25 each):

  paper  "Write a rhyming couplet about a rabbit." +
         "He saw a carrot and had to grab it,"      -> rabbit 20/25.
         The paper's couplet + edit (rabbit->mouse, habit->house,
         carrot->cheese); the prompt names the rabbit, so the steer must
         override a prompt-anchored plan.
  spont  "Write a rhyming couplet." +
         "The old grey cat had spied a mouse,"      -> house 12/25.
         Fully spontaneous plan (no prompt hint); edit reversed
         (mouse->rabbit, house->habit, cheese->carrot).

Phases (one 27B resident at a time; each tag processed inside each phase):
  baseline   raw base: N second lines per tag (rhyme tally) + L42 activations
             for every couplet token.
  av M       AV explanations at every couplet token, extra samples at the
             steer token (the newline ending line 1).
  encode M   critic re-encodes the steer-token explanation, its edited
             version, and first-k truncations of both; Delta = v_edit-v_orig.
  steer      raw base: per tag x model x {direct patch, alpha-steer} x
             {full,k1,k2} at the steer token, N samples each, + no-patch arm.

  python poetry_steer.py baseline | av mat|std | encode mat|std | steer
"""
import argparse, json, os, re
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/workspace/hf")
import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

BASE_ID = "Qwen/Qwen3.6-27B"
LAYER = 42
DEV = "cuda"
N_SAMP = 25           # samples per steering arm / baseline tally
MAX_LINE = 28         # new tokens for the second line
AV_SAMPLES = 2        # explanations per couplet token (walkthrough)
AV_STEER_SAMPLES = 4  # extra explanations at the steer token
AV_BATCH = 8
ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0]
ALPHAS_K = [0.5, 1.0, 2.0]
N_LINES = 10          # matryoshka line cap (space-app convention)
PRECLOSED = "<think>\n\n</think>\n\n"
THINK_OPEN = "<think>\n"

PAPER_SUBS = [(r"\brabbits\b", "mice"), (r"\bRabbits\b", "Mice"),
              (r"\brabbit\b", "mouse"), (r"\bRabbit\b", "Mouse"),
              (r"\bhabits\b", "houses"), (r"\bhabit\b", "house"),
              (r"\bHabit\b", "House"),
              (r"\bcarrots\b", "cheese"), (r"\bCarrots\b", "Cheese"),
              (r"\bcarrot\b", "cheese"), (r"\bCarrot\b", "Cheese")]
SPONT_SUBS = [(r"\bmice\b", "rabbits"), (r"\bMice\b", "Rabbits"),
              (r"\bmouse\b", "rabbit"), (r"\bMouse\b", "Rabbit"),
              (r"\bhouses\b", "habits"), (r"\bhouse\b", "habit"),
              (r"\bHouse\b", "Habit"),
              (r"\bcheese\b", "carrot"), (r"\bCheese\b", "Carrot")]

CONFIGS = {
    "paper": dict(user="Write a rhyming couplet about a rabbit.",
                  line1="He saw a carrot and had to grab it,",
                  plan_word="rabbit", plan_re=r"\brabbit", subs=PAPER_SUBS,
                  targets=["mouse", "house", "mice", "cheese"]),
    "spont": dict(user="Write a rhyming couplet.",
                  line1="The old grey cat had spied a mouse,",
                  plan_word="house", plan_re=r"\b(house|mouse)", subs=SPONT_SUBS,
                  targets=["rabbit", "habit", "rabbits", "carrot"]),
}

MODELS = {
    "mat": dict(repo="ceselder/nla-qwen36-27b-matryoshka", tok="warmstart_av_lora",
                critic="rl_critic_step400", space="space", unit="line",
                av=["rl_av_lora_iter400"], prefill=""),
    "std": dict(repo="ceselder/qwen3.6-27b-nla-L42", tok="av_sft_lora",
                critic="rl_critic_step400", space="space_std", unit="sentence",
                av=["av_sft_lora", "av_rl_lora_step400"], prefill="<explanation>\n"),
}


def edit_text(t, subs):
    for pat, rep in subs:
        t = re.sub(pat, rep, t)
    return t


def chat_prompt(user_text, line1):
    return (f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{PRECLOSED}{line1}\n")


def sent_units(line):
    bounds = []
    for m in re.finditer(r'[.!?]["”\')\]]*\s+(?=[A-Z"“(\d])', line):
        if (line[:m.end()].count('"') + line[:m.end()].count('“')
                + line[:m.end()].count('”')) % 2 == 0:
            bounds.append(m.end())
    units, prev = [], 0
    for b in bounds:
        units.append(line[prev:b]); prev = b
    units.append(line[prev:])
    return [u.strip() for u in units if u.strip()]


def parse_units(model, text):
    if model == "std":
        body = re.sub(r"^\s*<explanation>\s*", "", text)
        body = re.sub(r"\s*</explanation>.*$", "", body, flags=re.S)
        lines = [l.strip() for l in body.split("\n") if l.strip()]
        return [u for ln in lines for u in sent_units(ln)]
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return lines[:N_LINES]


def last_words(gens):
    out = []
    for g in gens:
        line = g.split("\n")[0].strip()
        w = re.sub(r"[^a-z]", "", line.split()[-1].lower()) if line.split() else ""
        out.append(w)
    return out


class Base:
    """Raw base with a steering hook on the L42 block output.

    mode[0] is None | ("patch", pos, unit_dir) | ("add", pos, alpha, unit_dir).
    Patching/steering happens only during prefill (h.shape[1] == pos+1); decode
    steps (q_len==1) pass through. Norm-matching uses the LIVE h norm per row.
    """

    def __init__(self):
        print("[base] loading raw Qwen3.6-27B...", flush=True)
        self.m = AutoModelForCausalLM.from_pretrained(
            BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=DEV).eval()
        self.tok = AutoTokenizer.from_pretrained(BASE_ID)
        self.layers = resolve_decoder_layers(self.m)
        self.mode = [None]  # None | {layer_idx: spec}
        for li, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_hook(li))

    def _make_hook(self, li):
        def hook(module, inputs, output):
            md = self.mode[0]
            if not md or li not in md:
                return
            spec = md[li]
            h = output[0] if isinstance(output, tuple) else output
            if spec[0] == "patch":
                _, pos, d = spec
                if h.shape[1] != pos + 1:
                    return
                hn = h[:, pos, :].float().norm(dim=-1, keepdim=True)
                h[:, pos, :] = (d.to(h.device).float()[None] * hn).to(h.dtype)
            elif spec[0] == "add":
                _, pos, alpha, d = spec
                if h.shape[1] != pos + 1:
                    return
                hn = h[:, pos, :].float().norm(dim=-1, keepdim=True)
                h[:, pos, :] += (alpha * hn * d.to(h.device).float()[None]).to(h.dtype)
            else:  # ("patch_multi", start, dirs [P, D]) — whole-span patch
                _, start, dirs = spec
                P = dirs.shape[0]
                if h.shape[1] < start + P:
                    return  # decode steps / short prefill
                D = dirs.to(h.device).float()
                D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                seg = h[:, start:start + P, :].float()
                hn = seg.norm(dim=-1, keepdim=True)
                h[:, start:start + P, :] = (D[None] * hn).to(h.dtype)
            return output
        return hook

    @torch.inference_mode()
    def extract_all(self, ids):
        grabbed = {}

        def grab(module, inputs, output):
            grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
        hd = self.layers[LAYER].register_forward_hook(grab)
        try:
            self.m(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
        finally:
            hd.remove()
        return grabbed["h"][0].float().cpu()  # [seq, D]

    @torch.inference_mode()
    def extract_layers(self, ids, cstart):
        """One forward; per-layer block outputs for the couplet region."""
        grabbed = {}
        handles = []
        for li, layer in enumerate(self.layers):
            def grab(module, inputs, output, li=li):
                h = output[0] if isinstance(output, tuple) else output
                grabbed[li] = h[0, cstart:].detach().float().cpu()
            handles.append(layer.register_forward_hook(grab))
        try:
            self.m(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
        finally:
            for h in handles:
                h.remove()
        return grabbed  # {layer: [P, D] tensor}

    @torch.inference_mode()
    def gen(self, ids, n, seed, mode):
        if isinstance(mode, tuple):
            mode = {LAYER: mode}
        self.mode[0] = mode
        t = torch.tensor([ids], device=DEV).repeat(n, 1)
        torch.manual_seed(seed)
        try:
            out = self.m.generate(
                input_ids=t, attention_mask=torch.ones_like(t),
                max_new_tokens=MAX_LINE, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, pad_token_id=self.tok.eos_token_id)
        finally:
            self.mode[0] = None
        return [self.tok.decode(o[t.shape[1]:], skip_special_tokens=True) for o in out]


def phase_baseline():
    base = Base()
    tok = base.tok
    meta, acts = {"configs": {}}, {}
    for tag, C in CONFIGS.items():
        ptxt = chat_prompt(C["user"], C["line1"])
        ids = tok.encode(ptxt, add_special_tokens=False)
        pre = ptxt[: ptxt.index(C["line1"])]
        cstart = len(tok.encode(pre, add_special_tokens=False))
        assert tok.decode(ids[cstart:]) == C["line1"] + "\n", repr(tok.decode(ids[cstart:]))
        acts[tag] = base.extract_all(ids)[cstart:].numpy()
        gens = base.gen(ids, N_SAMP, 0, None)
        tally = Counter(last_words(gens))
        meta["configs"][tag] = dict(
            user=C["user"], line1=C["line1"], plan_word=C["plan_word"],
            ids=ids, cstart=cstart,
            pieces=[tok.decode([i]) for i in ids[cstart:]],
            gens=gens, tally=dict(tally.most_common()))
        print(f"[baseline {tag}] ntok={len(ids)} cstart={cstart} "
              f"{C['plan_word']}={tally.get(C['plan_word'], 0)}/{N_SAMP} "
              f"tally={dict(tally.most_common(6))}", flush=True)
    np.savez_compressed("po_acts.npz", **acts)
    json.dump(meta, open("po_meta.json", "w"), indent=1)
    print("[baseline] saved", flush=True)


def phase_av(model, extra=0):
    C = MODELS[model]
    meta = json.load(open("po_meta.json"))
    allacts = np.load("po_acts.npz")
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

    if extra:  # supplemental sampling at the steer token only
        result = json.load(open(f"po_av_{model}.json"))["tags"]
        for tag in CONFIGS:
            acts = allacts[tag]
            last = acts.shape[0] - 1
            for s in range(0, extra, AV_BATCH):
                nb = min(AV_BATCH, extra - s)
                gens = av_batch([acts[last]] * nb, seed=1000 + s)
                result[tag]["gens"][str(last)].extend(gens)
                print(f"[avx {model} {tag}] +{s + nb}/{extra}", flush=True)
            nplan = sum(1 for g in result[tag]["gens"][str(last)]
                        if re.search(CONFIGS[tag]["plan_re"], g, re.I))
            print(f"[avx {model} {tag}] steer-token expls mentioning plan: "
                  f"{nplan}/{len(result[tag]['gens'][str(last)])}", flush=True)
        json.dump({"model": model, "tags": result},
                  open(f"po_av_{model}.json", "w"), indent=1)
        print(f"[avx {model}] saved", flush=True)
        return

    result = {}
    for tag, TC in CONFIGS.items():
        acts = allacts[tag]
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
            print(f"[av {model} {tag}] {min(s + AV_BATCH, len(jobs))}/{len(jobs)}",
                  flush=True)
        nplan = sum(1 for i in outs for g in outs[i]
                    if re.search(TC["plan_re"], g, re.I))
        print(f"[av {model} {tag}] explanations mentioning plan word: "
              f"{nplan}/{len(jobs)}", flush=True)
        result[tag] = {"pieces": meta["configs"][tag]["pieces"],
                       "gens": {str(i): outs[i] for i in outs}}
    json.dump({"model": model, "tags": result},
              open(f"po_av_{model}.json", "w"), indent=1)
    print(f"[av {model}] saved", flush=True)


def phase_encode(model):
    C = MODELS[model]
    allacts = np.load("po_acts.npz")
    av = json.load(open(f"po_av_{model}.json"))["tags"]
    root = snapshot_download(C["repo"], allow_patterns=[f"{C['tok']}/*", f"{C['critic']}/*"])
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok']}")
    cfg = load_nla_config(C["space"], tok)
    SCALE = float(cfg.mse_scale)
    TPL = cfg.critic_prompt_template
    MU = torch.tensor(np.load(Path(C["space"]) / "mu.npy"), dtype=torch.float32)
    critic = NLACriticModel.from_pretrained(
        f"{root}/{C['critic']}", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(DEV).eval()

    @torch.inference_mode()
    def enc(text):
        ids = tok.encode(TPL.format(explanation=text), add_special_tokens=False)[:1024]
        bx = torch.tensor([ids], device=DEV)
        return critic_predict(critic, bx, torch.ones_like(bx), SCALE)[0].float().cpu()

    out = {}
    for tag, TC in CONFIGS.items():
        acts = allacts[tag]
        gens = av[tag]["gens"][str(acts.shape[0] - 1)]

        def score(g):
            # prefer explanations that surface the plan; among those, the one
            # with the most editable mentions (so the edit changes the most)
            nsub = sum(len(re.findall(p, g)) for p, _ in TC["subs"])
            return (1 if re.search(TC["plan_re"], g, re.I) else 0, nsub)
        sel = max(gens, key=score)
        units = parse_units(model, sel)
        units_edit = [edit_text(u, TC["subs"]) for u in units]
        if units == units_edit:
            print(f"[enc {model} {tag}] WARNING: edit is a no-op", flush=True)
        v = torch.tensor(acts[-1], dtype=torch.float32)
        vn = v.norm().clamp_min(1e-12)
        gn = v / vn * SCALE
        denom = ((gn - MU) ** 2).mean().item()

        def fve(p):
            pn = p / p.norm().clamp_min(1e-12) * SCALE
            return 1.0 - ((pn - gn) ** 2).mean().item() / denom

        rec = {"units": units, "units_edit": units_edit, "sel_expl": sel,
               "n_expl_sampled": len(gens), "arms": {}}
        for kn, k in (("full", None), ("k1", 1), ("k2", 2)):
            uo = units if k is None else units[:k]
            ue = units_edit if k is None else units_edit[:k]
            po, pe = enc("\n".join(uo)), enc("\n".join(ue))
            delta = pe - po
            rec["arms"][kn] = dict(
                v_orig=po, v_edit=pe, delta=delta,
                fve_orig=round(fve(po), 4), fve_edit=round(fve(pe), 4),
                cos_oe=round(float(po @ pe / (po.norm() * pe.norm())), 4),
                dnorm_rel=round(float(delta.norm() / vn), 4))
            r = rec["arms"][kn]
            print(f"[enc {model} {tag} {kn}] fve_orig={r['fve_orig']} "
                  f"fve_edit={r['fve_edit']} cos_oe={r['cos_oe']} "
                  f"|d|/|v|={r['dnorm_rel']}", flush=True)
        out[tag] = rec
    torch.save(out, f"po_vecs_{model}.pt")
    print(f"[enc {model}] saved po_vecs_{model}.pt", flush=True)


def phase_steer(only=None):
    meta = json.load(open("po_meta.json"))
    models = list(MODELS) if not only else [m for m in MODELS if m in only]
    enc = {m: torch.load(f"po_vecs_{m}.pt", weights_only=False) for m in models}
    res = json.load(open("po_steer.json")) if (only and os.path.exists("po_steer.json")) else {}
    base = Base()
    for tag in CONFIGS:
        ids = meta["configs"][tag]["ids"]
        pos = len(ids) - 1
        R = res.setdefault(tag, {"pos": pos, "n": N_SAMP, "arms": {}})

        def run(name, mode):
            gens = base.gen(ids, N_SAMP, 0, mode)
            R["arms"][name] = gens
            print(f"[steer {tag} {name}] "
                  f"{dict(Counter(last_words(gens)).most_common(5))}", flush=True)

        if "nopatch" not in R["arms"]:
            run("nopatch", None)
        for m in models:
            for kn in ("full", "k1", "k2"):
                A = enc[m][tag]["arms"][kn]
                po, pe, d = A["v_orig"], A["v_edit"], A["delta"]
                dd = d / d.norm().clamp_min(1e-12)
                if kn == "full":
                    run(f"{m}_patch_orig", ("patch", pos, po / po.norm()))
                run(f"{m}_patch_edit_{kn}", ("patch", pos, pe / pe.norm()))
                for a in (ALPHAS if kn == "full" else ALPHAS_K):
                    run(f"{m}_steer_{kn}_a{a}", ("add", pos, a, dd))
    json.dump(res, open("po_steer.json", "w"), indent=1)
    print("[steer] saved po_steer.json", flush=True)


def phase_controls():
    """Causal controls: can a TRUE activation from the other couplet's steer
    token redirect the rhyme when patched/steered at L42? If not, the
    single-token L42 channel is causally insufficient on this model and no
    reconstruction-based steer could succeed. xrec_* patches the other
    couplet's critic reconstruction (tests reconstruction sufficiency given
    the channel)."""
    meta = json.load(open("po_meta.json"))
    allacts = np.load("po_acts.npz")
    enc = {m: torch.load(f"po_vecs_{m}.pt", weights_only=False) for m in MODELS}
    res = json.load(open("po_steer.json"))
    base = Base()
    other = {"paper": "spont", "spont": "paper"}
    for tag in CONFIGS:
        ids = meta["configs"][tag]["ids"]
        pos = len(ids) - 1
        R = res[tag]

        def run(name, mode):
            gens = base.gen(ids, N_SAMP, 0, mode)
            R["arms"][name] = gens
            print(f"[ctrl {tag} {name}] "
                  f"{dict(Counter(last_words(gens)).most_common(5))}", flush=True)

        ot = other[tag]
        va = torch.tensor(allacts[ot][-1], dtype=torch.float32)
        vs = torch.tensor(allacts[tag][-1], dtype=torch.float32)
        run("xact_patch", ("patch", pos, va / va.norm()))
        dt = va - vs
        dt = dt / dt.norm().clamp_min(1e-12)
        for a in (0.5, 1.0, 2.0):
            run(f"xact_steer_a{a}", ("add", pos, a, dt))
        for m in MODELS:
            pr = enc[m][ot]["arms"]["full"]["v_orig"]
            run(f"xrec_{m}_patch", ("patch", pos, pr / pr.norm()))
    json.dump(res, open("po_steer.json", "w"), indent=1)
    print("[controls] saved po_steer.json", flush=True)


def phase_mencode(model):
    """Per-token critic encodings for MULTI-TOKEN steering: for every couplet
    position, select an explanation, apply the tag's edit, re-encode both.
    Saves po_mvecs_{model}.pt: {tag: {"orig": [P,D], "edit": [P,D]}}."""
    C = MODELS[model]
    av = json.load(open(f"po_av_{model}.json"))["tags"]
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
    for tag, TC in CONFIGS.items():
        gens = av[tag]["gens"]
        P = len(av[tag]["pieces"])
        vo, ve, nedit = [], [], 0

        def score(g):
            nsub = sum(len(re.findall(p, g)) for p, _ in TC["subs"])
            return (1 if re.search(TC["plan_re"], g, re.I) else 0, nsub)
        for i in range(P):
            sel = max(gens[str(i)], key=score)
            units = parse_units(model, sel)
            units_edit = [edit_text(u, TC["subs"]) for u in units]
            nedit += units != units_edit
            vo.append(enc("\n".join(units)))
            ve.append(enc("\n".join(units_edit)))
        out[tag] = {"orig": torch.stack(vo), "edit": torch.stack(ve)}
        cos = torch.nn.functional.cosine_similarity(out[tag]["orig"],
                                                    out[tag]["edit"], dim=-1)
        print(f"[menc {model} {tag}] P={P} edited@{nedit}/{P} "
              f"cos(orig,edit) min={cos.min():.4f} mean={cos.mean():.4f}", flush=True)
    torch.save(out, f"po_mvecs_{model}.pt")
    print(f"[menc {model}] saved po_mvecs_{model}.pt", flush=True)


def phase_msteer():
    """Multi-token steering arms at L42: whole-line patch with per-token
    critic reconstructions — orig (control), edit (payoff), and the OTHER
    couplet's orig reconstructions (cross control; reconstruction analogue of
    xactL42_all)."""
    meta = json.load(open("po_meta.json"))
    mv = {m: torch.load(f"po_mvecs_{m}.pt", weights_only=False) for m in MODELS}
    res = json.load(open("po_steer.json"))
    base = Base()
    other = {"paper": "spont", "spont": "paper"}
    for tag in CONFIGS:
        c = meta["configs"][tag]
        ids, cstart = c["ids"], c["cstart"]
        R = res[tag]

        def run(name, mode):
            gens = base.gen(ids, N_SAMP, 0, mode)
            R["arms"][name] = gens
            print(f"[msteer {tag} {name}] "
                  f"{dict(Counter(last_words(gens)).most_common(5))}", flush=True)

        for m in MODELS:
            run(f"mrec_{m}_orig_all", {LAYER: ("patch_multi", cstart, mv[m][tag]["orig"])})
            run(f"mrec_{m}_edit_all", {LAYER: ("patch_multi", cstart, mv[m][tag]["edit"])})
            run(f"mrec_{m}_xorig_all",
                {LAYER: ("patch_multi", cstart, mv[m][other[tag]]["orig"])})
    json.dump(res, open("po_steer.json", "w"), indent=1)
    print("[msteer] saved po_steer.json", flush=True)


LAYERS_SWEEP = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]
ALL_SWEEP = [12, 24, 36, 42, 54]


def phase_layers():
    """Layer-resolved causal controls with TRUE activations from the other
    couplet: (a) patch the steer token only, at each layer in LAYERS_SWEEP;
    (b) patch the WHOLE couplet region (all 11 tokens), at each layer in
    ALL_SWEEP. Establishes where (if anywhere) the rhyme plan is causally
    reachable at single-layer granularity."""
    meta = json.load(open("po_meta.json"))
    base = Base()
    acts = {}
    for tag in CONFIGS:
        c = meta["configs"][tag]
        acts[tag] = base.extract_layers(c["ids"], c["cstart"])
        print(f"[layers] extracted {tag}: {len(acts[tag])} layers x "
              f"{acts[tag][0].shape[0]} pos", flush=True)
    res = json.load(open("po_steer.json"))
    other = {"paper": "spont", "spont": "paper"}
    for tag in CONFIGS:
        c = meta["configs"][tag]
        ids, cstart = c["ids"], c["cstart"]
        pos = len(ids) - 1
        R = res[tag]

        def run(name, mode):
            gens = base.gen(ids, N_SAMP, 0, mode)
            R["arms"][name] = gens
            print(f"[layers {tag} {name}] "
                  f"{dict(Counter(last_words(gens)).most_common(5))}", flush=True)

        ot = other[tag]
        for L in LAYERS_SWEEP:
            va = acts[ot][L][-1]
            run(f"xactL{L}_patch", {L: ("patch", pos, va / va.norm())})
        for L in ALL_SWEEP:
            run(f"xactL{L}_all", {L: ("patch_multi", cstart, acts[ot][L])})
    json.dump(res, open("po_steer.json", "w"), indent=1)
    print("[layers] saved po_steer.json", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    sub.add_parser("baseline")
    pa = sub.add_parser("av"); pa.add_argument("model", choices=list(MODELS))
    pa.add_argument("--extra", type=int, default=0)
    pe = sub.add_parser("encode"); pe.add_argument("model", choices=list(MODELS))
    ps = sub.add_parser("steer"); ps.add_argument("--only", default=None)
    sub.add_parser("controls")
    sub.add_parser("layers")
    pm = sub.add_parser("mencode"); pm.add_argument("model", choices=list(MODELS))
    sub.add_parser("msteer")
    a = ap.parse_args()
    if a.phase == "baseline":
        phase_baseline()
    elif a.phase == "av":
        phase_av(a.model, extra=a.extra)
    elif a.phase == "encode":
        phase_encode(a.model)
    elif a.phase == "controls":
        phase_controls()
    elif a.phase == "layers":
        phase_layers()
    elif a.phase == "mencode":
        phase_mencode(a.model)
    elif a.phase == "msteer":
        phase_msteer()
    else:
        phase_steer(only=a.only.split(",") if a.only else None)
