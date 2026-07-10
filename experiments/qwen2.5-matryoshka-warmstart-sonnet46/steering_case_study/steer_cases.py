"""Steering case study: edit NLA explanation lines → causally steer the model.

Standalone (transformers-only, one 48GB GPU) version of the explorer Spaces'
intervention feature, run over FRESH fork-heavy texts to find cases where a
targeted line edit visibly and consistently redirects the continuation.

Pipeline per case (mirrors the v3 Space app):
  extract   full Qwen2.5-7B-Instruct, block-L20 output at the last prefix token
  verbalize v3 AV (iter200), embedding injection at the marker, T=1, 9 lines
  encode    v3 AR critic on original vs edited lines → v̂_orig, v̂_edit
  patch     norm-matched v̂ replaces the L20 block output at that position
  continue  seed-matched T=1 continuations: no-patch / v̂_orig / v̂_edit × seeds

Two phases with a JSON handoff (so a human can craft targeted edits):
  python steer_cases.py verbalize            # texts.json → cases.json
  python steer_cases.py steer                # cases.json + edits.json → results
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/nla")  # repo root holding nla_inference.py
from nla_inference import NLACritic  # noqa: E402

RL_REPO = "syvb/nla-qwen2.5-7b-L20-v3-rl"
RL_ITER = "iter_0000200"
SUBJECT_ID = "Qwen/Qwen2.5-7B-Instruct"
N_LINES = 9
MAX_NEW_AV = 220
CONT_TOKENS = 90
SEEDS = [0, 1, 2]
HERE = Path(__file__).resolve().parent
DEV = "cuda"


# ── shared loading ────────────────────────────────────────────────────────────
def load_ckpts():
    root = snapshot_download(RL_REPO, allow_patterns=[f"{RL_ITER}/*"])
    return f"{root}/{RL_ITER}/av", f"{root}/{RL_ITER}/ar"


class _StopForward(Exception):
    pass


class Subject:
    """Full subject model: L-layer extraction hook + patched continuation."""

    def __init__(self, layer: int):
        self.layer = layer
        self.model = AutoModelForCausalLM.from_pretrained(
            SUBJECT_ID, torch_dtype=torch.bfloat16).to(DEV).eval()
        self.tok = AutoTokenizer.from_pretrained(SUBJECT_ID)
        self.patch = [None]  # (pos, vec) — same hook pattern as the Space

        def _patch_hook(module, inputs, output):
            pr = self.patch[0]
            if pr is None:
                return
            pos, vec = pr
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] == pos + 1:
                h[:, pos, :] = vec.to(h.dtype)
            return output

        self.model.model.layers[layer].register_forward_hook(_patch_hook)

    @torch.inference_mode()
    def extract(self, ids: list[int]) -> torch.Tensor:
        grabbed = {}

        def grab(module, inputs, output):
            grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
            raise _StopForward

        h = self.model.model.layers[self.layer].register_forward_hook(grab)
        try:
            try:
                self.model(input_ids=torch.tensor([ids], device=DEV), use_cache=False)
            except _StopForward:
                pass
            return grabbed["h"][0, -1].float()
        finally:
            h.remove()

    @torch.inference_mode()
    def continue_text(self, ids: list[int], seed: int,
                      patch_vec: torch.Tensor | None) -> str:
        t = torch.tensor([ids], device=DEV)
        self.patch[0] = None if patch_vec is None else (len(ids) - 1, patch_vec.to(DEV))
        torch.manual_seed(seed)
        try:
            out = self.model.generate(
                input_ids=t, attention_mask=torch.ones_like(t),
                max_new_tokens=CONT_TOKENS, do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, pad_token_id=self.tok.eos_token_id)
        finally:
            self.patch[0] = None
        return self.tok.decode(out[0, len(ids):], skip_special_tokens=True)


class Verbalizer:
    """v3 AV: embedding injection at the marker token, T=1 line sampling."""

    def __init__(self, av_dir: str):
        meta = yaml.safe_load(open(f"{av_dir}/nla_meta.yaml"))
        tpl = meta["prompt_templates"].get("av") or meta["prompt_templates"]["actor"]
        tokens = meta["tokens"]
        self.inj_scale = float(meta["extraction"]["injection_scale"])
        self.tok = AutoTokenizer.from_pretrained(av_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            av_dir, torch_dtype=torch.bfloat16).to(DEV).eval()
        self.prompt_ids = self.tok.apply_chat_template(
            [{"role": "user", "content": tpl.format(injection_char=tokens["injection_char"])}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt")
        pos = (self.prompt_ids[0] == tokens["injection_token_id"]).nonzero(as_tuple=True)[0]
        assert len(pos) == 1, "marker must appear exactly once"
        self.inj_pos = int(pos[0])

    @torch.inference_mode()
    def verbalize(self, v: torch.Tensor, seed: int) -> list[str]:
        emb = self.model.get_input_embeddings()(self.prompt_ids.to(DEV)).clone()
        vn = v / v.norm().clamp_min(1e-12) * self.inj_scale
        emb[0, self.inj_pos] = vn.to(torch.bfloat16)
        torch.manual_seed(seed)
        out = self.model.generate(
            inputs_embeds=emb,
            attention_mask=torch.ones(1, emb.shape[1], device=DEV, dtype=torch.long),
            max_new_tokens=MAX_NEW_AV, do_sample=True, temperature=1.0,
            top_p=1.0, top_k=0, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()][:N_LINES]


def fve_of(pred: torch.Tensor, gold: torch.Tensor, mu: torch.Tensor,
           scale: float) -> tuple[float, float]:
    pn = pred / pred.norm().clamp_min(1e-12) * scale
    gn = gold / gold.norm().clamp_min(1e-12) * scale
    denom = ((gn - mu) ** 2).mean().item()
    fve = 1.0 - ((pn - gn) ** 2).mean().item() / denom
    cos = float(pn @ gn / (pn.norm() * gn.norm()))
    return fve, cos


# ── phases ────────────────────────────────────────────────────────────────────
def phase_verbalize():
    texts = json.load(open(HERE / "texts.json"))
    av_dir, ar_dir = load_ckpts()
    ar_meta = yaml.safe_load(open(f"{ar_dir}/nla_meta.yaml"))
    layer = ar_meta["critic"]["extraction_layer_index"]
    subj = Subject(layer)
    av = Verbalizer(av_dir)
    critic = NLACritic(ar_dir, device=DEV)
    mu = torch.tensor(np.load(HERE / "mu.npy"), dtype=torch.float32)

    cases = []
    for i, t in enumerate(texts):
        ids = subj.tok(t["text"], add_special_tokens=True)["input_ids"]
        pos = len(ids) - 1
        v = subj.extract(ids).cpu()
        lines = av.verbalize(v.to(DEV), seed=1000 + i)
        preds = critic.reconstruct_batch(
            ["\n".join(lines[:k]) for k in range(1, len(lines) + 1)])
        fves = [round(fve_of(p, v, mu, critic.mse_scale)[0], 4) for p in preds]
        tok_piece = subj.tok.decode([ids[pos]])
        cases.append({"id": t["id"], "text": t["text"], "note": t.get("note", ""),
                      "pos": pos, "token": tok_piece, "lines": lines, "fve": fves})
        print(f"[{t['id']}] pos={pos} tok={tok_piece!r} fve_full={fves[-1] if fves else None}")
        for k, ln in enumerate(lines):
            print(f"    {k}: {ln}")
    json.dump(cases, open(HERE / "cases.json", "w"), indent=1)
    print(f"[saved] {HERE / 'cases.json'}")


def phase_steer():
    cases = {c["id"]: c for c in json.load(open(HERE / "cases.json"))}
    edits = json.load(open(HERE / "edits.json"))
    av_dir, ar_dir = load_ckpts()
    ar_meta = yaml.safe_load(open(f"{ar_dir}/nla_meta.yaml"))
    subj = Subject(ar_meta["critic"]["extraction_layer_index"])
    critic = NLACritic(ar_dir, device=DEV)
    mu = torch.tensor(np.load(HERE / "mu.npy"), dtype=torch.float32)

    results = []
    for e in edits:
        # e["case"] (optional) lets several edits target one case — e.g. a
        # semantic flip AND a neutral-rewording control with distinct ids.
        c = cases[e.get("case", e["id"])]
        ids = subj.tok(c["text"], add_special_tokens=True)["input_ids"]
        assert len(ids) - 1 == c["pos"], "tokenization drifted since verbalize"
        v = subj.extract(ids).cpu()
        preds = critic.reconstruct_batch(["\n".join(c["lines"]), "\n".join(e["lines"])])
        stats = [fve_of(p, v, mu, critic.mse_scale) for p in preds]
        vecs = [p / p.norm().clamp_min(1e-12) * v.norm() for p in preds]
        shift = float(vecs[0] @ vecs[1] / (vecs[0].norm() * vecs[1].norm()))
        conts = {"nopatch": [], "orig": [], "edit": []}
        for seed in SEEDS:
            conts["nopatch"].append(subj.continue_text(ids, seed, None))
            conts["orig"].append(subj.continue_text(ids, seed, vecs[0]))
            conts["edit"].append(subj.continue_text(ids, seed, vecs[1]))
        results.append({
            "id": e["id"], "note": e.get("note", ""), "pos": c["pos"],
            "token": c["token"], "orig_lines": c["lines"], "edit_lines": e["lines"],
            "fve_orig": round(stats[0][0], 4), "fve_edit": round(stats[1][0], 4),
            "cos_shift": round(shift, 4), "continuations": conts})
        print(f"[{e['id']}] fve {stats[0][0]:.3f}→{stats[1][0]:.3f} "
              f"cos(v̂o,v̂e)={shift:.3f}", flush=True)
        for s, (o, ed) in enumerate(zip(conts["orig"], conts["edit"])):
            print(f"  seed{s} ORIG: {o[:100]!r}")
            print(f"  seed{s} EDIT: {ed[:100]!r}")
    json.dump(results, open(HERE / "steer_results.json", "w"), indent=1)
    print(f"[saved] {HERE / 'steer_results.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["verbalize", "steer"])
    args = ap.parse_args()
    (phase_verbalize if args.phase == "verbalize" else phase_steer)()
