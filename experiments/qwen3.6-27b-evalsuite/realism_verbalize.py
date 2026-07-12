"""Verbalize a realism-ladder scenario with one NLA (mat_rl | mat_warm | std).

For a given scenario (system+user from the realism chain), render it with the base
Qwen chat template, extract layer-42 activations at a uniform sample of positions in
ONE forward pass (adapters disabled → activations are identical across all three
models, so we cache and reuse), then verbalize each position's activation with the
chosen AV at T=1.

Output per position: the AV's explanation lines (matryoshka: salience-ordered
bullets; standard: <explanation> prose). Downstream we judge each line for
eval/fiction awareness and ask WHERE in the explanation it appears.

  python3 realism_verbalize.py --scenario-file realism_versions.json --idx 1 --model mat_rl
  python3 realism_verbalize.py --scenario-file realism_v0.json       --idx 0 --model std

Reuses control_gen.CFG so the three model pipelines stay identical to the WildChat
control run. Fits on one A100 80GB (AV only, no critic).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

import control_gen as CG  # CFG, BASE_ID, LAYER, MIN_POS, MAX_NEW, THINK_OPEN, PRECLOSED, bullet_lines, expl_lines

HERE = Path(__file__).resolve().parent
N_POS_DEFAULT = 220


def load_scenario(scenario_file, idx):
    data = json.load(open(HERE / scenario_file))
    if "versions" in data:                         # realism_versions*.json
        v = next(x for x in data["versions"] if x["i"] == idx)
        return v["system"], v["user"]
    return data["system"], data["user"]            # realism_v0.json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True,
                    help="comma-separated file:idx pairs, e.g. realism_v0.json:0,realism_versions.json:1")
    ap.add_argument("--model", required=True, choices=list(CG.CFG))
    ap.add_argument("--n-pos", type=int, default=N_POS_DEFAULT)
    ap.add_argument("--av-batch", type=int, default=16)
    args = ap.parse_args()
    C = CG.CFG[args.model]
    scenarios = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in args.scenarios.split(",")]

    pats = [f"{C['tok_sub']}/*"] + [f"{sub}/*" for _, sub in C["adapters"]]
    root = snapshot_download(C["repo"], allow_patterns=pats)
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok_sub']}")
    cfg = load_nla_config(str(HERE / C["sidecar"]), tok)
    render_tok = AutoTokenizer.from_pretrained(CG.BASE_ID)

    print("[load] base + adapters…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        CG.BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    actor = None
    for name, sub in C["adapters"]:
        if actor is None:
            actor = PeftModel.from_pretrained(base, f"{root}/{sub}", adapter_name=name)
        else:
            actor.load_adapter(f"{root}/{sub}", adapter_name=name)  # mutates in place; return is _IncompatibleKeys
    actor.base_model.set_adapter([n for n, _ in C["adapters"]])
    actor.eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    layers = resolve_decoder_layers(actor.get_base_model())

    # fixed AV prompt (+ prefill), reused for every scenario
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], cfg.injection_char, tok)
    assert ptxt.endswith(CG.THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(CG.THINK_OPEN)] + CG.PRECLOSED + C["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    parse = CG.bullet_lines if C["mode"] == "bullet" else CG.expl_lines

    for scenario_file, idx in scenarios:
        tag = f"{Path(scenario_file).stem}_v{idx}"
        system_txt, user_txt = load_scenario(scenario_file, idx)
        msgs = [{"role": "system", "content": system_txt}, {"role": "user", "content": user_txt}]
        text = render_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = render_tok.encode(text, add_special_tokens=False)
        pieces = [render_tok.decode([i]) for i in ids]
        print(f"[scenario] {tag}: {len(ids)} tokens", flush=True)

        # extract sampled positions in ONE adapters-disabled forward (cached; shared across models)
        acts_p = HERE / f"realism_acts_{tag}.npz"
        if acts_p.exists():
            z = np.load(acts_p); acts = z["acts"]; positions = z["positions"].tolist()
            print(f"[cache] {len(acts)} activations", flush=True)
        else:
            hi = len(ids) - 1
            n = min(args.n_pos, hi - CG.MIN_POS)
            positions = sorted(set(np.linspace(CG.MIN_POS, hi, n).astype(int).tolist()))
            grabbed = {}

            def grab(m, i, o):
                grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()

            h = layers[CG.LAYER].register_forward_hook(grab)
            try:
                with torch.inference_mode(), actor.disable_adapter():
                    actor(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
            finally:
                h.remove()
            full = grabbed["h"][0].float().cpu().numpy()     # [seq, d]
            acts = full[positions]                            # [n, d]
            np.savez_compressed(acts_p, acts=acts, positions=np.array(positions))
            print(f"[extract] {len(acts)} positions", flush=True)

        out = []
        torch.manual_seed(0)
        with torch.inference_mode():
            for s in range(0, len(acts), args.av_batch):
                chunk = acts[s: s + args.av_batch]
                pt = base_pt.repeat(len(chunk), 1)
                vref[0] = torch.tensor(np.stack(chunk), dtype=torch.float32).cuda()
                try:
                    gen = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                         max_new_tokens=CG.MAX_NEW, do_sample=True, temperature=1.0,
                                         top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
                finally:
                    vref[0] = None
                for o in gen:
                    txt = C["prefill"] + tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                    out.append(parse(txt))
                print(f"  {tag} verbalize {min(s + args.av_batch, len(acts))}/{len(acts)}", flush=True)

        outp = f"realism_verb_{tag}_{args.model}.json"
        payload = {"model": args.model, "scenario_file": scenario_file, "idx": idx,
                   "positions": positions, "pieces": [pieces[p] for p in positions], "lines": out}
        json.dump(payload, open(HERE / outp, "w"))
        print(f"[saved] {outp}  sample={out[len(out)//2]}", flush=True)

    print("VERB_DONE", flush=True)


if __name__ == "__main__":
    main()
