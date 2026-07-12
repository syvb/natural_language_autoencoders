"""Generate NLA explanations at 1 random token of each WildChat conversation —
proper chat-format CONTROL data for the eval-awareness ROC (the old controls
were pretraining docs, confounding format with eval-awareness).

One model per run (base + that AV). Shared extraction is cached to
control_acts.npz so it's computed once. Eval-awareness needs only the AV text,
so no critic — fits on one A100 80GB.

  python control_gen.py --model mat_rl    # ceselder/nla-qwen36-27b-matryoshka rl_av_lora_iter400
  python control_gen.py --model mat_warm  #                                    warmstart_av_lora
  python control_gen.py --model std       # ceselder/qwen3.6-27b-nla-L42 av_sft_lora+av_rl_lora_step400
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

BASE_ID = "Qwen/Qwen3.6-27B"
LAYER = 42
N_LINES = 10
MAX_NEW = 256
MAX_TOK = 2048          # cap rendered convo length
MIN_POS = 50            # need left-context
N_CONV = 1000
THINK_OPEN, PRECLOSED = "<think>\n", "<think>\n\n</think>\n\n"
HERE = Path(__file__).resolve().parent


class _Stop(Exception):
    pass

CFG = {
    "mat_rl":   dict(repo="ceselder/nla-qwen36-27b-matryoshka", adapters=[("rl", "rl_av_lora_iter400")],
                     tok_sub="warmstart_av_lora", sidecar="space", mode="bullet", prefill=""),
    "mat_warm": dict(repo="ceselder/nla-qwen36-27b-matryoshka", adapters=[("ws", "warmstart_av_lora")],
                     tok_sub="warmstart_av_lora", sidecar="space", mode="bullet", prefill=""),
    "std":      dict(repo="ceselder/qwen3.6-27b-nla-L42",
                     adapters=[("sft", "av_sft_lora"), ("rl", "av_rl_lora_step400")],
                     tok_sub="av_sft_lora", sidecar="space_std", mode="explanation", prefill="<explanation>\n"),
}


def bullet_lines(text):
    return [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()][:N_LINES]


def expl_lines(text):
    after = text.split("<explanation>", 1)[-1]
    body = after.split("</explanation>", 1)[0]
    return [ln.strip() for ln in body.split("\n") if ln.strip()][:N_LINES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--av-batch", type=int, default=16)
    args = ap.parse_args()
    C = CFG[args.model]
    pats = [f"{C['tok_sub']}/*"] + [f"{sub}/*" for _, sub in C["adapters"]]
    root = snapshot_download(C["repo"], allow_patterns=pats)
    tok = AutoTokenizer.from_pretrained(f"{root}/{C['tok_sub']}")
    cfg = load_nla_config(str(HERE / C["sidecar"]), tok)

    render_tok = AutoTokenizer.from_pretrained(BASE_ID)  # standard chat template (NLA tok carries the injection template)
    print("[load] base + adapters…", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
    actor = None
    for name, sub in C["adapters"]:
        if actor is None:
            actor = PeftModel.from_pretrained(base, f"{root}/{sub}", adapter_name=name)
        else:
            actor.load_adapter(f"{root}/{sub}", adapter_name=name)
    actor.base_model.set_adapter([n for n, _ in C["adapters"]])
    actor.eval()
    vref = [None]
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id, cfg.injection_right_neighbor_id, layer_idx=1)
    layers = resolve_decoder_layers(actor.get_base_model())

    # positions + activations (shared, cached)
    acts_p, ids_p = HERE / "control_acts.npz", HERE / "control_ids.json"
    if acts_p.exists() and ids_p.exists():
        z = np.load(acts_p); acts = z["acts"]; meta = json.load(open(ids_p))
        print(f"[cache] {len(acts)} activations", flush=True)
    else:
        convos = json.load(open(HERE / "control_convos.json"))
        rng = np.random.RandomState(0)
        acts, meta = [], []
        with torch.inference_mode():
            for ci, msgs in enumerate(convos):
                try:
                    text = render_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    ids = render_tok.encode(text, add_special_tokens=False)
                except Exception:
                    continue
                ids = ids[:MAX_TOK]
                if len(ids) < MIN_POS + 5:
                    continue
                pos = int(rng.randint(MIN_POS, len(ids) - 1))
                grabbed = {}

                def grab(m, i, o):
                    grabbed["h"] = (o[0] if isinstance(o, tuple) else o).detach()
                    raise _Stop

                h = layers[LAYER].register_forward_hook(grab)
                try:
                    with actor.disable_adapter():
                        try:
                            actor(input_ids=torch.tensor([ids[:pos + 1]], device="cuda"), use_cache=False)
                        except _Stop:
                            pass
                finally:
                    h.remove()
                acts.append(grabbed["h"][0, -1].float().cpu().numpy())
                meta.append({"convo": ci, "pos": pos, "ntok": len(ids)})
                if len(acts) % 100 == 0:
                    print(f"  extracted {len(acts)}", flush=True)
        acts = np.stack(acts)
        np.savez_compressed(acts_p, acts=acts)
        json.dump(meta, open(ids_p, "w"))
        print(f"[extract] {len(acts)} positions saved", flush=True)

    # build the fixed AV prompt (+ prefill), verbalize batched at T=1
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    ptxt = build_prompt_text([{"role": "user", "content": content}], cfg.injection_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + C["prefill"]
    prompt_ids = tok.encode(ptxt, add_special_tokens=False)
    base_pt = torch.tensor([prompt_ids], device="cuda")
    parse = bullet_lines if C["mode"] == "bullet" else expl_lines

    out = []
    torch.manual_seed(0)
    with torch.inference_mode():
        for s in range(0, len(acts), args.av_batch):
            chunk = acts[s: s + args.av_batch]
            pt = base_pt.repeat(len(chunk), 1)
            vref[0] = torch.tensor(np.stack(chunk), dtype=torch.float32).cuda()
            try:
                gen = actor.generate(input_ids=pt, attention_mask=torch.ones_like(pt),
                                     max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0,
                                     top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
            finally:
                vref[0] = None
            for o in gen:
                txt = C["prefill"] + tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
                out.append(parse(txt))
            if (s // args.av_batch) % 10 == 0:
                print(f"  verbalize {min(s + args.av_batch, len(acts))}/{len(acts)}", flush=True)
    payload = {"model": args.model, "lines": out}
    json.dump(payload, open(HERE / f"control_expl_{args.model}.json", "w"))
    print(f"[saved] control_expl_{args.model}.json  sample={out[0]}", flush=True)


if __name__ == "__main__":
    main()
