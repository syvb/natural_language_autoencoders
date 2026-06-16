"""Gate G Stage 1: BoN-SFT the AV. Teach the AV to emit the best-of-N
(reconstruction-rewarded) description when the target state is injected.

LoRA on the AV's attention projections; inject the target state at the AV's
marker; CE on the selected best_text. Saves a LoRA adapter for sample_av.py.

  python sft_av.py --src acts24.npy --best best_h24.jsonl --out av_bon_lora
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
GC = WORK.parent / "gate_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)         # injected target state (e.g. acts24)
    ap.add_argument("--best", required=True)        # best_*.jsonl from score_select
    ap.add_argument("--out", required=True)
    ap.add_argument("--av-dir", default=str(WORK / "models" / "av"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    am = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    inj_char, inj_id = am["tokens"]["injection_char"], am["tokens"]["injection_token_id"]
    inj_scale = float(am["extraction"]["injection_scale"])
    template = am["prompt_templates"].get("av") or am["prompt_templates"]["actor"]
    tok = AutoTokenizer.from_pretrained(args.av_dir, trust_remote_code=True)

    from peft import LoraConfig, get_peft_model
    base = AutoModelForCausalLM.from_pretrained(
        args.av_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model = get_peft_model(base, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.train()
    embed = model.get_input_embeddings()

    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": template.format(injection_char=inj_char)}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    inj_pos = prompt_ids.index(inj_id)
    with torch.no_grad():
        prompt_emb = embed(torch.tensor(prompt_ids, device="cuda")).detach()

    src = np.load(GC / args.src, mmap_mode="r")
    rows = [json.loads(l) for l in open(args.best)]
    rng = np.random.default_rng(args.seed)
    print(f"[sft_av] {len(rows)} (state,best_text) pairs", flush=True)

    def label_ids(text):
        body = text if text.startswith("<explanation>") else f"<explanation>\n{text}\n</explanation>"
        return tok(body, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    P = prompt_emb.shape[0]
    step = 0
    for ep in range(args.epochs):
        order = rng.permutation(len(rows))
        for b0 in range(0, len(order) - args.batch + 1, args.batch):
            chunk = [rows[i] for i in order[b0:b0 + args.batch]]
            labs = [label_ids(r["best_text"]) for r in chunk]
            maxl = max(len(l) for l in labs); B = len(chunk); seq = P + maxl
            embs = torch.zeros(B, seq, 3584, device="cuda", dtype=torch.bfloat16)
            tgt = torch.full((B, seq), -100, device="cuda", dtype=torch.long)
            attn = torch.zeros(B, seq, device="cuda", dtype=torch.long)
            for k, (r, l) in enumerate(zip(chunk, labs)):
                pe = prompt_emb.clone()
                v = torch.tensor(np.asarray(src[r["i"]], dtype=np.float32), device="cuda")
                pe[inj_pos] = (v / v.norm().clamp_min(1e-9) * inj_scale).to(torch.bfloat16)
                lt = torch.tensor(l, device="cuda")
                with torch.no_grad():
                    le = embed(lt).detach()
                embs[k, :P] = pe; embs[k, P:P + len(lt)] = le
                tgt[k, P:P + len(lt)] = lt; attn[k, :P + len(lt)] = 1
            out = model(inputs_embeds=embs, attention_mask=attn)
            logits = out.logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % 25 == 0:
                print(f"[sft_av] ep{ep} step {step} loss {loss.item():.4f}", flush=True)
    model.save_pretrained(args.out)
    print(f"[sft_av] saved adapter -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
