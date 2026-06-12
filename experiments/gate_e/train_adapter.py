"""Gate E2: linear adapter W (delta -> AV injection embedding), AV frozen.

W is affine (3584->3584 + bias), init alpha*I with alpha = inj_scale /
median ||delta|| (starts at the E1 behavior). Loss: CE on the label tokens
of "<explanation>\nchanged "SRC" to "TGT"\n</explanation>" appended after
the AV's chat-template prompt with W(delta) spliced at the injection slot.
Only W trains; the 7B AV provides gradients w.r.t. its input embedding.

Training data: TRAIN-split positions of the lexical edit types only —
role_reversal is fully held out (the generalization gate). Symmetry
augmentation: (delta, src->tgt) AND (-delta, tgt->src).

Usage: python train_adapter.py --seed 0 --out W_seed0.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLAClient

HELD_OUT_TYPE = "role_reversal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    client = NLAClient(WORK / "models" / "av")  # loads cfg + tokenizer
    cfg = client.cfg
    tok = client.tokenizer
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(WORK / "models" / "av"), torch_dtype=torch.bfloat16,
        device_map="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    embed = model.get_input_embeddings()

    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": cfg.actor_prompt_template.format(
            injection_char=cfg.injection_char)}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    inj_pos = next(p for p in range(1, len(prompt_ids) - 1)
                   if prompt_ids[p] == cfg.injection_token_id)
    prompt_emb = embed(torch.tensor(prompt_ids, device="cuda")).detach()

    meta = json.loads((WORK / "meta_e.json").read_text())
    D = np.load(WORK / "delta20.npy", mmap_mode="r")
    rows = []
    for i, m in enumerate(meta):
        if m["split"] != "train" or m["type"] == HELD_OUT_TYPE:
            continue
        rows.append((i, +1, m["source"], m["target"]))
        rows.append((i, -1, m["target"], m["source"]))  # antisymmetry aug
    rng = np.random.default_rng(args.seed)
    rng.shuffle(rows)
    print(f"[adapter] {len(rows)} training examples", flush=True)

    med = float(np.median(np.linalg.norm(
        np.asarray(D[:2000], dtype=np.float32), axis=1)))
    alpha = cfg.injection_scale / med
    W = torch.eye(3584, device="cuda") * alpha
    W.requires_grad_(True)
    b = torch.zeros(3584, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=args.lr)

    def label_ids(src, tgt):
        text = f'<explanation>\nchanged "{src}" to "{tgt}"\n</explanation>'
        return tok(text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    step = 0
    for ep in range(args.epochs):
        for b0 in range(0, len(rows) - args.batch + 1, args.batch):
            chunk = rows[b0:b0 + args.batch]
            labs = [label_ids(s, t) for _, _, s, t in chunk]
            maxl = max(len(l) for l in labs)
            B = len(chunk)
            seq = prompt_emb.shape[0] + maxl
            embs = torch.zeros(B, seq, 3584, device="cuda", dtype=torch.bfloat16)
            tgt = torch.full((B, seq), -100, device="cuda", dtype=torch.long)
            attn = torch.zeros(B, seq, device="cuda", dtype=torch.long)
            for k, ((i, sgn, _, _), l) in enumerate(zip(chunk, labs)):
                d = torch.tensor(np.asarray(D[i], dtype=np.float32),
                                 device="cuda") * sgn
                pe = prompt_emb.clone()
                pe[inj_pos] = (d @ W + b).to(torch.bfloat16)
                lt = torch.tensor(l, device="cuda")
                embs[k, :len(pe)] = pe
                embs[k, len(pe):len(pe) + len(lt)] = embed(lt).detach()
                tgt[k, len(pe):len(pe) + len(lt)] = lt
                attn[k, :len(pe) + len(lt)] = 1
            out = model(inputs_embeds=embs, attention_mask=attn)
            logits = out.logits[:, :-1].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt[:, 1:].reshape(-1), ignore_index=-100)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"[adapter] ep{ep} step {step} loss {loss.item():.4f}",
                      flush=True)
    np.savez(WORK / args.out, W=W.detach().float().cpu().numpy(),
             b=b.detach().float().cpu().numpy())
    print(f"[adapter] saved {args.out}")


if __name__ == "__main__":
    main()
