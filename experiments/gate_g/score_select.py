"""Gate G Stage 1: score BoN candidates with the conditioned reward critic and
keep the best per position.

reward(text) = cos( critic(cond_state, text), target_state ).  The critic
already has cond_state, so this rewards text for the reconstruction value it
adds beyond it = the diff content. Picks argmax over the N samples.

  python score_select.py --cond acts16.npy --target acts24.npy \
     --candidates cand_h24.jsonl --reward-dir ckpt_g8_text_rm --out best_h24.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import load_file

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic
GC = WORK.parent / "gate_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--reward-dir", required=True)
    ap.add_argument("--ar-dir", default=str(WORK / "models" / "ar"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    critic = NLACritic(args.ar_dir, device="cuda", dtype=torch.bfloat16)
    backbone, tok, template = critic.backbone, critic.tokenizer, critic.template
    from peft import PeftModel
    model = PeftModel.from_pretrained(backbone, args.reward_dir).eval()
    head = critic.value_head
    head.load_state_dict(load_file(str(Path(args.reward_dir) / "value_head.safetensors")))
    head.eval()
    cm = json.loads((Path(args.reward_dir) / "cond_meta.json").read_text())
    marker_id, inj_scale, prefix = cm["marker_id"], cm["inj_scale"], cm["prefix"]
    embed = model.get_input_embeddings()

    cond = np.load(GC / args.cond, mmap_mode="r")
    target = np.load(GC / args.target, mmap_mode="r")

    @torch.no_grad()
    def reward(i, texts):
        # batch the N candidate texts for one position, same injected cond state
        built = []
        for t in texts:
            full = prefix + template.format(explanation=t)
            ids = tok(full, add_special_tokens=True, truncation=True, max_length=320)["input_ids"]
            built.append((ids, ids.index(marker_id)))
        maxl = max(len(ids) for ids, _ in built)
        B = len(built)
        embs = torch.zeros(B, maxl, 3584, device="cuda", dtype=torch.bfloat16)
        attn = torch.zeros(B, maxl, device="cuda", dtype=torch.long)
        v = torch.tensor(np.asarray(cond[i], dtype=np.float32), device="cuda")
        v = (v / v.norm().clamp_min(1e-9) * inj_scale).to(torch.bfloat16)
        for k, (ids, mpos) in enumerate(built):
            off = maxl - len(ids)
            e = embed(torch.tensor(ids, device="cuda")).detach().clone()
            e[mpos] = v
            embs[k, off:] = e
            attn[k, off:] = 1
        h = model(inputs_embeds=embs, attention_mask=attn, use_cache=False).logits[:, -1]
        pred = head(h).float()
        gold = torch.tensor(np.asarray(target[i], dtype=np.float32), device="cuda").unsqueeze(0)
        return F.cosine_similarity(pred, gold, dim=-1).cpu().numpy()

    rows = [json.loads(l) for l in open(args.candidates)]
    fout = open(args.out, "w")
    bests, firsts = [], []
    for n, r in enumerate(rows):
        cands = [s for s in r["samples"] if s.strip()]
        if not cands:
            continue
        rs = reward(r["i"], cands)
        bi = int(np.argmax(rs))
        fout.write(json.dumps({"i": r["i"], "best_text": cands[bi],
                               "best_reward": float(rs[bi]),
                               "mean_reward": float(rs.mean()),
                               "n_cand": len(cands)}) + "\n")
        bests.append(float(rs[bi])); firsts.append(float(rs[0]))
        if n % 500 == 0:
            print(f"[score] {n}/{len(rows)}  best={np.mean(bests):.4f} "
                  f"first={np.mean(firsts):.4f}", flush=True)
    fout.close()
    print(f"[score] done. mean best-of-N reward {np.mean(bests):.4f} vs "
          f"first-sample {np.mean(firsts):.4f} (BoN lift {np.mean(bests)-np.mean(firsts):+.4f})",
          flush=True)


if __name__ == "__main__":
    main()
