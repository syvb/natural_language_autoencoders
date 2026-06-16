"""Gate G AV2 BoN: score candidates with the TWO-TEXT AR2 reward model.

reward(candidate) = cos( AR2("[before] AV(h16) [after] candidate"), h24 ).
AR2 already holds the before-description, so this rewards the candidate for the
reconstruction value it adds BEYOND it = the diff. Pure text->vector, no injection.

  python score_select_text.py --target acts24.npy --candidates cand_av2.jsonl \
     --before av_h16_out --reward-dir reward_ar2 --out best_av2.jsonl
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic
GC = WORK.parent / "gate_c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--before", required=True)       # av dir with before-state (h16) text
    ap.add_argument("--reward-dir", required=True)
    ap.add_argument("--ar-dir", default=str(WORK / "models" / "ar"))
    ap.add_argument("--lam", type=float, default=0.0,
                    help="length penalty: select by recon - lam*(words/100)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    critic = NLACritic(args.ar_dir, device="cuda", dtype=torch.bfloat16)
    backbone, tok, template = critic.backbone, critic.tokenizer, critic.template
    pad_id = tok.pad_token_id or tok.eos_token_id
    from peft import PeftModel
    model = PeftModel.from_pretrained(backbone, args.reward_dir).eval()
    head = critic.value_head
    head.load_state_dict(load_file(str(Path(args.reward_dir) / "value_head.safetensors")))
    head.eval()

    before = {}
    for fn in glob.glob(f"{args.before}/*.jsonl"):
        for l in open(fn):
            r = json.loads(l); before[r["i"]] = r.get("explanation") or ""
    target = np.load(GC / args.target, mmap_mode="r")

    @torch.no_grad()
    def reward(i, cands):
        texts = [f"[before] {before.get(i,'')} [after] {c}" for c in cands]
        ids = [tok(template.format(explanation=t), add_special_tokens=True,
                   truncation=True, max_length=512)["input_ids"] for t in texts]
        maxl = max(len(x) for x in ids); B = len(ids)
        x = torch.full((B, maxl), pad_id, dtype=torch.long)
        m = torch.zeros(B, maxl, dtype=torch.long)
        for k, v in enumerate(ids):
            x[k, maxl - len(v):] = torch.tensor(v); m[k, maxl - len(v):] = 1
        h = model(input_ids=x.cuda(), attention_mask=m.cuda(), use_cache=False).logits[:, -1]
        pred = head(h).float()
        gold = torch.tensor(np.asarray(target[i], dtype=np.float32), device="cuda").unsqueeze(0)
        return F.cosine_similarity(pred, gold, dim=-1).cpu().numpy()

    rows = [json.loads(l) for l in open(args.candidates)]
    fout = open(args.out, "w")
    bests, firsts, sel_len = [], [], []
    for n, r in enumerate(rows):
        cands = [s for s in r["samples"] if s.strip()]
        if not cands:
            continue
        rs = reward(r["i"], cands)
        wlen = np.array([len(c.split()) for c in cands], dtype=np.float32)
        bi = int(np.argmax(rs - args.lam * wlen / 100.0))   # length-penalized selection
        fout.write(json.dumps({"i": r["i"], "best_text": cands[bi],
                               "best_reward": float(rs[bi]), "mean_reward": float(rs.mean()),
                               "best_words": int(wlen[bi])}) + "\n")
        bests.append(float(rs[bi])); firsts.append(float(rs[0])); sel_len.append(int(wlen[bi]))
        if n % 500 == 0:
            print(f"[score-text] {n}/{len(rows)} lam={args.lam} recon(best)={np.mean(bests):.4f} "
                  f"sel_words={np.mean(sel_len):.0f}", flush=True)
    fout.close()
    print(f"[score-text] done lam={args.lam}. recon(best) {np.mean(bests):.4f} vs first "
          f"{np.mean(firsts):.4f}; mean selected length {np.mean(sel_len):.0f} words", flush=True)


if __name__ == "__main__":
    main()
