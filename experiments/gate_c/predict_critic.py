"""Export held-out predictions from a trained Gate C critic checkpoint.

Loads NLACritic from a ckpt dir produced by train_critic.py (nla_meta.yaml
must be copied in from models/ar — train_critic doesn't save it), runs the
eval pair file batched with LEFT padding (same convention as training), and
saves preds_<name>.npy [n, 3584] fp16 plus tidx_<name>.npy (row order).

Usage:
  python predict_critic.py --ckpt ckpt_armA_8k --pairs armA_eval.json \
      --name armA_8k [--micro-batch 16 --max-len 320]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).parent
import sys
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=320)
    args = ap.parse_args()

    critic = NLACritic(args.ckpt, device="cuda", dtype=torch.bfloat16)
    model, head, tok = critic.backbone, critic.value_head, critic.tokenizer
    pad_id = tok.pad_token_id or tok.eos_token_id
    rows = json.loads(Path(args.pairs).read_text())

    preds = []
    with torch.inference_mode():
        for k0 in range(0, len(rows), args.micro_batch):
            chunk = rows[k0:k0 + args.micro_batch]
            ids_list = [tok(critic.template.format(explanation=r["text"]),
                            add_special_tokens=True, truncation=True,
                            max_length=args.max_len)["input_ids"]
                        for r in chunk]
            maxlen = max(len(ids) for ids in ids_list)
            x = torch.full((len(chunk), maxlen), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
            for b, ids in enumerate(ids_list):
                x[b, maxlen - len(ids):] = torch.tensor(ids)
                mask[b, maxlen - len(ids):] = 1
            h = model.model(x.cuda(), attention_mask=mask.cuda(),
                            use_cache=False).last_hidden_state[:, -1]
            preds.append(head(h).float().cpu().numpy())
            if (k0 // args.micro_batch) % 20 == 0:
                print(f"[predict] {k0}/{len(rows)}", flush=True)

    preds = np.concatenate(preds).astype(np.float16)
    np.save(WORK / f"preds_{args.name}.npy", preds)
    np.save(WORK / f"tidx_{args.name}.npy",
            np.array([r["tidx"] for r in rows], dtype=np.int64))
    print(f"[predict] saved preds_{args.name}.npy {preds.shape}")


if __name__ == "__main__":
    main()
