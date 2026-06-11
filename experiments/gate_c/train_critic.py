"""Gate C critic SFT: fine-tune the released L20 AR on (text -> diff vector).

Standalone (no Miles). Direction-only MSE exactly like the repo: both
prediction and target L2-normalized to mse_scale=sqrt(d); loss = MSE.
Initialized from kitft/nla-qwen2.5-7b-L20-ar (backbone + value head).
Logs to wandb (project nla-layer-diff).

Usage:
  python train_critic.py --pairs arm_A.json --targets diff_targets.npy \
      --run-name armA_8k --out ckpt_armA [--epochs 2 --lr 1e-5 ...]

pairs json: [{"text": str, "tidx": int}]  (tidx indexes targets npy rows)
Eval: --eval-pairs/--eval-targets scored every --eval-every steps (cos mean).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic

MSE_SCALE = math.sqrt(3584)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--eval-pairs", default=None)
    ap.add_argument("--eval-targets", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--ar-dir", default=str(WORK / "models" / "ar"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=50)
    return ap.parse_args()


class PairData(torch.utils.data.Dataset):
    def __init__(self, pairs_path, targets, tokenizer, template, max_len):
        self.rows = json.loads(Path(pairs_path).read_text())
        self.targets = targets
        self.tok = tokenizer
        self.template = template
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, k):
        r = self.rows[k]
        prompt = self.template.format(explanation=r["text"])
        ids = self.tok(prompt, add_special_tokens=True, truncation=True,
                       max_length=self.max_len)["input_ids"]
        return ids, np.asarray(self.targets[r["tidx"]], dtype=np.float32)


def collate(batch, pad_id):
    # LEFT-pad so the suffix-anchored extraction at the final position works
    maxlen = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    tgts = torch.zeros((len(batch), 3584), dtype=torch.float32)
    for b, (ids, t) in enumerate(batch):
        x[b, maxlen - len(ids):] = torch.tensor(ids)
        mask[b, maxlen - len(ids):] = 1
        tgts[b] = torch.from_numpy(t)
    return x, mask, tgts


def norm_to(v, s):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * s


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_API_KEY",
                          Path("~/.wandb_key").expanduser().read_text().strip()
                          if Path("~/.wandb_key").expanduser().exists() else "")
    import wandb
    wandb.init(project="nla-layer-diff", name=args.run_name, config=vars(args))

    critic = NLACritic(args.ar_dir, device="cuda", dtype=torch.bfloat16)
    model, head, tok = critic.backbone, critic.value_head, critic.tokenizer
    model.gradient_checkpointing_enable()
    model.train(); head.train()
    pad_id = tok.pad_token_id or tok.eos_token_id

    targets = np.load(args.targets, mmap_mode="r")
    ds = PairData(args.pairs, targets, tok, critic.template, args.max_len)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.micro_batch, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate(b, pad_id), drop_last=True)

    params = list(model.parameters()) + list(head.parameters())
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr)
        print("[train] using PagedAdamW8bit")
    except Exception as e:
        print(f"[train] bitsandbytes unavailable ({e}); fp32 AdamW")
        opt = torch.optim.AdamW(params, lr=args.lr)

    steps_total = args.epochs * len(dl) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(args.warmup, 1)) *
        (0.5 * (1 + math.cos(math.pi * min(s / max(steps_total, 1), 1.0)))))

    eval_set = None
    if args.eval_pairs:
        etgt = np.load(args.eval_targets, mmap_mode="r")
        eval_set = PairData(args.eval_pairs, etgt, tok, critic.template, args.max_len)

    @torch.no_grad()
    def evaluate():
        model.eval(); head.eval()
        coss = []
        for k0 in range(0, min(len(eval_set), 512), args.micro_batch):
            batch = [eval_set[k] for k in range(k0, min(k0 + args.micro_batch, len(eval_set)))]
            x, mask, tgts = collate(batch, pad_id)
            h = model.model(x.cuda(), attention_mask=mask.cuda(),
                            use_cache=False).last_hidden_state[:, -1]
            pred = head(h).float().cpu()
            coss.append(F.cosine_similarity(pred, tgts, dim=-1))
        model.train(); head.train()
        return torch.cat(coss).mean().item()

    step = micro = 0
    for epoch in range(args.epochs):
        for x, mask, tgts in dl:
            h = model.model(x.cuda(), attention_mask=mask.cuda(),
                            use_cache=False).last_hidden_state[:, -1]
            pred = head(h).float()
            loss = F.mse_loss(norm_to(pred, MSE_SCALE),
                              norm_to(tgts.cuda().float(), MSE_SCALE))
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0],
                               "step": step})
                if eval_set is not None and step % args.eval_every == 0:
                    c = evaluate()
                    wandb.log({"eval_cos": c, "step": step})
                    print(f"[train] step {step}/{steps_total} loss {loss.item():.4f} eval_cos {c:.4f}", flush=True)

    final = evaluate() if eval_set is not None else float("nan")
    print(f"[train] FINAL eval_cos {final:.4f}")
    wandb.log({"final_eval_cos": final})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    from safetensors.torch import save_file
    save_file({k: v for k, v in head.state_dict().items()}, str(out / "value_head.safetensors"))
    (out / "RESULT.json").write_text(json.dumps({"final_eval_cos": final, "run": args.run_name}))
    wandb.finish()


if __name__ == "__main__":
    main()
