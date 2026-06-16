"""Gate G Stage 0: h_20-conditioned critic. Reconstruct h_22 from
(injected h_20  +  explanation text).

The released L20 AR is text->vector. We add ONE conditioning input: h_20,
injected at a marker token (same single-token-embedding trick the AV uses),
scaled to the AV's injection_scale (a proven in-distribution embedding norm).
The critic backbone gets LoRA (r=16, q/k/v/o) so it can learn to read the
injected state; the value head trains fully. Embeddings stay frozen (the
conditioning propagates through attention, not the embedding table).

Loss: direction-only MSE = 2(1-cos), both vectors L2-normed to sqrt(d), the
repo convention. Target = h_22. Same script trains every arm; arms differ
only in the pairs json's "text" field (none / cat / text).

Usage:
  python train_critic_cond.py --pairs arm_text_train.json \
     --eval-pairs arm_text_eval.json --cond acts20.npy --targets acts22.npy \
     --run-name g_text --out ckpt_text
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
import yaml

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import NLACritic

MSE_SCALE = math.sqrt(3584)
GC = WORK.parent / "gate_c"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--eval-pairs", required=True)
    ap.add_argument("--cond", default=str(GC / "acts20.npy"))      # h_20 (injected)
    ap.add_argument("--targets", default=str(GC / "acts22.npy"))   # h_22 (reconstructed)
    ap.add_argument("--ar-dir", default=str(WORK / "models" / "ar"))
    ap.add_argument("--av-dir", default=str(WORK / "models" / "av"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=320)
    ap.add_argument("--eval-every", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-model", action="store_true",
                    help="save LoRA adapter + value head (for use as a reward model)")
    ap.add_argument("--no-cond", action="store_true",
                    help="pure text->vector AR: no injected condition state, no marker")
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.environ.setdefault("WANDB_API_KEY",
                          Path("~/.wandb_key").expanduser().read_text().strip()
                          if Path("~/.wandb_key").expanduser().exists() else "")
    import wandb
    wandb.init(project="nla-layer-diff", name=args.run_name, config=vars(args),
               mode="online" if os.environ.get("WANDB_API_KEY") else "offline")

    # AV sidecar -> marker char/id + injection scale for the conditioning token
    # (skipped for --no-cond: pure text->vector AR needs no AV/marker)
    marker = marker_id = inj_scale = None
    if not args.no_cond:
        avmeta = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
        marker = avmeta["tokens"]["injection_char"]
        marker_id = avmeta["tokens"]["injection_token_id"]
        inj_scale = float(avmeta["extraction"]["injection_scale"])

    critic = NLACritic(args.ar_dir, device="cuda", dtype=torch.bfloat16)
    backbone, head, tok = critic.backbone, critic.value_head, critic.tokenizer
    template = critic.template
    pad_id = tok.pad_token_id or tok.eos_token_id
    if not args.no_cond:
        assert tok.encode(marker, add_special_tokens=False) == [marker_id], "marker drift"

    from peft import LoraConfig, get_peft_model
    backbone.gradient_checkpointing_enable()
    model = get_peft_model(backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    model.train()
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)
    embed = model.get_input_embeddings()
    for p in embed.parameters():
        p.requires_grad_(False)

    cond = None if args.no_cond else np.load(args.cond, mmap_mode="r")
    tgt = np.load(args.targets, mmap_mode="r")
    # --no-cond: pure text->vector AR (no injected condition state, no marker)
    prefix = "" if args.no_cond else f"prior:{marker}\n"   # marker = injection site

    def build_ids(text):
        full = prefix + template.format(explanation=text)
        ids = tok(full, add_special_tokens=True, truncation=True,
                  max_length=args.max_len)["input_ids"]
        mpos = None if args.no_cond else ids.index(marker_id)
        return ids, mpos

    def make_batch(rows):
        built = [build_ids(r["text"]) for r in rows]
        maxl = max(len(ids) for ids, _ in built)
        B = len(rows)
        embs = torch.zeros(B, maxl, 3584, device="cuda", dtype=torch.bfloat16)
        attn = torch.zeros(B, maxl, device="cuda", dtype=torch.long)
        tgts = torch.zeros(B, 3584, dtype=torch.float32)
        for k, ((ids, mpos), r) in enumerate(zip(built, rows)):
            off = maxl - len(ids)                       # LEFT pad
            with torch.no_grad():
                e = embed(torch.tensor(ids, device="cuda")).detach().clone()
                if mpos is not None:                     # inject the condition state
                    v = torch.tensor(np.asarray(cond[r["tidx"]], dtype=np.float32),
                                     device="cuda")
                    v = v / v.norm().clamp_min(1e-9) * inj_scale
                    e[mpos] = v.to(torch.bfloat16)
            embs[k, off:] = e
            attn[k, off:] = 1
            tgts[k] = torch.from_numpy(np.asarray(tgt[r["tidx"]], dtype=np.float32))
        return embs, attn, tgts

    train_rows = json.loads(Path(args.pairs).read_text())
    eval_rows = json.loads(Path(args.eval_pairs).read_text())
    print(f"[cond] {len(train_rows)} train / {len(eval_rows)} eval", flush=True)

    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.PagedAdamW8bit(params, lr=args.lr)
    except Exception as e:
        print(f"[cond] bnb unavailable ({e}); AdamW"); opt = torch.optim.AdamW(params, lr=args.lr)
    steps_total = args.epochs * (len(train_rows) // args.micro_batch) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(args.warmup, 1)) *
        (0.5 * (1 + math.cos(math.pi * min(s / max(steps_total, 1), 1.0)))))

    def fwd(embs, attn):
        out = model(inputs_embeds=embs, attention_mask=attn, use_cache=False)
        return head(out.logits[:, -1])   # lm_head is Identity -> logits == hidden

    @torch.no_grad()
    def evaluate():
        model.eval(); head.eval()
        coss = []
        for k0 in range(0, len(eval_rows), args.micro_batch):
            rows = eval_rows[k0:k0 + args.micro_batch]
            embs, attn, tgts = make_batch(rows)
            pred = fwd(embs, attn).float().cpu()
            coss.append(F.cosine_similarity(pred, tgts, dim=-1))
        model.train(); head.train()
        return torch.cat(coss)   # per-position cos, eval_rows order

    rng = np.random.default_rng(args.seed)
    step = micro = 0
    for ep in range(args.epochs):
        order = rng.permutation(len(train_rows))
        for b0 in range(0, len(order) - args.micro_batch + 1, args.micro_batch):
            rows = [train_rows[i] for i in order[b0:b0 + args.micro_batch]]
            embs, attn, tgts = make_batch(rows)
            pred = fwd(embs, attn).float()
            def n(v): return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9) * MSE_SCALE
            loss = F.mse_loss(n(pred), n(tgts.cuda()))
            (loss / args.accum).backward()
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    wandb.log({"loss": loss.item(), "step": step})
                if step % args.eval_every == 0:
                    c = evaluate().mean().item(); wandb.log({"eval_cos": c, "step": step})
                    print(f"[cond] step {step}/{steps_total} loss {loss.item():.4f} "
                          f"eval_cos {c:.4f}", flush=True)

    ev = evaluate(); final = ev.mean().item()
    print(f"[cond] FINAL eval_cos {final:.4f}", flush=True)
    wandb.log({"final_eval_cos": final})
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "eval_cos.npy", ev.numpy())   # per-position, for paired bootstrap
    np.save(out / "eval_tidx.npy", np.array([r["tidx"] for r in eval_rows]))
    (out / "RESULT.json").write_text(json.dumps(
        {"final_eval_cos": final, "run": args.run_name,
         "n_train": len(train_rows), "n_eval": len(eval_rows)}))
    if args.save_model:
        model.save_pretrained(str(out))                       # LoRA adapter
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in head.state_dict().items()},
                  str(out / "value_head.safetensors"))
        (out / "cond_meta.json").write_text(json.dumps(
            {"marker_id": marker_id, "inj_scale": inj_scale, "prefix": prefix}))
        print(f"[cond] saved reward model (adapter + head) -> {out}", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
