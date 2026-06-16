"""Gate G GRPO: RL the AV policy toward concise, diff-focused descriptions.

For each position, inject h24 into the policy AV and sample a group of G
descriptions. Reward each:
    r = cos(AR2("[before] AV(h16) [after] gen"), h24) - lam * words/100
Group-relative advantage A = (r - mean)/(std+eps). Update the policy LoRA:
    loss = -(A * mean_token_logp) + beta * KL(policy || frozen-AV)
KL keeps it linguistic; lam forces concision; the before-conditioned recon makes
the surviving tokens be the diff. Logs recon+length+KL per step (the frontier).

  python grpo_av.py --reward-dir reward_ar2 --lam 0.05 --steps 250 --out grpo_av_lora
"""
import argparse, glob, json, os, re, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

WORK = Path(__file__).parent; sys.path.insert(0, str(WORK))
from nla_inference import NLACritic
GC = WORK.parent / "gate_c"
EXPL = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward-dir", required=True)
    ap.add_argument("--src", default="acts24.npy")
    ap.add_argument("--target", default="acts24.npy")
    ap.add_argument("--before", default="av_h16_out")
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--ar-lr", type=float, default=1e-4)   # AR2 co-training lr
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2)         # positions/step (OOM-safe, M2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--max-new", type=int, default=180)
    ap.add_argument("--av-dir", default=str(WORK / "models" / "av"))
    ap.add_argument("--ar-dir", default=str(WORK / "models" / "ar"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    os.environ.setdefault("WANDB_API_KEY",
        Path("~/.wandb_key").expanduser().read_text().strip() if Path("~/.wandb_key").expanduser().exists() else "")
    import wandb
    wandb.init(project="nla-layer-diff", name=f"grpo_lam{args.lam}", config=vars(args),
               mode="online" if os.environ.get("WANDB_API_KEY") else "offline")

    am = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    inj_char, inj_id = am["tokens"]["injection_char"], am["tokens"]["injection_token_id"]
    inj_scale = float(am["extraction"]["injection_scale"])
    template_av = am["prompt_templates"].get("av") or am["prompt_templates"]["actor"]
    tok = AutoTokenizer.from_pretrained(args.av_dir, trust_remote_code=True)
    PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    from peft import LoraConfig, get_peft_model, PeftModel
    base = AutoModelForCausalLM.from_pretrained(args.av_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    # Qwen-Instruct stops on <|im_end|>, not the stock eos — collect ALL terminators
    # so generated-length detection (build_embeds) and generate() are correct (B1 fix).
    _eos = getattr(base.generation_config, "eos_token_id", None) or tok.eos_token_id
    TERM = set(_eos if isinstance(_eos, (list, tuple)) else [_eos]); TERM.add(tok.eos_token_id)
    print(f"[grpo] eos/terminators={sorted(TERM)} pad={PAD} "
          f"tok.eos={tok.eos_token_id} im_end={tok.convert_tokens_to_ids('<|im_end|>')}", flush=True)
    policy = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
    embed = policy.get_input_embeddings()
    prompt_ids = tok.apply_chat_template([{"role": "user",
        "content": template_av.format(injection_char=inj_char)}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    inj_pos = prompt_ids.index(inj_id); P = len(prompt_ids)
    with torch.no_grad():
        prompt_emb = embed(torch.tensor(prompt_ids, device="cuda")).detach()

    # AR2 reward model CO-TRAINS with the policy (NLA autoencoder; frozen AR is
    # hackable AND can't read the concise code we're pushing toward). Warm-start
    # from reward_ar2; target stays the REAL h24, so it can't collude.
    critic = NLACritic(args.ar_dir, device="cuda", dtype=torch.bfloat16)
    rm = PeftModel.from_pretrained(critic.backbone, args.reward_dir, is_trainable=True).eval()
    head = critic.value_head; head.load_state_dict(load_file(str(Path(args.reward_dir) / "value_head.safetensors")))
    for p in head.parameters(): p.requires_grad_(True)
    rm_opt = torch.optim.AdamW([p for p in rm.parameters() if p.requires_grad] + list(head.parameters()), lr=args.ar_lr)
    rtok, rtemplate = critic.tokenizer, critic.template
    rpad = rtok.pad_token_id or rtok.eos_token_id

    before = {}
    for fn in glob.glob(f"{args.before}/*.jsonl"):
        for l in open(fn): r = json.loads(l); before[r["i"]] = r.get("explanation") or ""
    src = np.load(GC / args.src, mmap_mode="r"); target = np.load(GC / args.target, mmap_mode="r")
    meta = json.loads((GC / "meta2.json").read_text())
    train_pos = [i for i, m in enumerate(meta) if m["split"] == "train" and before.get(i, "").strip()]
    eval_pos = [i for i, m in enumerate(meta) if m["split"] == "holdout" and before.get(i, "").strip()][:64]
    rng = np.random.default_rng(0)
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr)

    def inj_vec(i):
        v = torch.tensor(np.asarray(src[i], dtype=np.float32), device="cuda")
        return (v / v.norm().clamp_min(1e-9) * inj_scale).to(torch.bfloat16)

    @torch.no_grad()
    def rollout(positions):
        policy.gradient_checkpointing_disable(); policy.config.use_cache = True   # fast cached gen
        B = len(positions)
        embs = prompt_emb.unsqueeze(0).repeat(B, 1, 1).clone()
        for k, i in enumerate(positions): embs[k, inj_pos] = inj_vec(i)
        attn = torch.ones(B, P, device="cuda", dtype=torch.long)
        gen = policy.generate(inputs_embeds=embs, attention_mask=attn, do_sample=True,
            temperature=args.temp, top_p=0.95, max_new_tokens=args.max_new,
            num_return_sequences=args.group, pad_token_id=PAD, eos_token_id=sorted(TERM))
        return gen  # [B*G, T] generated token ids

    @torch.no_grad()
    def reward(positions, gen):
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        BG = len(texts); recon = np.zeros(BG, np.float32); wlen = np.zeros(BG, np.float32)
        ar_ids = []; expls = []
        for j, t in enumerate(texts):
            m = EXPL.search(t); e = m.group(1).strip() if m else ""
            expls.append(e)
            wlen[j] = len(e.split()) if e else max(len(t.split()), 1)
            i = positions[j // args.group]
            ar_ids.append(rtok(rtemplate.format(explanation=f"[before] {before.get(i,'')} [after] {e}"),
                add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
        for s in range(0, BG, 32):
            ch = ar_ids[s:s + 32]; ml = max(len(x) for x in ch); b = len(ch)
            x = torch.full((b, ml), rpad, dtype=torch.long); mm = torch.zeros(b, ml, dtype=torch.long)
            for k, v in enumerate(ch): x[k, ml - len(v):] = torch.tensor(v); mm[k, ml - len(v):] = 1
            h = rm(input_ids=x.cuda(), attention_mask=mm.cuda(), use_cache=False).logits[:, -1]
            pred = head(h).float()
            for k in range(b):
                i = positions[(s + k) // args.group]
                g = torch.tensor(np.asarray(target[i], dtype=np.float32), device="cuda")
                recon[s + k] = F.cosine_similarity(pred[k:k + 1], g.unsqueeze(0), dim=-1).item()
        return recon, wlen, expls

    def ar2_update(positions, expls):   # co-train AR2 to reconstruct REAL h24 from current AV text
        ids = [rtok(rtemplate.format(explanation=f"[before] {before.get(positions[j//args.group],'')} [after] {e}"),
                    add_special_tokens=True, truncation=True, max_length=512)["input_ids"]
               for j, e in enumerate(expls)]
        tot = nb = 0
        for s in range(0, len(ids), 16):
            ch = ids[s:s + 16]; ml = max(len(x) for x in ch); b = len(ch)
            x = torch.full((b, ml), rpad, dtype=torch.long); mm = torch.zeros(b, ml, dtype=torch.long)
            tg = torch.zeros(b, 3584)
            for k, v in enumerate(ch):
                x[k, ml - len(v):] = torch.tensor(v); mm[k, ml - len(v):] = 1
                tg[k] = torch.tensor(np.asarray(target[positions[(s + k) // args.group]], dtype=np.float32))
            h = rm(input_ids=x.cuda(), attention_mask=mm.cuda(), use_cache=False).logits[:, -1]
            loss = (1 - F.cosine_similarity(head(h).float(), tg.cuda(), dim=-1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in rm.parameters() if p.requires_grad] + list(head.parameters()), 1.0)
            rm_opt.step(); rm_opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        return tot / max(nb, 1)

    def build_embeds(positions, gen):
        BG = gen.shape[0]; lens = []
        for j in range(BG):
            g = gen[j].tolist()
            stop = next((k for k, t in enumerate(g) if t in TERM), None)   # first terminator
            lens.append(stop + 1 if stop is not None else len(g))
        maxT = max(lens); L = P + maxT
        embs = torch.zeros(BG, L, 3584, device="cuda", dtype=torch.bfloat16)
        attn = torch.zeros(BG, L, device="cuda", dtype=torch.long)
        lab = torch.full((BG, L), -100, device="cuda", dtype=torch.long)
        for j in range(BG):
            i = positions[j // args.group]; T = lens[j]
            pe = prompt_emb.clone(); pe[inj_pos] = inj_vec(i)
            gt = gen[j, :T]
            with torch.no_grad():
                ge = embed(gt).detach()
            embs[j, :P] = pe; embs[j, P:P + T] = ge; attn[j, :P + T] = 1; lab[j, P:P + T] = gt
        return embs, attn, lab

    def token_logp(embs, attn, lab, grad):
        def run():
            out = policy(inputs_embeds=embs, attention_mask=attn, use_cache=False)
            lp = -F.cross_entropy(out.logits[:, :-1].float().reshape(-1, out.logits.shape[-1]),
                lab[:, 1:].reshape(-1), ignore_index=-100, reduction="none").reshape(lab.shape[0], -1)
            mask = (lab[:, 1:] != -100).float()
            return lp, mask
        if grad:
            policy.gradient_checkpointing_enable(); policy.config.use_cache = False  # memory-safe bwd
            return run()
        policy.gradient_checkpointing_disable()                       # no bwd -> no gc (m1)
        with torch.no_grad():
            with policy.disable_adapter():
                return run()

    @torch.no_grad()
    def fixed_eval(step):   # greedy 1/pos on FIXED holdout set -> clean recon/words trend + sample dump
        policy.gradient_checkpointing_disable(); policy.config.use_cache = True
        recs, wds, expl_all = [], [], []
        for s in range(0, len(eval_pos), args.batch):
            chunk = eval_pos[s:s + args.batch]
            embs = prompt_emb.unsqueeze(0).repeat(len(chunk), 1, 1).clone()
            for k, i in enumerate(chunk): embs[k, inj_pos] = inj_vec(i)
            attn = torch.ones(len(chunk), P, device="cuda", dtype=torch.long)
            gen = policy.generate(inputs_embeds=embs, attention_mask=attn, do_sample=False,
                max_new_tokens=args.max_new, num_return_sequences=1, pad_token_id=PAD, eos_token_id=sorted(TERM))
            texts = tok.batch_decode(gen, skip_special_tokens=True); ar_ids = []
            for k, t in enumerate(texts):
                m = EXPL.search(t); e = m.group(1).strip() if m else ""
                expl_all.append(e)
                wds.append(len(e.split()) if e else max(len(t.split()), 1))
                ar_ids.append(rtok(rtemplate.format(explanation=f"[before] {before.get(chunk[k],'')} [after] {e}"),
                    add_special_tokens=True, truncation=True, max_length=512)["input_ids"])
            ml = max(len(x) for x in ar_ids); b = len(ar_ids)
            x = torch.full((b, ml), rpad, dtype=torch.long); mm = torch.zeros(b, ml, dtype=torch.long)
            for k, v in enumerate(ar_ids): x[k, ml - len(v):] = torch.tensor(v); mm[k, ml - len(v):] = 1
            h = rm(input_ids=x.cuda(), attention_mask=mm.cuda(), use_cache=False).logits[:, -1]
            pred = head(h).float()
            for k in range(b):
                g = torch.tensor(np.asarray(target[chunk[k]], dtype=np.float32), device="cuda")
                recs.append(F.cosine_similarity(pred[k:k + 1], g.unsqueeze(0), dim=-1).item())
        with open("grpo_samples.jsonl", "a") as f:    # dump first few descriptions for inspection
            f.write(json.dumps({"step": step, "mean_words": float(np.mean(wds)),
                "samples": [{"pos": eval_pos[j], "words": wds[j], "recon": round(recs[j], 4), "text": expl_all[j]}
                            for j in range(min(4, len(expl_all)))]}) + "\n")
        return float(np.mean(recs)), float(np.mean(wds))

    print(f"[grpo] {len(train_pos)} train / {len(eval_pos)} fixed-eval positions, "
          f"lam={args.lam} beta={args.beta} G={args.group} B={args.batch}", flush=True)
    t0 = time.time()
    for step in range(args.steps):
        positions = rng.choice(train_pos, args.batch, replace=False).tolist()
        gen = rollout(positions)
        recon, wlen, expls = reward(positions, gen)
        recon = np.nan_to_num(recon, nan=0.0)        # guard degenerate cosine (m2)
        tag_fail = float(np.mean([0.0 if e.strip() else 1.0 for e in expls]))  # no <explanation>
        rew = recon - args.lam * wlen / 100.0
        # group-relative advantage; zero groups with no real spread (M1: avoid
        # blowing up tiny-std groups into spurious +-10 advantages)
        adv = np.zeros_like(rew)
        for b in range(args.batch):
            sl = slice(b * args.group, (b + 1) * args.group)
            g = rew[sl]; gs = g.std()
            adv[sl] = (g - g.mean()) / gs if gs > 1e-3 else 0.0
        adv_t = torch.tensor(adv, device="cuda", dtype=torch.float32)
        embs, attn, lab = build_embeds(positions, gen)
        lp, mask = token_logp(embs, attn, lab, grad=True)
        ref_lp, _ = token_logp(embs, attn, lab, grad=False)
        ntok = mask.sum(1).clamp_min(1)
        seq_lp = (lp * mask).sum(1) / ntok
        logr = (ref_lp - lp).clamp(-10, 10)
        kl = ((torch.exp(logr) - logr - 1) * mask).sum(1) / ntok       # k3, per seq
        pg = -(adv_t * seq_lp).mean()
        loss = pg + args.beta * kl.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        ar2_loss = ar2_update(positions, expls)   # co-adapt the reward model
        if step % 5 == 0 or step == args.steps - 1:
            log = {"step": step, "recon": float(recon.mean()), "words": float(wlen.mean()),
                   "reward": float(rew.mean()), "kl": float(kl.mean().item()),
                   "ar2_loss": float(ar2_loss), "tag_fail": tag_fail,
                   "rate": (step + 1) / (time.time() - t0)}
            wandb.log(log)
            print(f"[grpo] {step}/{args.steps} recon {log['recon']:.4f} words {log['words']:.0f} "
                  f"kl {log['kl']:.3f} ar2 {log['ar2_loss']:.3f} tagfail {tag_fail:.2f} "
                  f"({log['rate']:.2f} it/s)", flush=True)
        if step % 20 == 0 or step == args.steps - 1:
            er, ew = fixed_eval(step)
            wandb.log({"eval_recon": er, "eval_words": ew, "step": step})
            print(f"[grpo]   FIXED-EVAL {step}: recon {er:.4f} words {ew:.1f}", flush=True)
    policy.save_pretrained(args.out)
    rm.save_pretrained(args.out + "_ar2")
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in head.state_dict().items()},
              str(Path(args.out + "_ar2") / "value_head.safetensors"))
    print(f"[grpo] saved {args.out} + co-trained AR2", flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
