"""Gate G Stage 1: batched AV sampling via HF generate (NOT sglang).

sglang's input_embeds path caps ~3-4/s; batched generate on one H100 does
~50-100 seq/s, which is what makes BoN-SFT affordable. Injects a target state
(SRC_ACTS) into the AV prompt and samples N completions per position.

  python sample_av.py --src acts24.npy --split train --n 8 --temp 0.8 \
     --out cand_h24.jsonl [--adapter av_bon_lora] [--limit 200]

Output jsonl: {"i": int, "samples": [str, ...]}  (explanation-extracted).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
GC = WORK.parent / "gate_c"
EXPL = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)              # acts file to inject (target state)
    ap.add_argument("--split", default="train")          # train/holdout/all
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)       # 0 = all
    ap.add_argument("--adapter", default=None)            # LoRA dir (SFT'd AV)
    ap.add_argument("--av-dir", default=str(WORK / "models" / "av"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta = json.loads((GC / "meta2.json").read_text())
    am = yaml.safe_load((Path(args.av_dir) / "nla_meta.yaml").read_text())
    inj_char, inj_id = am["tokens"]["injection_char"], am["tokens"]["injection_token_id"]
    inj_scale = float(am["extraction"]["injection_scale"])
    template = am["prompt_templates"].get("av") or am["prompt_templates"]["actor"]

    tok = AutoTokenizer.from_pretrained(args.av_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.av_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).eval()

    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": template.format(injection_char=inj_char)}],
        tokenize=True, add_generation_prompt=True, return_dict=False)
    inj_pos = prompt_ids.index(inj_id)
    embed = model.get_input_embeddings()
    with torch.no_grad():
        prompt_emb = embed(torch.tensor(prompt_ids, device="cuda")).detach()  # [L,d]
    L = prompt_emb.shape[0]

    src = np.load(GC / args.src, mmap_mode="r")
    if args.split == "all":
        idxs = list(range(len(meta)))
    else:
        idxs = [i for i, m in enumerate(meta) if m["split"] == args.split]
    if args.limit:
        idxs = idxs[:args.limit]

    done = set()
    outp = WORK / args.out
    if outp.exists():
        for line in outp.open():
            done.add(json.loads(line)["i"])
    idxs = [i for i in idxs if i not in done]
    print(f"[sample] {len(idxs)} positions, N={args.n}, temp={args.temp}", flush=True)

    import time
    t0 = time.time()
    f = outp.open("a")
    for b0 in range(0, len(idxs), args.batch):
        bi = idxs[b0:b0 + args.batch]
        B = len(bi)
        embs = prompt_emb.unsqueeze(0).repeat(B, 1, 1).clone()
        for k, i in enumerate(bi):
            v = torch.tensor(np.asarray(src[i], dtype=np.float32), device="cuda")
            v = v / v.norm().clamp_min(1e-9) * inj_scale
            embs[k, inj_pos] = v.to(torch.bfloat16)
        attn = torch.ones(B, L, device="cuda", dtype=torch.long)
        with torch.no_grad():
            gen = model.generate(
                inputs_embeds=embs, attention_mask=attn, do_sample=True,
                temperature=args.temp, top_p=args.top_p, max_new_tokens=args.max_new,
                num_return_sequences=args.n, pad_token_id=tok.eos_token_id)
        texts = tok.batch_decode(gen, skip_special_tokens=True)  # [B*N]
        for k, i in enumerate(bi):
            sams = []
            for j in range(args.n):
                t = texts[k * args.n + j]
                m = EXPL.search(t)
                sams.append(m.group(1).strip() if m else "")
            f.write(json.dumps({"i": i, "samples": sams}) + "\n")
        f.flush()
        n_done = b0 + B
        if (b0 // args.batch) % 5 == 0:
            rate = n_done * args.n / (time.time() - t0)
            print(f"[sample] {n_done}/{len(idxs)} positions  {rate:.1f} seq/s", flush=True)
    f.close()
    print(f"[sample] done {len(idxs)} in {time.time()-t0:.0f}s "
          f"({len(idxs)*args.n/(time.time()-t0):.1f} seq/s)", flush=True)


if __name__ == "__main__":
    main()
