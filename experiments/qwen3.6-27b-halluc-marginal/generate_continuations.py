"""GPU-side: generate the model's OWN continuation from each context's activation
position, so the faithfulness judge can see what the model actually does next
(the activation encodes the model's predictions, which the corpus continuation
may not match). One greedy + K sampled(T=1) continuations of ~24 tokens each,
starting from the verbatim prefix_ids (activation is at prefix_ids[-1]).

    python generate_continuations.py --out results/model_continuations.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
BASE_ID = "Qwen/Qwen3.6-27B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(HERE.parent / "qwen3.6-27b-suffix-eval" / "data" / "manifest.json"))
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--samples", type=int, default=3)   # T=1 samples in addition to greedy
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results" / "model_continuations.json"))
    args = ap.parse_args()
    assert torch.cuda.is_available()

    contexts = json.load(open(args.manifest))["contexts"]
    if args.limit:
        contexts = contexts[: args.limit]
    tok = AutoTokenizer.from_pretrained(args.base)
    print(f"[load] {args.base} bf16…", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0}).eval()

    out = []
    t0 = time.time()
    with torch.inference_mode():
        for i, c in enumerate(contexts):
            ids = torch.tensor([c["prefix_ids"]], device="cuda")
            am = torch.ones_like(ids)
            g = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=args.max_new,
                               do_sample=False, pad_token_id=tok.eos_token_id)
            greedy = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
            samples = []
            for s in range(args.samples):
                torch.manual_seed(args.seed * 100 + s)
                gs = model.generate(input_ids=ids, attention_mask=am, max_new_tokens=args.max_new,
                                    do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                    pad_token_id=tok.eos_token_id)
                samples.append(tok.decode(gs[0, ids.shape[1]:], skip_special_tokens=True))
            out.append({"ci": c["ci"], "greedy": greedy, "samples": samples})
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(contexts)} ({time.time()-t0:.0f}s)", flush=True)
    Path(args.out).write_text(json.dumps(out))
    print(f"[write] {args.out} ({len(out)} contexts)", flush=True)
    print("GEN_DONE", flush=True)
    print("sample[0] greedy:", out[0]["greedy"][:160], flush=True)


if __name__ == "__main__":
    main()
