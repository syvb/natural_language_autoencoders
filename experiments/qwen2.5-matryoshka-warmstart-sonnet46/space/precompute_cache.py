"""Precompute the Space's per-token analysis for every default text → precache.json.

The Space serves default-text clicks from this file instantly (no GPU task, no
visitor quota). Regenerate whenever default_texts.json or the checkpoint
changes — easiest via `bash precompute_on_vast.sh` (rents a cheap GPU, runs
this, fetches the result back, destroys the box).

Mirrors app.py's per-click pipeline (KEEP IN SYNC — same sidecar-driven
config, same line splitting, same FVE formula):
  1. EXTRACT   truncated base model, layer-LAYER hidden state at each position
               (forwarded per-prefix, exactly like a click).
  2. VERBALIZE the AV greedy-decodes the injected vector. Batched across
               positions — greedy is prefix-stable, so taking the first
               N_LINES lines of a full 220-token decode equals app.py's
               stop-after-N-lines early exit (up to bf16 batch noise).
  3. RECONSTRUCT critic scores cumulative line prefixes; FVE vs mu.npy.

Needs ~16GB VRAM (models are loaded one stage at a time) and ~45GB of HF
downloads. Runs anywhere CUDA is available:

    python precompute_cache.py --out precache.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_inference import NLACritic

RL_REPO = "syvb/nla-qwen2.5-7b-L20-v3-rl"
RL_ITER = "iter_0000200"
EXTRACTOR_ID = "Qwen/Qwen2.5-7B-Instruct"
N_LINES = 10
MAX_TEXT_TOKENS = 2048
HERE = Path(__file__).resolve().parent


def _normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12) * scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", default=str(HERE / "default_texts.json"))
    ap.add_argument("--mu", default=str(HERE / "mu.npy"))
    ap.add_argument("--out", default=str(HERE / "precache.json"))
    ap.add_argument("--batch", type=int, default=32, help="AV generation batch size")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs a CUDA GPU"

    root = snapshot_download(RL_REPO, allow_patterns=[f"{RL_ITER}/*"])
    av_dir, ar_dir = f"{root}/{RL_ITER}/av", f"{root}/{RL_ITER}/ar"
    av_meta = yaml.safe_load(open(f"{av_dir}/nla_meta.yaml"))
    ar_meta = yaml.safe_load(open(f"{ar_dir}/nla_meta.yaml"))
    tpl = av_meta["prompt_templates"].get("av") or av_meta["prompt_templates"]["actor"]
    inj_id = av_meta["tokens"]["injection_token_id"]
    inj_char = av_meta["tokens"]["injection_char"]
    injection_scale = float(av_meta["extraction"]["injection_scale"])
    layer = (ar_meta.get("critic") or {}).get("extraction_layer_index")
    assert layer is not None, "AR sidecar missing critic.extraction_layer_index"
    assert "<explanation>" not in tpl and "bullet" in tpl, "unexpected AV template"

    tok = AutoTokenizer.from_pretrained(av_dir)
    mu = torch.tensor(np.load(args.mu), dtype=torch.float32)
    texts = json.load(open(args.texts))

    docs = []
    for text in texts:
        ids = tok(text.strip(), add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
        docs.append({"text": text, "ids": ids})
    flat = [(d_i, idx) for d_i, d in enumerate(docs) for idx in range(len(d["ids"]))]
    print(f"{len(docs)} texts, {len(flat)} positions total", flush=True)

    # ── stage 1: extract v at every position (per-prefix, like a click) ──────
    t0 = time.time()
    extractor = AutoModelForCausalLM.from_pretrained(EXTRACTOR_ID, torch_dtype=torch.bfloat16)
    extractor.model.layers = extractor.model.layers[: layer + 1]
    extractor.model.norm = torch.nn.Identity()
    extractor.lm_head = torch.nn.Identity()
    extractor.to("cuda").eval()
    vecs = []
    with torch.inference_mode():
        for n, (d_i, idx) in enumerate(flat):
            ids_t = torch.tensor([docs[d_i]["ids"][: idx + 1]], device="cuda")
            v = extractor.model(ids_t, use_cache=False).last_hidden_state[0, -1]
            vecs.append(v.float().cpu())
            if (n + 1) % 200 == 0:
                print(f"  extract {n + 1}/{len(flat)}", flush=True)
    del extractor
    torch.cuda.empty_cache()
    print(f"extraction done in {time.time() - t0:.0f}s", flush=True)

    # ── stage 2: AV verbalization, batched greedy ─────────────────────────────
    t0 = time.time()
    av = AutoModelForCausalLM.from_pretrained(av_dir, torch_dtype=torch.bfloat16)
    av.to("cuda").eval()
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": tpl.format(injection_char=inj_char)}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    inj_pos = (prompt_ids[0] == inj_id).nonzero(as_tuple=True)[0]
    assert len(inj_pos) == 1, f"marker appears {len(inj_pos)}x in prompt"
    inj_pos = int(inj_pos[0])
    base_emb = av.get_input_embeddings()(prompt_ids.to("cuda"))  # [1, T, d]

    all_lines: list[list[str]] = []
    with torch.inference_mode():
        for s in range(0, len(flat), args.batch):
            chunk = vecs[s : s + args.batch]
            emb = base_emb.repeat(len(chunk), 1, 1).clone()
            for i, v in enumerate(chunk):
                emb[i, inj_pos] = _normalize(v.to("cuda"), injection_scale).to(torch.bfloat16)
            out = av.generate(
                inputs_embeds=emb,
                attention_mask=torch.ones(emb.shape[:2], device="cuda", dtype=torch.long),
                max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id,
            )  # inputs_embeds ⇒ returns generated tokens only
            for text in tok.batch_decode(out, skip_special_tokens=True):
                all_lines.append(
                    [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()][:N_LINES])
            print(f"  verbalize {min(s + args.batch, len(flat))}/{len(flat)}", flush=True)
    del av, base_emb
    torch.cuda.empty_cache()
    print(f"verbalization done in {time.time() - t0:.0f}s", flush=True)

    # ── stage 3: critic FVE per cumulative line prefix ────────────────────────
    t0 = time.time()
    critic = NLACritic(ar_dir, device="cuda")
    mse_scale = critic.mse_scale
    results: list[dict | None] = []
    for n, ((d_i, idx), v, lines) in enumerate(zip(flat, vecs, all_lines)):
        if not lines:
            results.append(None)  # app computes live (and shows its empty-card)
            continue
        gold_n = _normalize(v, mse_scale)
        denom = ((gold_n - mu) ** 2).mean().item()
        preds = critic.reconstruct_batch(
            ["\n".join(lines[:k]) for k in range(1, len(lines) + 1)])
        fve, cos = [], []
        for pred in preds:
            pred_n = _normalize(pred, mse_scale)
            fve.append(round(1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom, 6))
            cos.append(round(float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm())), 6))
        results.append({"lines": lines, "fve": fve, "cos": cos})
        if (n + 1) % 200 == 0:
            print(f"  score {n + 1}/{len(flat)}", flush=True)
    print(f"scoring done in {time.time() - t0:.0f}s", flush=True)

    entries = []
    pos = 0
    for d in docs:
        n = len(d["ids"])
        entries.append({"text": d["text"], "ids": d["ids"],
                        "results": results[pos : pos + n]})
        pos += n
    payload = {
        "meta": {
            "rl_repo": RL_REPO, "rl_iter": RL_ITER, "extractor": EXTRACTOR_ID,
            "n_lines": N_LINES, "gpu": torch.cuda.get_device_name(0),
            "generated_unix": int(time.time()),
        },
        "entries": entries,
    }
    Path(args.out).write_text(json.dumps(payload))
    n_ok = sum(r is not None for r in results)
    print(f"wrote {args.out}: {n_ok}/{len(flat)} positions "
          f"({Path(args.out).stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
