---
title: NLA v3 — Activation Explorer
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
license: apache-2.0
short_description: Click a token, read its activation in plain English
models:
  - syvb/nla-qwen2.5-7b-L20-v3-rl
  - Qwen/Qwen2.5-7B-Instruct
---

# NLA v3 — read a language model's mind, one token at a time

Interactive demo of a **natural-language autoencoder** (NLA) for Qwen2.5-7B
layer-20 activations:

- **AV (actor)** — a fine-tuned Qwen2.5-7B that verbalizes an injected
  activation vector into a salience-ordered list of short descriptions.
- **AR (critic)** — a 21-layer truncated Qwen2.5-7B + linear head that
  reconstructs the activation from text.

Click any token of any text: the Space extracts the layer-20 hidden state at
that position, verbalizes it into 9 lines, then reconstructs the vector from
cumulative line prefixes and plots **FVE** (fraction of variance explained) per
line — marginal or cumulative. The v3 pair was RL-trained with random
token-truncation (~U[1,120]) so the actor front-loads the most
reconstruction-relevant content.

Checkpoints: [syvb/nla-qwen2.5-7b-L20-v3-rl](https://huggingface.co/syvb/nla-qwen2.5-7b-L20-v3-rl)
(iter 200). Code: [natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders).

---

## Development (this folder is the Space source, checked into the repo)

- `app.py` — the Gradio ZeroGPU app.
- `requirements.txt`, `README.md` — Space config (the front-matter above is the card).
- `mu.npy`, `default_texts.json` — data assets; regenerate with `build_assets.py`.
- `build_assets.py` — rebuilds the two data assets from the v3 held-out eval set.
- `precache.json` — baked per-position results (lines + FVE/cos) for every token
  of every default text; the app serves these instantly with no GPU task.
- `precompute_cache.py` — regenerates `precache.json` on any CUDA box (~16GB
  VRAM, models staged sequentially). Mirrors `app.py`'s pipeline — keep in sync.
- `precompute_on_vast.sh` — one command to run the above on a cheap rented
  Vast.ai GPU and fetch the result back (~$0.5, ~30-60 min). Run it whenever
  `default_texts.json` or the checkpoint changes; `deploy.sh` warns when
  `precache.json` no longer covers the current default texts.
- `deploy.sh` — regenerates assets, **vendors `nla_inference.py` from the repo
  root** (not committed here, to avoid a drifting duplicate), and uploads to
  `syvb/nla-v3-explorer`.
- `embed.html` — fully static, client-side explorer for embedding in posts
  (no Gradio, no GPU, no server). It fetches `precache.json` from the Space
  repo (CORS-enabled), so redeploying the Space refreshes its data with no
  HTML rebuild. Deep links: `#t=<text>&p=<pos>&m=<marginal|cumulative>`;
  theme override: `?theme=dark|light`.
- `build_embed.py` — regenerates `embed.html` from `embed_template.html`.
  Only needed when the template changes; `--inline` bakes the data into a
  self-contained ~0.5MB file (no network needed at view time).

Deploy: `bash deploy.sh` (needs `~/.hf_token`). The `NLACritic` class and the
injection/normalization helpers come from the repo-root `nla_inference.py`.

Changing the example texts: edit `build_assets.py`'s selection (or
`default_texts.json` directly), run `bash precompute_on_vast.sh`, then
`bash deploy.sh`.
