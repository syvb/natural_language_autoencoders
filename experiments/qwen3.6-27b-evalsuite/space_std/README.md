---
title: NLA Qwen3.6-27B — Activation Explorer
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
license: apache-2.0
short_description: Read Qwen3.6-27B activations in English (standard NLA)
models:
  - ceselder/qwen3.6-27b-nla-L42
  - Qwen/Qwen3.6-27B
---

# NLA Qwen3.6-27B — read a language model's mind, one token at a time

Interactive demo of a **natural-language autoencoder** (NLA) for **Qwen3.6-27B**
layer-42 activations — the **standard** NLA (tagged `<explanation>` format; a
[matryoshka variant](https://huggingface.co/spaces/syvb/nla-qwen36-27b-explorer)
front-loads by salience). Same explorer as the
[v3 (Qwen2.5-7B) Space](https://huggingface.co/spaces/syvb/nla-v3-explorer),
retargeted to the released 27B checkpoints:

- **AV (actor)** — Qwen3.6-27B + `av_sft_lora` + `av_rl_lora_step400`, verbalizes
  an injected activation into an `<explanation>` of 2-3 text snippets.
- **AR (critic)** — the co-trained `rl_critic_step400` (43-layer truncated
  Qwen3.6-27B + linear head) reconstructs the activation from text.

Click any token of any text: the Space extracts the layer-42 hidden state at
that position (from the raw base, adapters disabled), verbalizes it, then
reconstructs the vector from cumulative line prefixes and plots **FVE** (fraction
of variance explained) per line — marginal or cumulative.

Checkpoints: [ceselder/qwen3.6-27b-nla-L42](https://huggingface.co/ceselder/qwen3.6-27b-nla-L42).
NLA training code: [EasyNLA](https://github.com/asherps/EasyNLA) (the runtime
subset is vendored under `./nla`).

## Sampling recipe (validated 2026-07-08 — see `experiments/qwen3.6-27b-evalsuite`)

- **Stack**: transformers 5.5.4, peft 0.19.1 (the `qwen3_5` hybrid
  linear-attention arch). No torchao; the ZeroGPU app runs the torch forward
  (fla can't use Triton under ZeroGPU emulation).
- **Actor**: base + `av_sft_lora` + `av_rl_lora_step400` (both adapters active
  ≡ merge SFT, load RL). Extraction disables **both** adapters for the raw-base
  activation; generation runs with them active.
- **Prompt**: the trained tail — chat template with `enable_thinking=False`
  (pre-closed `<think>\n\n</think>\n\n`) + a `<explanation>\n` prefill that opens
  the trained tagged format. The open-think template default is off-distribution
  (rambles, rarely terminates).
- **Injection**: EasyNLA's karvonen add-norm-matched hook at the layer-1
  residual — `h' = h + ‖h‖·v̂`, direction only (no `injection_scale`), marker
  `㈜` (id 158983) with the left/right neighbor check.
- **Generation**: `do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
  max_new_tokens=256`, stopping at `</explanation>` (never trained past the close
  tag). Output = the newline snippets of the `<explanation>` body.
- **Reconstruction**: the model's own `rl_critic_step400` via
  `NLACriticModel` + suffix-anchored `critic_predict`, critic prompt
  `Summary of the following text: <text>{expl}</text> <summary>`, both sides
  normalized to `mse_scale = √5120 ≈ 71.55`.

## Hardware

Runs on **ZeroGPU `size="xlarge"`** — a full RTX Pro 6000 Blackwell (96 GB),
free on PRO — in **bf16**. `@spaces.GPU(size="xlarge")` is mandatory (the 48 GB
`large` default is too small). The tricky part is that ZeroGPU packs the whole
module-level model to a **150 GB-capped ephemeral disk** at launch, and a 92 GB
bf16 model needs three tricks to fit (all in `app.py`):

1. `ZEROGPU_OFFLOAD_DIR=/tmp/...` — the default `/data-nvme` is only 76 GB, and
   the pack uses `O_DIRECT` (can't target a FUSE bucket).
2. `low_cpu_mem_usage=False` (faults weights resident) **and delete the ~92 GB
   download cache before launch**, so the 92 GB pack is the only ephemeral use.
3. `ZEROGPU_MMAP_AUTOPRUNE_PATTERN=<no-match>` — otherwise ZeroGPU's post-pack
   autoprune `lstat()`s the freed blobs and crashes.

Dead ends: **8-bit** (`bitsandbytes`) fits on size but quantizing 27 B at
startup exceeds the 30-min launch timeout; an **HF Bucket** for the cache
doesn't help — Xet-FUSE re-caches read weights onto the ephemeral disk.

## Development

- `app.py` — the Gradio ZeroGPU app (`@spaces.GPU(size="xlarge")`).
- `nla/` — vendored runtime subset of EasyNLA (`config`, `models`, `injection`,
  `utils/*`). Pinned commit in `nla/VENDORED_FROM.txt`.
- `nla_meta.yaml` — the held-out `av_eval` sidecar (marker ids, prompt/critic
  templates, `mse_scale`, `d_model`); asserted against the live tokenizer at
  startup.
- `mu.npy` — the FVE baseline: mean of the held-out `av_eval` activations,
  each normalized to `mse_scale`. Rebuild with `build_mu.py`.
- `default_texts.json` — example texts (shared with the v3 Space).
- `deploy.sh` — uploads the Space (ZeroGPU `zero-a10g`). Set `SPACE_REPO`;
  needs `~/.hf_token` (and a PRO/Team account for ZeroGPU).

Precache: the sample texts (including the Agentic-Misalignment blackmail
honeypot) are precomputed to `precache.json` so their clicks serve instantly;
any other input computes live.
