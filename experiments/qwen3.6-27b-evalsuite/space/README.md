---
title: NLA Qwen3.6-27B — Activation Explorer
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
license: apache-2.0
short_description: Read Qwen3.6-27B's activations in plain English
models:
  - ceselder/qwen3.6-27b-nla-L42
  - Qwen/Qwen3.6-27B
---

# NLA Qwen3.6-27B — read a language model's mind, one token at a time

Interactive demo of a **natural-language autoencoder** (NLA) for **Qwen3.6-27B**
layer-42 activations. Same explorer as the [v3 (Qwen2.5-7B) Space](https://huggingface.co/spaces/syvb/nla-v3-explorer),
retargeted to the released 27B checkpoints:

- **AV (actor)** — Qwen3.6-27B + `av_sft_lora` + `av_rl_lora_step400`, verbalizes
  an injected activation into a salience-ordered list of short descriptions.
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
  linear-attention arch). No torchao; fla not needed for the forward.
- **Actor**: base + `av_sft_lora` + `av_rl_lora_step400`. Both LoRA adapters are
  loaded and kept *active* (numerically the "merge SFT, load RL on top" recipe),
  so a single 27B instance also serves extraction with the adapters disabled.
- **Prompt**: the trained tail — chat template with `enable_thinking=False`
  (pre-closed `<think>\n\n</think>\n\n`) + a `- ` prefill (bypasses the RL
  first-token drift onto `<`). The open-think template default is
  off-distribution (rambles, rarely terminates).
- **Injection**: EasyNLA's karvonen add-norm-matched hook at the layer-1
  residual — `h' = h + ‖h‖·v̂`, direction only (no `injection_scale`), marker
  `㈜` (id 158983) with the left/right neighbor check.
- **Generation**: `do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
  max_new_tokens=256`. No `<explanation>` tags in this run's format — the raw
  newline-separated lines are the output.
- **Reconstruction**: the model's own `rl_critic_step400` via
  `NLACriticModel` + suffix-anchored `critic_predict`, critic prompt
  `Summary of the following text: <text>{expl}</text> <summary>`, both sides
  normalized to `mse_scale = √5120 ≈ 71.55`.

## Hardware

Runs on **ZeroGPU `size="xlarge"`** — a full RTX Pro 6000 Blackwell (96 GB),
free on PRO. Models are loaded in **8-bit** (`bitsandbytes`, near-lossless):
ZeroGPU packs the whole module-level model to a **150 GB-capped ephemeral disk**
at launch, and bf16 (~92 GB pack + the FUSE-cached weights read back) blows past
that; 8-bit halves the pack to ~46 GB. The bf16 weight cache lives in a mounted
**HF Bucket** (`syvb/nla-qwen36-27b-cache`, `HF_HOME=/bucket/hf`), off the
ephemeral limit; the offload stays on `/tmp` (`O_DIRECT`, which FUSE can't do).
`@spaces.GPU(size="xlarge")` is mandatory (the 48 GB `large` default is too
small). The bucket must be mounted read-write at `/bucket` via
`set_space_volumes` — `deploy.sh` does this.

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

No precache: default-text clicks compute live (a runtime LRU still de-dupes
repeat clicks within a session).
