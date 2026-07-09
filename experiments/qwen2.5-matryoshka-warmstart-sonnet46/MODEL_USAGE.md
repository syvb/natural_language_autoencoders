# Working with the trained NLA models — a practical guide

Everything you need to *use* the Qwen2.5-7B L20 NLA AV/AR pair for experiments:
where the checkpoints are, the value-head corruption you **must** know about, and
copy-pasteable recipes for AV inference, AR reconstruction, and activation
extraction. Read the **value-head** section before touching the critic.

## What these models are

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`, `d_model = 3584`, extraction **layer 20**.
- **AV (actor / verbalizer):** activation → text. A full Qwen2.5-7B HF model + an
  `nla_meta.yaml` sidecar. At inference you inject the (normalized) activation at a
  marker token and greedily decode an `<explanation>` (a bulleted list of snippets).
- **AR (critic / reconstructor):** text → activation. The **reward model** for RL:
  reward = −direction-MSE(AR(text), gold activation). It's a *truncated* Qwen2.5
  backbone (first 21 layers, final-norm replaced by Identity) + a **`value_head`**
  `Linear(3584, 3584, bias=False)` saved separately as `value_head.safetensors`.
  Extraction is **suffix-anchored**: read the backbone hidden state at the **last
  token** and apply `value_head`.
- This pair was warm-started (SFT, "matryoshka" from Sonnet-4.6) then RL'd with
  random-length truncation. See `README.md` "RL outcome" for the training story.

**Sidecar is the contract.** Never hardcode token IDs / scales — read them from the
checkpoint's `nla_meta.yaml` and assert against the live tokenizer. Confirmed values
for the RLed AV: `injection_scale = 150.0`, `mse_scale = 59.8665`, marker char `㈎`
(id `149705`, neighbors `29`/`522`), `prompt_templates.{actor,critic}` present.

## Checkpoint inventory (HuggingFace)

| What | Repo / path | Head status |
|---|---|---|
| RLed AV/AR (truncation, **final PoC**) | `syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard` → `kl0.01/iter_0000200/{av,ar}` | AV ✅ · **AR value_head ❌ corrupt** |
| RLed AV/AR (earlier / KL-off ref) | same repo → `kl0.01/iter_0000100/...`, `kl0/iter_{0000100,0000200}/...` | AV ✅ · **AR value_head ❌ corrupt** |
| Warm-start (pre-RL) AV | `syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46` | ✅ |
| Warm-start (pre-RL) AR | `syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46` | ✅ clean value_head |
| Published baseline AV/AR | `kitft/nla-qwen2.5-7b-L20-{av,ar}` | ✅ clean |
| Eval data + gold activations | dataset `syvb/nla-qwen2.5-7b-L20-matryoshka-warmstart-sonnet46` (`av_eval.parquet`, …) | — |

`av/` = full HF model + `nla_meta.yaml`. `ar/` = backbone safetensors +
`value_head.safetensors` + `nla_meta.yaml`. All private; use `~/.hf_token` (user `syvb`).

## ⚠️ The value-head problem (read this before using any RLed critic)

**Every pushed RLed AR `value_head.safetensors` is corrupt** (all four:
`kl0`/`kl0.01` × `iter_100`/`iter_200`). ~12% of weights are destroyed — NaN plus
finite values up to ~3e38 — in **exactly half the output rows (1792/3584)**: the
fingerprint of an FSDP critic-shard that was written as uninitialized memory during
checkpoint save. **Training was unaffected** (the live in-memory head was fine, which
is why the AV trained and `fve_nrm` reached ~0.68); only the *export* is broken, and
the DCP critic checkpoints were deleted, so these exact heads are **unrecoverable**.

A corrupt head makes `value_head(h)` all-NaN → every reconstruction/FVE is NaN. The
AV checkpoints and the AR **backbones** are fine — only the readout is gone.

**Workaround — build a working RLed critic (validated):** pair the RLed AR backbone
with the **clean SFT-warmstart value head**. The head barely changes over 200 RL
steps (median |w| matches; round-trip full-length FVE with this hybrid = 0.705 ≈ the
training `fve_nrm`), and for length/ranking analyses the curve *shape* is set by the
backbone anyway.

```python
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
snapshot_download("syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard",
                  allow_patterns="kl0.01/iter_0000200/ar/*", local_dir="/workspace/dl")
shutil.move("/workspace/dl/kl0.01/iter_0000200/ar", "/workspace/ar_ckpt")
# swap the corrupt head for the clean SFT-warmstart head:
shutil.move("/workspace/ar_ckpt/value_head.safetensors",
            "/workspace/ar_ckpt/value_head.CORRUPT.safetensors")
vh = hf_hub_download("syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46", "value_head.safetensors")
shutil.copy(vh, "/workspace/ar_ckpt/value_head.safetensors")
```
Alternatives needing **no** swap: the warm-start AR or the published `kitft` AR (both
have clean native heads) — but those measure the pre-RL / a different model, not the
truncation-RLed critic. Always sanity-check before trusting a head:
`torch.isfinite(load_file(".../value_head.safetensors")["weight"]).all()`.

**Fixed for future RL runs** (commit `c560ac4`): `nla/train_actor.py:save_model` now
re-gathers the head via `.full_tensor()` and raises on a non-finite head;
`nla/models.py:save_pretrained` + `push_checkpoint_to_hf.sh` add finiteness guards;
tests in `tests/test_value_head_save_guard.py`. So new checkpoints won't have this —
but the *already-pushed* ones above still do.

## Recipe 1 — AV inference (transformers only, no sglang/miles)

Stock `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, ≥24 GB VRAM. Deps:
`transformers==4.57.1 pyyaml safetensors "huggingface_hub>=0.34,<1.0" accelerate`.

```python
import yaml, torch
from transformers import AutoTokenizer, Qwen2ForCausalLM           # NOT AutoModel (see gotchas)
from nla.injection import inject_at_marked_positions
from nla.schema import normalize_activation, extract_explanation

AV = "/workspace/av_ckpt"                                          # NOT /workspace/av (PyAV clash!)
m = yaml.safe_load(open(f"{AV}/nla_meta.yaml")); T = m["tokens"]
inj_id, left, right = T["injection_token_id"], T["injection_left_neighbor_id"], T["injection_right_neighbor_id"]
inj_char, scale = T["injection_char"], m["extraction"]["injection_scale"]
actor_tmpl = m["prompt_templates"]["actor"]                        # ends with <concept>{injection_char}</concept>

tok = AutoTokenizer.from_pretrained(AV)
av = Qwen2ForCausalLM.from_pretrained(AV, dtype=torch.bfloat16, device_map="cuda").eval()
emb = av.get_input_embeddings()
prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content": actor_tmpl.format(injection_char=inj_char)}],
    add_generation_prompt=True, return_tensors="pt").to("cuda")

def verbalize(vec):                                               # vec: raw [3584] activation
    e = emb(prompt_ids)
    v = normalize_activation(torch.tensor(vec, dtype=torch.float32).view(1, -1), scale)
    e2 = inject_at_marked_positions(prompt_ids, e, v, inj_id, left, right)
    with torch.no_grad():
        out = av.generate(inputs_embeds=e2, attention_mask=torch.ones_like(prompt_ids),
                          max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
    return extract_explanation(tok.decode(out[0], skip_special_tokens=True))
```
Batch by repeating `prompt_ids` N× and stacking N vectors (the prompt is identical
across samples). **CJK smoke test:** healthy output is English; whole outputs of
Chinese = injection silently failed. A few stray CJK chars = minor leak.
Canonical script: `fve_truncation_sweep/gen_md_samples.py`.

## Recipe 2 — AR reconstruction (NLACritic)

```python
from nla_inference import NLACritic           # repo-root module
import torch
critic = NLACritic("/workspace/ar_ckpt", device="cuda")   # backbone + value_head (swapped, see above)
vhat = critic.reconstruct("some explanation text")         # -> raw [3584] predicted activation
# direction-only FVE over a set, normalized to mse_scale, raw-mean baseline:
#   1 - mean||p̂/||p̂||·ms - g/||g||·ms||^2 / mean||g/||g||·ms - mean||^2
```
The critic prompt template (`nla_meta.prompt_templates.critic`) is suffix-anchored;
reconstruction reads the **last** token. `reconstruct()` is one forward pass; to batch
many, right-pad and gather the last *real* token (`attention_mask.sum(1)-1`) — see
`fve_truncation_sweep/sweep_fve.py:reconstruct_batch`. Canonical full round-trip:
`fve_truncation_sweep/sweep_fve.py` and `eval_round_trip_fve.py`.

## Recipe 3 — extract activations that match training

Gold activations for `av_eval` are in the dataset repo (`activation_vector` column).
To make **new** activations from your own text, replicate the training convention
exactly (else the AV reads a mismatched vector):

```python
import torch, torch.nn as nn
from transformers import AutoTokenizer, Qwen2ForCausalLM
from nla.arch_adapters import resolve_decoder_layers
LAYER = 20
btok = AutoTokenizer.from_pretrained("/workspace/models/qwen2.5-7b-instruct"); btok.padding_side = "right"
bm = Qwen2ForCausalLM.from_pretrained("/workspace/models/qwen2.5-7b-instruct", dtype=torch.bfloat16, device_map="cuda").eval()
layers = resolve_decoder_layers(bm); bm.model.layers = nn.ModuleList(list(layers[:LAYER + 1]))  # keep 0..20
cap = {}; layers[LAYER].register_forward_hook(lambda m,i,o: cap.__setitem__("h", o[0] if isinstance(o,tuple) else o))
ids = btok(text, add_special_tokens=True)["input_ids"]            # training used raw text + BOS; ≥50 tokens
cap.clear(); bm.model(input_ids=torch.tensor([ids]).cuda(), use_cache=False)
vec = cap["h"][0, -1].float().cpu().numpy()                       # raw, NO normalization (norm="none")
```
Notes: training fed **raw web text** (not chat-templated) — chat-format activations
are **out-of-distribution** for the AV (still partly readable; see `persona_probe/`).
`_MIN_POSITION = 50` (need left-context). Reference: `01_extract_activations.py`.

## Environment + gotchas (these will bite you)

- **Don't name a checkpoint dir `av`.** CWD is on `sys.path`, so `/workspace/av`
  shadows the **PyAV** package torchvision imports → `module 'av' has no attribute
  'logging'`, which masquerades as `Could not find Qwen2ForCausalLM`. Use `av_ckpt`.
- **Use `Qwen2ForCausalLM.from_pretrained` directly**, not `AutoModelForCausalLM` —
  the auto lazy-mapping can swallow the real ImportError of an OOD env.
- **Pin `huggingface_hub<1.0`** (transformers 4.57 breaks on hf_hub 1.x); but the
  `hf buckets` CLI needs ≥1.21 (do bucket ops from the dev box).
- 7B bf16 ≈ 15 GB; AV + (truncated) AR co-fit on a 48 GB card, but loading both at
  once is tight — extract all activations, free the base model, then load the AV/AR.
- See `~/ENV.md` §11 for the full transformers-only inference notes and §8 for the
  miles/SGLang RL stack (only needed to *re-train*, not to use the models).

## Ready-made bundles to copy from
- `fve_truncation_sweep/` — AV+AR round-trip FVE vs truncation length (+ control); has
  `setup_box.sh` (downloads everything incl. the head swap), `sweep_fve.py`, `plot.py`.
- `persona_probe/` — AV as a readout of context personas; activation-extraction +
  injection end-to-end, three extraction distances.
- `gen_md_samples.py` (in fve bundle) — dump N held-out AV generations to markdown.

## Re-training (only if you need to)
RL recipe + the random-length-truncation setup: `run_rl_truncated.sh`,
`setup_rl_box_lmsys.sh`. Two stability fixes are essential and already in-tree: the
**grad-finiteness guard** (`nla/grad_guard.py`, RL critic NaN) and the **value-head
save fix** (above). Smoke-test on critic1 is a FALSE PASS for stability bugs —
reproduce on the sharded critic≥2 layout.
