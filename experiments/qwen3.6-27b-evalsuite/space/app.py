"""NLA Qwen3.6-27B explorer — click a token, read its L42 activation as lines.

Same explorer as the v3 (Qwen2.5-7B) Space, retargeted to the released
Qwen3.6-27B NLAs (ceselder/nla-qwen36-27b-matryoshka). The pipeline and the
sampling recipe are the ones empirically validated in
experiments/qwen3.6-27b-evalsuite (suite_common.py):

Pipeline per click (a dedicated GPU keeps all three roles resident):
  1. EXTRACT   the RAW Qwen3.6-27B base (adapters disabled) forward over the
               text; the layer-42 block output at the clicked token is the
               activation vector v (EasyNLA datagen convention: hidden_states
               [LAYER+1], last token of the causal prefix).
  2. VERBALIZE the AV actor (raw Qwen3.6-27B base + rl_av_lora_iter400, the
               matryoshka RL LoRA; the adapter is disabled for extraction)
               samples a salience-ordered list of feature lines at T=1. The
               activation is injected by EasyNLA's karvonen add-norm-matched
               hook at the layer-1 residual (direction-only; no injection_scale).
               Prompt: trained tail (pre-closed <think></think>), no prefill;
               decoding stops after N_LINES lines (the policy rarely EOSes).
  3. RECONSTRUCT the co-trained critic (rl_critic_step400, NLACriticModel) reads
               cumulative line prefixes (1..k) and predicts v̂_k; FVE_k =
               1 − ||n(v̂_k)−n(v)||² / ||n(v)−μ||², μ = mean of normalized
               held-out activations (mu.npy), both sides scaled to mse_scale.

Everything model-specific (marker token ids, prompt/critic templates, d_model,
mse_scale) is read from the shipped nla_meta.yaml sidecar via EasyNLA's
load_nla_config, asserted against the live tokenizer at startup — nothing
hardcoded.

Runs on ZeroGPU size="xlarge" (a full RTX Pro 6000 Blackwell, 96GB) in bf16.
ZeroGPU packs the whole module-level model to a 150GB-capped ephemeral disk at
launch; a 92GB bf16 model fits only with three tricks (see the top env block +
_free_cache_and_report): pack the offload on /tmp (default /data-nvme is 76GB),
load low_cpu_mem_usage=False + delete the ~92GB download cache before launch so
the pack is the only ephemeral use, and disable ZeroGPU's post-pack autoprune
(it would lstat the freed blobs and crash). Models are placed on the emulated
GPU at import and materialize in the GPU fork; the whole per-click pipeline runs
inside one @spaces.GPU call. (8-bit fit the size but quantizing 27B at startup
blew the 30-min launch timeout; a mounted HF Bucket didn't help — Xet-FUSE
re-caches read weights back onto the ephemeral disk.)
"""

import os

# Storage layout for a 92GB bf16 model on ZeroGPU (evicts past a 150GB
# ephemeral-storage limit). At demo.launch() ZeroGPU packs the whole module-
# level model to ZEROGPU_OFFLOAD_DIR (O_DIRECT, must be a real block fs — not a
# FUSE bucket). The platform default (/data-nvme, 76GB) is too small, so pack on
# the big ephemeral fs (/tmp). To keep ephemeral under 150GB we (a) load weights
# into RAM (low_cpu_mem_usage=False) so the pack reads RAM, not an mmap'd cache,
# and (b) delete the ~92GB download cache before launch — leaving the 92GB pack
# as the only ephemeral consumer. (A bucket doesn't help here: Xet-FUSE re-caches
# read weights onto the ephemeral disk, and O_DIRECT can't target FUSE.) Set
# before importing spaces, which reads this env at import.
os.environ["ZEROGPU_OFFLOAD_DIR"] = "/tmp/zerogpu-tensors"
# Disable ZeroGPU's own post-pack autoprune: it lstat()s + unlinks the mmap'd
# cache blobs AFTER packing (unmap_capture filters by this glob), but we free
# them ourselves BEFORE the pack (to keep ephemeral < 150GB), so its lstat hits
# a deleted path and crashes. A glob that matches no real blob path makes
# unmap_capture capture nothing → no lstat, no crash. The pack still reads the
# already-resident (low_cpu_mem_usage=False) tensor pages fine.
os.environ["ZEROGPU_MMAP_AUTOPRUNE_PATTERN"] = "__zerogpu_autoprune_disabled__"
os.makedirs(os.environ["ZEROGPU_OFFLOAD_DIR"], exist_ok=True)

import spaces  # must be imported before any CUDA touch

import html as html_lib
import json
import re
import time
from collections import OrderedDict
from threading import Lock, Thread

import gradio as gr
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from nla.config import load_nla_config
from nla.models import NLACriticModel
from nla.utils import build_prompt_text, critic_predict, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

# ── checkpoints (matryoshka NLA — the ordering-trained model that front-loads
#    by salience, which is what the FVE-per-line viz shows) ─────────────────────
MODEL_REPO = "ceselder/nla-qwen36-27b-matryoshka"
BASE_ID = "Qwen/Qwen3.6-27B"          # text-only weights load via AutoModelForCausalLM
RL_SUBDIR = "rl_av_lora_iter400"      # single RL LoRA on the RAW text base (no SFT merge)
TOK_SUBDIR = "warmstart_av_lora"      # carries the tokenizer + the bullet-format sidecar
CRITIC_SUBDIR = "rl_critic_step400"   # co-trained reconstructor

N_LINES = 10          # cap the explanation lines the viz plots
MAX_TEXT_TOKENS = 5120  # bounds the extraction forward; fits the ~4.7k-token with-response honeypot
MAX_NEW = 256         # the training cap — the matryoshka RL policy rarely EOSes
GOOD_MIN_POS = 10     # very early positions have little left-context
LAYER = 42            # extraction layer (block output = hidden_states[LAYER+1])

# Trained generation tail: the chat template opens `<think>\n`; the 27B NLAs
# were trained with enable_thinking=False (pre-closed think pair). The template
# default (open think) is off-distribution — outputs ramble and rarely
# terminate. The matryoshka RL model natively opens straight into a salience-
# ordered list of feature lines (no <explanation> tags, no prefill — a prefill
# hurts it), and almost never emits EOS, so decoding is bounded by a
# stop-after-N-lines criterion + the max_new cap.
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
PREFILL = ""

HERE = os.path.dirname(os.path.abspath(__file__))
CJK_RE = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")

# ── sidecar config + tokenizer ───────────────────────────────────────────────
root = snapshot_download(MODEL_REPO)
RL_DIR = f"{root}/{RL_SUBDIR}"
TOK_DIR = f"{root}/{TOK_SUBDIR}"
CRITIC_DIR = f"{root}/{CRITIC_SUBDIR}"

tok = AutoTokenizer.from_pretrained(TOK_DIR)
# load_nla_config reads ./nla_meta.yaml (the matryoshka warmstart_av_lora
# bullet-format sidecar shipped alongside this app) and asserts marker id +
# neighbors against `tok`.
cfg = load_nla_config(HERE, tok)
INJ_CHAR = cfg.injection_char
MSE_SCALE = float(cfg.mse_scale)
CRITIC_TPL = cfg.critic_prompt_template
assert CRITIC_TPL and "{explanation}" in CRITIC_TPL, "sidecar missing critic template"

MU = torch.tensor(np.load(os.path.join(HERE, "mu.npy")), dtype=torch.float32)
DEFAULT_TEXTS = json.load(open(os.path.join(HERE, "default_texts.json")))
# Friendly names for the sample-text picker (same order as default_texts.json:
# 4 plain passages + the rendered blackmail chat scenario). Regenerated set →
# update these too. The last is a full Qwen chat transcript, not a short passage.
EXAMPLE_LABELS = [
    "Miss America pageant news",
    "iOS simulator error",
    "Ocean bacteria signalling",
    "DEA / CBD rescheduling",
    "⚠️ Blackmail honeypot + model's reply",
][: len(DEFAULT_TEXTS)]

# ── fixed AV prompt (the user text is never shown to the AV — only the vector)─
_content = cfg.actor_prompt_template.format(injection_char=INJ_CHAR)
_ptxt = build_prompt_text([{"role": "user", "content": _content}], INJ_CHAR, tok)
assert _ptxt.endswith(THINK_OPEN), repr(_ptxt[-30:])
_ptxt = _ptxt[: -len(THINK_OPEN)] + PRECLOSED + PREFILL
PROMPT_IDS = tok.encode(_ptxt, add_special_tokens=False)
PROMPT_LEN = len(PROMPT_IDS)
print(f"[prompt] len={PROMPT_LEN} tail={_ptxt[-40:]!r}", flush=True)

# ── models (bf16, loaded into RAM so ZeroGPU packs from RAM not an mmap) ───────
# low_cpu_mem_usage=False forces a real RAM copy (not an mmap of the cache
# files), which lets us delete the cache before ZeroGPU's launch-time pack — so
# the ~92GB O_DIRECT offload is the only thing on the 150GB-capped ephemeral fs.
# (bnb 8-bit fit the size but quantizing 27B at ZeroGPU startup blew the 30-min
# launch timeout; bf16 loads fast.)
#
# ONE base instance serves both roles: the matryoshka RL LoRA sits directly on
# the RAW text base (no SFT merge), so generation runs with the adapter active
# and extraction disables it (disable_adapter ⇒ raw base, the datagen convention).
print("[load] base Qwen3.6-27B (bf16, RAM)…", flush=True)
_base = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    low_cpu_mem_usage=False)
# torch_device="cpu": the adapter safetensors load would otherwise target a real
# CUDA device (peft's infer_device sees the emulated is_available()=True), which
# fails at module level. Load to CPU, then actor.to("cuda") moves the stack onto
# the emulated GPU (which the emulation supports for bf16).
actor = PeftModel.from_pretrained(_base, RL_DIR, torch_device="cpu")
actor.to("cuda").eval()

vref = [None]  # karvonen hook reads vref[0]; None ⇒ no-op (extraction/decode)
register_karvonen_hook(actor, vref, cfg.injection_token_id,
                       cfg.injection_left_neighbor_id,
                       cfg.injection_right_neighbor_id, layer_idx=1)
BASE_LAYERS = resolve_decoder_layers(actor.get_base_model())

# ── L42 activation patch (the intervention feature) ──────────────────────────
# patch_ref[0] is None (no-op) or (position, fp32 vector at raw-activation
# scale): during the prefill chunk that computes `position`, the L42 block
# output there is replaced by the vector — so everything downstream (layers
# 43.., the KV every later token attends to, and the very next sampled token)
# sees the patched activation. Decode steps have q_len=1 ≠ position+1 and are
# untouched. Registered once; live only inside _continue's try/finally.
patch_ref: list = [None]


def _l42_patch(module, inputs, output):
    pr = patch_ref[0]
    if pr is None:
        return
    pos, vec = pr
    h = output[0] if isinstance(output, tuple) else output
    if h.shape[1] == pos + 1:  # exactly the prefill chunk ending at the target
        h[:, pos, :] = vec.to(h.dtype)
    return output


BASE_LAYERS[LAYER].register_forward_hook(_l42_patch)

print("[load] critic rl_critic_step400 (NLACriticModel, bf16, RAM)…", flush=True)
critic = NLACriticModel.from_pretrained(
    CRITIC_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    low_cpu_mem_usage=False)
critic.to("cuda").eval()
DEV = "cuda"
print(f"[ready] d_model={cfg.d_model} mse_scale={MSE_SCALE:.2f} "
      f"marker={INJ_CHAR!r}(id={cfg.injection_token_id}) layer={LAYER}", flush=True)


def _free_cache_and_report() -> None:
    """Delete the bf16 download blobs (now copied into RAM) and log RAM + disk.

    Safe here because low_cpu_mem_usage=False made the weights real RAM tensors,
    not mmap views of these files — so ZeroGPU's launch-time pack reads RAM, and
    its autoprune (which lstat()s mmap'd source files) finds none to trip on.
    Deleting frees the ~92GB cache so the pack's ~92GB O_DIRECT offload is the
    only thing on the 150GB-capped ephemeral fs."""
    import glob
    import shutil
    from huggingface_hub.constants import HF_HUB_CACHE
    os.makedirs(os.environ["ZEROGPU_OFFLOAD_DIR"], exist_ok=True)
    freed = 0
    for blob in glob.glob(os.path.join(HF_HUB_CACHE, "models--*", "blobs", "*")):
        try:
            freed += os.path.getsize(blob)
            os.remove(blob)
        except OSError:
            pass
    try:
        mem = {k: int(v.split()[0]) for k, v in
               (ln.split(":", 1) for ln in open("/proc/meminfo"))}
        print(f"[ram] MemTotal={mem['MemTotal']/2**20:.0f}G "
              f"MemAvailable={mem['MemAvailable']/2**20:.0f}G", flush=True)
    except OSError:
        pass
    for p in [os.environ["ZEROGPU_OFFLOAD_DIR"], HF_HUB_CACHE, "/tmp"]:
        try:
            t, _, f = shutil.disk_usage(p)
            print(f"[disk] {p}: total={t/1e9:.0f}G free={f/1e9:.0f}G", flush=True)
        except OSError:
            pass
    print(f"[disk] freed {freed/1e9:.1f}GB of bf16 cache blobs before pack",
          flush=True)


_free_cache_and_report()

# Steering is v3-only (needs L42 27B trait directions we don't ship) — the
# accordion stays hidden and every steer key resolves to a plain analysis.
STEER_DIRS: dict[str, torch.Tensor] = {}


def _normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12) * scale


def _lines(text: str, streaming: bool) -> list[str]:
    """The model's salience-ordered feature lines: split on newlines, whitespace-
    strip, drop empties. Fed to BOTH the critic (its co-trained input) and the
    viz — the matryoshka emits plain lines (no bullet markers), so nothing is
    stripped, matching suite_fve. While streaming, drop the trailing not-yet-
    newline-terminated line so only complete lines are counted/shown."""
    if streaming:
        parts = text.split("\n")[:-1]
    else:
        parts = re.split(r"\n+", text)
    return [ln.strip() for ln in parts if ln.strip()]


class _StopAfterLines(StoppingCriteria):
    """End decoding once N_LINES complete (newline-terminated) feature lines have
    been generated — the matryoshka policy rarely emits EOS, so this bounds it.
    input_ids is [prompt + generated]; count only the generated tail."""

    def __init__(self, prompt_len: int):
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        gen = tok.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        return len(_lines(gen, streaming=True)) >= N_LINES


class _StopForward(Exception):
    """Raised from the layer-42 hook to abort the forward once we have the
    activation — skips every block after L42 (extraction needs nothing more)."""


@torch.inference_mode()
def _extract(prefix_ids: list[int]) -> np.ndarray:
    """Layer-42 block output at the last token of the causal prefix, from the
    RAW base (adapters disabled ⇒ karvonen hook also no-ops via vref=None)."""
    grabbed = {}

    def grab(module, inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
        raise _StopForward  # don't run blocks L43..N — we already have L42

    h = BASE_LAYERS[LAYER].register_forward_hook(grab)
    try:
        ids = torch.tensor([prefix_ids], device=DEV)
        with actor.disable_adapter():
            try:
                actor(input_ids=ids, use_cache=False)
            except _StopForward:
                pass
        return grabbed["h"][0, -1].float().cpu().numpy()
    finally:
        h.remove()


@torch.inference_mode()
def _reconstruct_batch(texts: list[str]) -> torch.Tensor:
    """Critic reconstruction of each explanation text → [N, d_model] (fp32 cpu).
    Right-padded, suffix-anchored — mirrors suite_fve.reconstruct."""
    idlists = [tok.encode(CRITIC_TPL.format(explanation=e),
                          add_special_tokens=False)[:1024] for e in texts]
    m = max(len(x) for x in idlists)
    pad = tok.eos_token_id
    bx = torch.full((len(idlists), m), pad, dtype=torch.long, device=DEV)
    attn = torch.zeros((len(idlists), m), dtype=torch.long, device=DEV)
    for r, q in enumerate(idlists):
        bx[r, : len(q)] = torch.tensor(q, dtype=torch.long)
        attn[r, : len(q)] = 1
    return critic_predict(critic, bx, attn, MSE_SCALE).float().cpu()


# duration=180: extraction (one 27B forward over ≤1024 tokens) + a 256-token
# T=1 generation + the critic's per-prefix reconstruction. size="xlarge" is
# REQUIRED — the ~93GB actor+critic exceed the 48GB "large" default slice.
@spaces.GPU(size="xlarge", duration=180)
def gpu_analyze(token_ids: list[int], idx: int, steer: str | None = None,
                strength: float = 0.0):
    """Extract activation at token idx, verbalize (streamed), score prefixes.

    Generator: yields {"lines": [...], "fve": None} as each explanation line
    finishes decoding, then the final dict with fve/cos. `steer`/`strength` are
    accepted for signature-compatibility with the v3 app but unused here (no
    27B trait directions are shipped)."""
    _t = time.time()
    v = _extract(token_ids[: idx + 1])            # [d] raw activation
    _t_ext = time.time() - _t
    vref[0] = torch.tensor(v[None], dtype=torch.float32, device=DEV)
    pt = torch.tensor([PROMPT_IDS], device=DEV)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen = Thread(target=actor.generate, kwargs=dict(   # generate() is no_grad
        input_ids=pt, attention_mask=torch.ones_like(pt),
        # temperature-1 sampling (never greedy); top_p/top_k explicit so a
        # generation_config.json can never silently reshape the sampling.
        max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0,
        top_k=0, pad_token_id=tok.eos_token_id,
        # stop once N_LINES feature lines exist (the matryoshka rarely EOSes) —
        # bounds latency instead of always decoding the full max_new budget.
        stopping_criteria=StoppingCriteriaList([_StopAfterLines(PROMPT_LEN)]),
        streamer=streamer,
    ))
    _t = time.time()
    try:
        gen.start()
        text, shown = PREFILL, 0   # PREFILL='' ⇒ streamed text is generation-only
        for piece in streamer:
            text += piece
            done = _lines(text, streaming=True)
            if len(done) > shown:
                shown = len(done)
                yield {"lines": done[:N_LINES], "fve": None}
        gen.join()
    finally:
        vref[0] = None
    _t_gen = time.time() - _t

    lines = _lines(text, streaming=False)[:N_LINES]
    if not lines:
        yield {"lines": [], "fve": [], "cos": []}
        return

    _t = time.time()
    gold_n = _normalize(torch.tensor(v, dtype=torch.float32), MSE_SCALE)
    denom = ((gold_n - MU) ** 2).mean().item()
    preds = _reconstruct_batch(["\n".join(lines[:k]) for k in range(1, len(lines) + 1)])
    fve, cos = [], []
    for pred in preds:
        pred_n = _normalize(pred, MSE_SCALE)
        fve.append(1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom)
        cos.append(float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm())))
    print(f"[timing] extract={_t_ext:.1f}s gen={_t_gen:.1f}s "
          f"recon={time.time() - _t:.1f}s lines={len(lines)}", flush=True)
    yield {"lines": lines, "fve": fve, "cos": cos}


@torch.inference_mode()
def _continue(prefix_ids: list[int], n_new: int, seed: int,
              patch_vec: torch.Tensor | None = None) -> str:
    """Continue the ORIGINAL text after the clicked token with the raw base
    (adapters disabled), optionally rewriting the L42 block output at the
    clicked position (the last prefix token) with `patch_vec` (fp32, raw-
    activation scale). Seed-matched T=1 sampling: identical activations ⇒
    identical text, so divergence between two conditions is caused by the
    patch, not by sampling noise."""
    ids = torch.tensor([prefix_ids], device=DEV)
    patch_ref[0] = (None if patch_vec is None
                    else (len(prefix_ids) - 1, patch_vec.to(DEV)))
    torch.manual_seed(int(seed))
    try:
        with actor.disable_adapter():
            out = actor.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=int(n_new), do_sample=True, temperature=1.0,
                top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    finally:
        patch_ref[0] = None
    return tok.decode(out[0, len(prefix_ids):], skip_special_tokens=True)


# duration=240: one extraction + one critic batch + three ≤160-token 27B
# continuations (seed-matched conditions run sequentially — batching them
# would interleave the RNG stream and break "same logits ⇒ same text").
@spaces.GPU(size="xlarge", duration=240)
def gpu_intervene(token_ids: list[int], idx: int, orig_lines: list[str],
                  edit_lines: list[str], n_new: int, seed: int):
    """Generator: reconstruction stats first, then the three continuations as
    each finishes — no-patch (true activation), v̂(original lines) (control:
    reconstruction error alone), v̂(edited lines) (treatment)."""
    prefix = token_ids[: idx + 1]
    v = _extract(prefix)
    vt = torch.tensor(v, dtype=torch.float32)
    gold_n = _normalize(vt, MSE_SCALE)
    denom = ((gold_n - MU) ** 2).mean().item()
    preds = _reconstruct_batch(["\n".join(orig_lines), "\n".join(edit_lines)])
    stats, vecs = [], []
    for pred in preds:
        pred_n = _normalize(pred, MSE_SCALE)
        stats.append({
            "fve": 1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom,
            "cos": float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm()))})
        # patch at the RAW activation's magnitude — direction from the critic
        vecs.append(pred / pred.norm().clamp_min(1e-12) * vt.norm())
    shift = float(vecs[0] @ vecs[1] / (vecs[0].norm() * vecs[1].norm()))
    out = {"stats": stats, "shift": shift, "raw": None, "orig": None, "edit": None}
    yield dict(out)
    out["raw"] = _continue(prefix, n_new, seed, None)
    yield dict(out)
    out["orig"] = _continue(prefix, n_new, seed, vecs[0])
    yield dict(out)
    out["edit"] = _continue(prefix, n_new, seed, vecs[1])
    yield dict(out)


# ── viz + ui (pure CPU; palette per the validated reference set) ─────────────
CSS = """
/* layout — overflow:visible: gradio's default overflow:hidden breaks position:sticky */
.gradio-container{max-width:1360px !important; margin:0 auto !important;
  overflow:visible !important;}
#nla-header h1{font-size:23px; margin-bottom:0;}
#nla-header p{margin-top:6px;}
#nla-click-idx{display:none !important;}
#nla-steer-r{display:none !important;}
.nla-side{position:sticky !important; top:14px; align-self:flex-start !important;}

/* design tokens (light) + dark overrides */
.gradio-container{
  --nla-surface:#fcfcfb; --nla-ink:#0b0b0b; --nla-ink2:#52514e; --nla-muted:#898781;
  --nla-grid:#e1e0d9; --nla-axis:#c3c2b7; --nla-pos:#2a78d6; --nla-neg:#e34948;
  --nla-warn:#c98500; --nla-wash:rgba(11,11,11,.045); --nla-ring:rgba(11,11,11,.10);
  --nla-hover:rgba(42,120,214,.16); --nla-sel:#2a78d6;
}
.dark .gradio-container, .gradio-container.dark{
  --nla-surface:#1a1a19; --nla-ink:#ffffff; --nla-ink2:#c3c2b7; --nla-muted:#898781;
  --nla-grid:#2c2c2a; --nla-axis:#383835; --nla-pos:#3987e5; --nla-neg:#e66767;
  --nla-warn:#c98500; --nla-wash:rgba(255,255,255,.055); --nla-ring:rgba(255,255,255,.10);
  --nla-hover:rgba(57,135,229,.30); --nla-sel:#3987e5;
}

/* token panel — long texts scroll inside the panel, never the page */
.tokpanel{background:var(--nla-surface); border:1px solid var(--nla-ring);
  border-radius:12px; overflow:hidden;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}
.tokhead{display:flex; justify-content:space-between; gap:12px; padding:8px 14px;
  font-size:10.5px; letter-spacing:.05em; text-transform:uppercase;
  color:var(--nla-muted); border-bottom:1px solid var(--nla-grid);}
.tokhead .trunc{color:var(--nla-warn); text-transform:none; letter-spacing:0;}
.tokscroll{padding:12px 14px 16px; max-height:56vh; overflow-y:auto;
  scrollbar-width:thin; white-space:pre-wrap; overflow-wrap:anywhere;
  font-size:14px; line-height:2.0; color:var(--nla-ink);}
.nla-tok{cursor:pointer; border-radius:4px; padding:2.5px 0;}
.nla-tok:nth-child(2n){background:var(--nla-wash);}
.nla-tok:hover, .nla-tok:focus-visible{background:var(--nla-hover); outline:none;}
.nla-tok.sel{background:var(--nla-sel); color:#fff;}
.nla-tok .nl{color:var(--nla-muted); font-size:10px;}
.nla-tok.sel .nl{color:rgba(255,255,255,.75);}
body.nla-heat-p .nla-tok{background-color:rgba(224,58,58, calc(var(--pa,0)*0.85));}
body.nla-heat-f .nla-tok{background-color:rgba(224,58,58, calc(var(--paf,0)*0.85));}
body.nla-heat-fn .nla-tok{background-color:rgba(224,58,58, calc(var(--pafn,0)*0.85));}
body[class*="nla-heat"] .nla-tok:hover, body[class*="nla-heat"] .nla-tok:focus-visible{background:var(--nla-hover);}
body[class*="nla-heat"] .nla-tok.sel{background:var(--nla-sel);}
.eabadge{display:inline-block; min-width:2.3em; text-align:center; font-size:9px; font-weight:700;
  padding:0 3px; margin-right:5px; border-radius:3px; color:#fff; vertical-align:middle;
  background:rgba(198,52,52, calc(var(--pea,0)*0.65 + 0.35));}

/* results card */
.nlaviz{background:var(--nla-surface); border:1px solid var(--nla-ring);
  border-radius:12px; padding:16px 18px; color:var(--nla-ink);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;}
.nlaviz .title{font-size:13px; font-weight:600; margin-bottom:2px;}
.nlaviz .sub{font-size:11.5px; color:var(--nla-ink2); margin-bottom:12px;}
.nlaviz .chips{display:flex; gap:22px; margin-bottom:14px; flex-wrap:wrap;}
.nlaviz .chip .v{font-size:20px; font-weight:650;}
.nlaviz .chip .l{font-size:10.5px; color:var(--nla-muted); text-transform:uppercase;
  letter-spacing:.04em; margin-top:1px;}
.nlaviz .row{display:grid; grid-template-columns:16px minmax(0,1fr) 96px 48px;
  gap:10px; align-items:center; padding:5px 6px; border-radius:6px;}
.nlaviz .row:hover{background:var(--nla-wash);}
.nlaviz .idx{font-size:11px; color:var(--nla-muted); text-align:right;
  font-variant-numeric:tabular-nums;}
.nlaviz .line{font-size:12.5px; line-height:1.45; color:var(--nla-ink);
  display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;}
.nlaviz .track{position:relative; height:14px;}
.nlaviz .zero{position:absolute; top:-3px; bottom:-3px; width:1px; background:var(--nla-axis);}
.nlaviz .bar{position:absolute; top:2px; height:10px;}
.nlaviz .bar.pos{background:var(--nla-pos); border-radius:0 4px 4px 0;}
.nlaviz .bar.neg{background:var(--nla-neg); border-radius:4px 0 0 4px;}
.nlaviz .pend{position:absolute; top:2px; height:10px; width:100%; border-radius:4px;
  background:var(--nla-wash); animation:nlapulse 1.2s ease-in-out infinite;}
@keyframes nlapulse{50%{opacity:.3;}}
.nlaviz .val{font-size:11.5px; color:var(--nla-ink2); text-align:right;
  font-variant-numeric:tabular-nums;}
.nlaviz .axisrow{display:grid; grid-template-columns:16px minmax(0,1fr) 96px 48px;
  gap:10px; padding:2px 6px 0;}
.nlaviz .axislab{position:relative; height:14px; font-size:10px; color:var(--nla-muted);
  font-variant-numeric:tabular-nums;}
.nlaviz .axislab span{position:absolute; transform:translateX(-50%);}
.nlaviz .note{font-size:11px; color:var(--nla-ink2); margin-top:10px;}
.nlaviz .warnic{color:var(--nla-warn);}
.nlaviz .empty{padding:36px 12px; text-align:center; color:var(--nla-muted); font-size:13px;}

/* intervention: three side-by-side continuations */
.contgrid{display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:12px;}
@media (max-width:900px){.contgrid{grid-template-columns:1fr;}}
.contbox{border:1px solid var(--nla-grid); border-radius:8px; overflow:hidden;
  display:flex; flex-direction:column;}
.contbox .chd{padding:6px 10px; font-size:10.5px; letter-spacing:.05em;
  text-transform:uppercase; color:var(--nla-muted); border-bottom:1px solid var(--nla-grid);
  display:flex; justify-content:space-between; gap:8px;}
.contbox .chd .tag{color:var(--nla-ink2); text-transform:none; letter-spacing:0;}
.contbox.treat .chd{color:var(--nla-neg);}
.contbox .ctx{padding:10px 12px; font-size:12.5px; line-height:1.55;
  white-space:pre-wrap; overflow-wrap:anywhere; color:var(--nla-ink);
  max-height:34vh; overflow-y:auto; scrollbar-width:thin;}
.contbox .ctx .pendc{color:var(--nla-muted); animation:nlapulse 1.2s ease-in-out infinite;}
mark.div{background:rgba(224,58,58,.22); color:inherit; border-radius:2px; padding:0;}
"""

# Clicks on the custom token spans are routed to the backend through a hidden
# textbox (#nla-click-idx): set its value to the token index, dispatch `input`.
CLICK_JS = """
() => {
  const send = (t) => {
    document.querySelectorAll('.nla-tok.sel').forEach((x) => x.classList.remove('sel'));
    t.classList.add('sel');
    const box = document.querySelector('#nla-click-idx textarea, #nla-click-idx input');
    if (!box) return;
    box.value = t.dataset.i;
    box.dispatchEvent(new Event('input', { bubbles: true }));
  };
  document.addEventListener('click', (e) => {
    const t = e.target.closest('.nla-tok');
    if (t) send(t);
  });
  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const t = e.target.closest && e.target.closest('.nla-tok');
    if (t) { e.preventDefault(); send(t); }
  });
}
"""


def _card(msg: str) -> str:
    return f'<div class="nlaviz"><div class="empty">{msg}</div></div>'


EMPTY_CARD = _card("👈 Click any token to read its activation.")


def render_tokens(pieces: list[str], n_total: int, paware=None) -> str:
    # paware[i] is None or a (p, p_fvew, p_fvewnorm) triple → three CSS vars, one
    # per heatmap mode; the body class picks which var tints the token.
    spans = []
    for i, p in enumerate(pieces):
        body = html_lib.escape(p).replace("\n", '<span class="nl">⏎</span><br>')
        tr = paware[i] if (paware and i < len(paware) and paware[i] is not None) else None
        if tr is not None:
            pa, paf, pafn = (v if v is not None else 0.0 for v in tr)
            style = f' style="--pa:{pa:.3f};--paf:{paf:.3f};--pafn:{pafn:.3f}"'
            ttl = f"#{i} · P(aware)={pa:.2f} · ×ΔFVE={paf:.2f} · ×ΔFVE/rank={pafn:.2f}"
        else:
            style, ttl = "", f"#{i}"
        spans.append(f'<span class="nla-tok" data-i="{i}" title="{ttl}"{style} '
                     f'role="button" tabindex="0">{body}</span>')
    trunc = (f' <span class="trunc">✂ truncated to the first {len(pieces)}</span>'
             if n_total > len(pieces) else "")
    head = (f'<div class="tokhead"><span>{len(pieces)} tokens{trunc}</span>'
            f'<span>click a token to analyze it</span></div>')
    return f'<div class="tokpanel">{head}<div class="tokscroll">{"".join(spans)}</div></div>'


def render_pending(state: dict, mode: str) -> str:
    """Streaming view: lines appear as they decode; bars are shimmer stubs."""
    title = {"marginal": "Additional FVE per explanation line (ΔFVE)",
             "ablation": "FVE lost when each line is deleted (leave-one-out)"}.get(
        mode, "Cumulative round-trip FVE by explanation-line prefix")
    tok_piece = html_lib.escape(state.get("token", ""))
    chips = (
        f'<div class="chips">'
        f'<div class="chip"><div class="v">…</div><div class="l">FVE</div></div>'
        f'<div class="chip"><div class="v">…</div><div class="l">cosine</div></div>'
        f'<div class="chip"><div class="v">{tok_piece or "—"}</div><div class="l">token @ {state.get("pos", "?")}</div></div>'
        f'</div>'
    )
    rows = []
    for i, ln in enumerate(state["lines"]):
        rows.append(
            f'<div class="row"><div class="idx">{i + 1}</div>'
            f'<div class="line">{html_lib.escape(ln)}</div>'
            f'<div class="track"><div class="pend"></div></div>'
            f'<div class="val">·</div></div>'
        )
    return (f'<div class="nlaviz"><div class="title">{title}</div>'
            f'<div class="sub">verbalizing the activation…</div>{chips}{"".join(rows)}'
            f'<div class="note">reconstruction scores arrive when all lines are decoded.</div></div>')


def render_viz(state: dict | None, mode: str) -> str:
    if not state:
        return EMPTY_CARD
    if state.get("fve") is None:
        return render_pending(state, mode)
    lines, fve, cos = state["lines"], state["fve"], state["cos"]
    marginal = [fve[0]] + [fve[k] - fve[k - 1] for k in range(1, len(fve))]
    loo = state.get("loo") if mode == "ablation" else None
    loo_missing = mode == "ablation" and not loo
    if loo and len(loo["loo"]) == len(lines):
        vals = [loo["full"] - x for x in loo["loo"]]  # necessity of each line
        title = "FVE lost when each line is deleted (leave-one-out)"
        sub = ("all OTHER lines kept — necessity, not salience order; a small bar with a "
               "large ΔFVE means the line is redundant with later ones")
    else:
        vals = marginal if mode == "marginal" else fve
        title = ("Additional FVE per explanation line (ΔFVE)" if mode == "marginal"
                 else "Cumulative round-trip FVE by explanation-line prefix")
        sub = ("how much reconstruction each successive line adds" if mode == "marginal"
               else "reconstruction quality from lines 1..k only")
        loo = None

    lo, hi = min(0.0, min(vals)), max(0.05, max(vals))
    span = hi - lo
    zero_pct = (0.0 - lo) / span * 100

    ea = state.get("paware_lines")  # per-line P(eval-aware), if precached
    rows = []
    for i, (ln, x) in enumerate(zip(lines, vals)):
        w = abs(x) / span * 100
        left = zero_pct if x >= 0 else zero_pct - w
        cls = "pos" if x >= 0 else "neg"
        pv = ea[i] if (ea and i < len(ea) and ea[i] is not None) else None
        badge = (f'<span class="eabadge" style="--pea:{pv:.3f}" '
                 f'title="P(eval-aware)={pv:.2f}">{pv:.2f}</span>' if pv is not None else "")
        tip = (f"line {i + 1} — ΔFVE {marginal[i]:+.3f}, cumulative {fve[i]:.3f}, "
               f"cos {cos[i]:.3f}"
               + (f", delete→lose {vals[i]:+.3f}, alone {loo['solo'][i]:+.3f}" if loo else "")
               + (f", P(eval-aware) {pv:.2f}" if pv is not None else ""))
        rows.append(
            f'<div class="row" title="{html_lib.escape(tip)}">'
            f'<div class="idx">{i + 1}</div>'
            f'<div class="line" title="{html_lib.escape(ln)}">{badge}{html_lib.escape(ln)}</div>'
            f'<div class="track"><div class="zero" style="left:{zero_pct:.2f}%"></div>'
            f'<div class="bar {cls}" style="left:{left:.2f}%;width:{max(w, 0.4):.2f}%"></div></div>'
            f'<div class="val">{x:+.3f}</div></div>'
        )
    axis_spans = [f'<span style="left:{zero_pct:.2f}%">0</span>',
                  f'<span style="left:100%">{hi:.2f}</span>']
    if zero_pct >= 10:  # negative bars present and the lo label won't collide with "0"
        axis_spans.insert(0, f'<span style="left:0%">{lo:.2f}</span>')
    axis = (f'<div class="axisrow"><div></div><div></div>'
            f'<div class="axislab">{"".join(axis_spans)}</div><div></div></div>')

    tok_piece = html_lib.escape(state.get("token", ""))
    chips = (
        f'<div class="chips">'
        f'<div class="chip"><div class="v">{fve[-1]:.3f}</div><div class="l">FVE · all {len(lines)} lines</div></div>'
        f'<div class="chip"><div class="v">{cos[-1]:.3f}</div><div class="l">cosine</div></div>'
        f'<div class="chip"><div class="v">{tok_piece or "—"}</div><div class="l">token @ {state.get("pos", "?")}</div></div>'
        f'</div>'
    )
    notes = []
    if loo_missing:
        notes.append('<span class="warnic">⚠</span> ablation scores are precomputed for the '
                     'sample texts only — showing marginal ΔFVE instead.')
    if state.get("pos", GOOD_MIN_POS) < GOOD_MIN_POS:
        notes.append('<span class="warnic">⚠</span> very early position — little left-context; '
                     'explanations may be generic.')
    if any(CJK_RE.search(ln) for ln in lines):
        notes.append('<span class="warnic">⚠</span> stray CJK character in output '
                     '(minor known drift of the RL policy).')
    note_html = f'<div class="note">{" ".join(notes)}</div>' if notes else ""
    return (f'<div class="nlaviz"><div class="title">{title}</div>'
            f'<div class="sub">{sub}</div>{chips}{"".join(rows)}{axis}{note_html}</div>')


# ── gradio wiring ────────────────────────────────────────────────────────────
def tokenize_text(text: str):
    text = (text or "").strip()
    if not text:
        return "", None, _card("Enter some text first.")
    all_ids = tok(text, add_special_tokens=True)["input_ids"]
    ids = all_ids[:MAX_TEXT_TOKENS]
    # one Rust-side batch call vs 2048 sequential decode() round-trips
    pieces = tok.batch_decode([[t] for t in ids])
    paware = [EVAL_AWARE.get(tuple(ids[: i + 1])) for i in range(len(ids))]
    return render_tokens(pieces, len(all_ids), paware), {"ids": ids, "pieces": pieces}, EMPTY_CARD


# Cross-user runtime cache: one temperature-1 sample per position (the pipeline
# depends only on the clicked token's left-context, so the key is the token-id
# PREFIX ids[:idx+1]). Hits skip the GPU entirely. LRU, ~20MB at capacity.
_CACHE_MAX = 1024
_result_cache: OrderedDict = OrderedDict()  # prefix tuple -> {lines, fve, cos}
_cache_lock = Lock()

# Baked per-position results (precache.json, made by precompute_cache.py) for
# every token of the default texts — served instantly, no ZeroGPU call. Keyed by
# the token-id prefix, immutable. A stale precache (texts/checkpoint changed)
# just misses and falls through to the live GPU path.
PRECACHE: dict = {}
_pc_entries: list = []
try:
    _pc_entries = json.load(open(os.path.join(HERE, "precache.json")))["entries"]
    for _e in _pc_entries:
        for _i, _r in enumerate(_e["results"]):
            if _r is not None:
                PRECACHE[tuple(_e["ids"][: _i + 1])] = _r
    print(f"[precache] {len(PRECACHE)} positions preloaded", flush=True)
except FileNotFoundError:
    print("[precache] no precache.json — default-text clicks compute live", flush=True)

# Optional per-token P(evaluation-awareness) overlays (eval_awareness.json, made by
# eval_awareness.py + eval_awareness_weighted.py — gpt-4o-mini judges each NLA line).
# Aligned 1:1 with the precache; keyed by the same token-id prefix. Three heatmap
# variants (raw / ΔFVE-weighted / rank-normalized-ΔFVE-weighted) + per-line scores.
EVAL_AWARE: dict = {}       # prefix -> (p, p_fvew, p_fvewnorm)
EVAL_AWARE_LINES: dict = {}  # prefix -> [per-line P(aware)]  (for the ΔFVE panel)
try:
    _ea = json.load(open(os.path.join(HERE, "eval_awareness.json")))["entries"]
    for _pe, _ee in zip(_pc_entries, _ea):
        _f = _ee.get("paware_fvew"); _fn = _ee.get("paware_fvewnorm")
        _pl = _ee.get("paware_lines")
        for _i, _p in enumerate(_ee["paware"]):
            if _p is not None:
                _key = tuple(_pe["ids"][: _i + 1])
                EVAL_AWARE[_key] = (_p, (_f[_i] if _f else None), (_fn[_i] if _fn else None))
                if _pl and _pl[_i] is not None:
                    EVAL_AWARE_LINES[_key] = _pl[_i]
    print(f"[eval-aware] {len(EVAL_AWARE)} positions preloaded "
          f"({len(EVAL_AWARE_LINES)} with per-line scores)", flush=True)
except FileNotFoundError:
    print("[eval-aware] no eval_awareness.json — heatmap toggle inert", flush=True)


# Optional per-line ablation scores (loo.json, made by loo_precompute.py on an
# A100): loo[k] = FVE of all-lines-minus-k, solo[k] = FVE of line k alone,
# full = all-lines FVE. Matryoshka lines are independent bullets, so the critic
# scores arbitrary subsets in-distribution — necessity (full−loo[k]) separates
# "unimportant" from "redundant with other lines", which prefix-ΔFVE conflates.
LOO: dict = {}  # prefix -> {"full": f, "loo": [...], "solo": [...]}
try:
    _lj = json.load(open(os.path.join(HERE, "loo.json")))["entries"]
    for _pe, _le in zip(_pc_entries, _lj):
        for _i, _l in enumerate(_le["loo"]):
            if _l is not None:
                LOO[tuple(_pe["ids"][: _i + 1])] = {
                    "full": _le["full"][_i], "loo": _l, "solo": _le["solo"][_i]}
    print(f"[loo] {len(LOO)} positions with ablation scores", flush=True)
except FileNotFoundError:
    print("[loo] no loo.json — ablation view falls back to marginal", flush=True)


def _precache_get(prefix) -> dict | None:
    r = PRECACHE.get(prefix)
    return dict(r) if r is not None else None


def _cache_get(key) -> dict | None:
    with _cache_lock:
        res = _result_cache.get(key)
        if res is None:
            return None
        _result_cache.move_to_end(key)
        return dict(res)


def _cache_put(key: tuple, res: dict) -> None:
    with _cache_lock:
        _result_cache[key] = res
        _result_cache.move_to_end(key)
        while len(_result_cache) > _CACHE_MAX:
            _result_cache.popitem(last=False)


def analyze_at(tokstate: dict | None, idx, mode: str,
               steer: str = "none", strength: float = 0.0):
    """Generator: streams (res_state, viz_html) — partial cards while the AV
    decodes, then the final scored card."""
    if not tokstate:
        yield None, _card("Tokenize some text first.")
        return
    ids, pieces = tokstate["ids"], tokstate["pieces"]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        yield None, _card("Click a token (or enter a valid position).")
        return
    if not (0 <= idx < len(ids)):
        yield None, _card(f"Position must be in [0, {len(ids) - 1}].")
        return
    key = tuple(ids[: idx + 1])
    # sig ties this analysis to its exact token prefix — on_intervene refuses to
    # patch if the text in tok_state has changed since the click.
    meta = {"token": pieces[idx].strip() or repr(pieces[idx]), "pos": idx,
            "sig": hash(key)}
    res = _precache_get(key) or _cache_get(key)
    if res is not None:
        res.update(meta)
        if key in EVAL_AWARE_LINES:            # per-line eval-awareness for the ΔFVE panel
            res["paware_lines"] = EVAL_AWARE_LINES[key]
        if key in LOO:                         # per-line ablation for the LOO view
            res["loo"] = LOO[key]
        yield res, render_viz(res, mode)
        return
    for res in gpu_analyze(ids, idx):
        res.update(meta)
        if res.get("fve") is None:
            yield gr.skip(), render_viz(res, mode)
    if res is None or not res["lines"]:
        yield None, _card("The AV produced no output for this activation — try another token.")
        return
    _cache_put(key, {"lines": res["lines"], "fve": res["fve"], "cos": res["cos"]})
    yield res, render_viz(res, mode)


INTERV_EMPTY = _card("Click a token, wait for its lines, then edit them here and continue.")


def tokenize_reset(text: str):
    """UI tokenize: besides the token panel, clear the stale analysis state —
    res_state/editor/intervention refer to positions of the PREVIOUS text."""
    tokens_html, tokstate, viz_html = tokenize_text(text)
    return tokens_html, tokstate, viz_html, None, gr.update(value=""), INTERV_EMPTY


def on_token_click(text: str, tokstate: dict | None, mode: str, steer: str,
                   strength, idx: str):
    """Outputs (tokens_out, tok_state, res_state, viz). tok_state is per-session
    server state — a Space restart under an open tab wipes it while the page
    still shows clickable tokens. Silently re-tokenize and carry on with the
    click instead of dead-ending."""
    if not tokstate:
        tokens_html, tokstate, _ = tokenize_text(text)
        if tokstate is None:
            yield gr.skip(), None, None, _card("Enter some text first.")
            return
        first = True
        for res, viz_html in analyze_at(tokstate, idx, mode, steer, strength):
            yield (tokens_html if first else gr.skip()), \
                  (tokstate if first else gr.skip()), res, viz_html
            first = False
        return
    for res, viz_html in analyze_at(tokstate, idx, mode, steer, strength):
        yield gr.skip(), gr.skip(), res, viz_html


def analyze_text(text: str, idx, mode: str, steer: str = "none", strength: float = 0.0):
    """Self-contained (text + position) — no State dependency. Powers the
    'Analyze position' button and the public API."""
    tokens_html, tokstate, _ = tokenize_text(text)
    if tokstate is None:
        yield tokens_html, None, None, _card("Enter some text first.")
        return
    for res, viz_html in analyze_at(tokstate, idx, mode, steer, strength):
        yield tokens_html, tokstate, res, viz_html


def on_mode_change(state: dict | None, mode: str):
    return render_viz(state, mode)


# ── intervention (edit the explanation → steer the continuation) ─────────────
def _mark_divergence(a: str, b: str) -> tuple[str, str]:
    """Escape both texts, highlighting everything after their common prefix —
    with seed-matched sampling the highlight starts exactly where the edit's
    causal effect kicks in."""
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    esc = html_lib.escape
    return (esc(a[:n]) + (f'<mark class="div">{esc(a[n:])}</mark>' if a[n:] else ""),
            esc(b[:n]) + (f'<mark class="div">{esc(b[n:])}</mark>' if b[n:] else ""))


def render_intervention(meta: dict, out: dict | None) -> str:
    if out is None:
        return _card("Edit (or delete) explanation lines above, then press "
                     "“Continue with & without the edit”.")
    s_o, s_e = out["stats"]
    pend = '<span class="pendc">generating…</span>'
    raw_h = html_lib.escape(out["raw"]) if out["raw"] is not None else pend
    if out["orig"] is not None and out["edit"] is not None:
        orig_h, edit_h = _mark_divergence(out["orig"], out["edit"])
    else:
        orig_h = html_lib.escape(out["orig"]) if out["orig"] is not None else pend
        edit_h = html_lib.escape(out["edit"]) if out["edit"] is not None else pend
    chips = (
        f'<div class="chips">'
        f'<div class="chip"><div class="v">{s_o["fve"]:.3f}</div><div class="l">FVE · original lines</div></div>'
        f'<div class="chip"><div class="v">{s_e["fve"]:.3f}</div><div class="l">FVE · edited lines</div></div>'
        f'<div class="chip"><div class="v">{out["shift"]:.3f}</div><div class="l">cos(v̂ orig, v̂ edit)</div></div>'
        f'<div class="chip"><div class="v">{html_lib.escape(meta.get("token", "—"))}</div>'
        f'<div class="l">patched token @ {meta.get("pos", "?")}</div></div>'
        f'</div>')
    boxes = (
        f'<div class="contbox"><div class="chd"><span>true activation</span>'
        f'<span class="tag">no patch</span></div><div class="ctx">{raw_h}</div></div>'
        f'<div class="contbox"><div class="chd"><span>reconstruction · original lines</span>'
        f'<span class="tag">control</span></div><div class="ctx">{orig_h}</div></div>'
        f'<div class="contbox treat"><div class="chd"><span>reconstruction · edited lines</span>'
        f'<span class="tag">your edit</span></div><div class="ctx">{edit_h}</div></div>')
    return (f'<div class="nlaviz"><div class="title">Continuations from the patched activation</div>'
            f'<div class="sub">the clicked token’s L42 activation is replaced by the critic’s '
            f'reconstruction of the lines; text after it is regenerated (seed-matched T=1 — '
            f'identical activations would give identical text, so the <mark class="div">'
            f'highlighted divergence</mark> is caused by your edit)</div>'
            f'{chips}<div class="contgrid">{boxes}</div></div>')


def on_intervene(tokstate: dict | None, res: dict | None, edited: str,
                 n_new: float, seed: float):
    """Generator: streams the intervention card as each continuation lands."""
    if not tokstate or not res or not res.get("lines"):
        yield _card("Analyze a token first — click one in the panel.")
        return
    if res.get("fve") is None:
        yield _card("Wait for the analysis to finish scoring, then intervene.")
        return
    orig_lines = res["lines"]
    edit_lines = [ln.strip() for ln in (edited or "").split("\n") if ln.strip()]
    if edit_lines == orig_lines:
        yield _card("The lines are unchanged — remove one, or rewrite one, then continue.")
        return
    idx = int(res["pos"])
    ids = tokstate["ids"]
    if idx >= len(ids) or res.get("sig") != hash(tuple(ids[: idx + 1])):
        yield _card("The text changed since this analysis — click a token again.")
        return
    for out in gpu_intervene(ids, idx, orig_lines, edit_lines,
                             int(n_new or 96), int(seed or 0)):
        yield render_intervention(res, out)


def res_to_editor(res: dict | None):
    """Populate the line editor (and clear any stale intervention) when a
    fresh analysis lands."""
    if res and res.get("lines") and res.get("fve") is not None:
        return gr.update(value="\n".join(res["lines"])), INTERV_EMPTY
    return gr.update(), gr.update()


with gr.Blocks(css=CSS, js=CLICK_JS, title="NLA Qwen3.6-27B explorer") as demo:
    gr.Markdown(
        "# 🔬 NLA — read Qwen3.6-27B's mind, one token at a time\n"
        "A **natural-language autoencoder** for Qwen3.6-27B layer-42 activations: click a token "
        "and the *actor* verbalizes its activation into a salience-ordered list of lines, while "
        "the *critic* reconstructs the vector from each line-prefix — the bars show how much of "
        "the vector (**FVE**, fraction of variance explained) the first *k* lines recover. "
        f"[Checkpoints](https://huggingface.co/{MODEL_REPO}) — the matryoshka `rl_av_lora_iter400` "
        "verbalizer (ordering-trained to front-load by salience), co-trained `rl_critic_step400` "
        "reconstructor.",
        elem_id="nla-header",
    )
    tok_state = gr.State(None)
    res_state = gr.State(None)
    tokens_out = gr.HTML(render=False)
    viz = gr.HTML(EMPTY_CARD, render=False)
    # created early (render=False) so the Examples click can reset them; they
    # render inside the intervention accordion below.
    edit_box = gr.Textbox(label="Explanation lines — one per line; delete or rewrite, then continue",
                          lines=10, max_lines=14, value="", render=False)
    interv_out = gr.HTML(INTERV_EMPTY, render=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text_in = gr.Textbox(label="Text", lines=5, max_lines=10,
                                 placeholder="Paste any text, then Tokenize…",
                                 value=DEFAULT_TEXTS[0])
            tokenize_btn = gr.Button("Tokenize", variant="primary")
            gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in],
                        example_labels=EXAMPLE_LABELS,
                        fn=tokenize_reset,
                        outputs=[tokens_out, tok_state, viz, res_state, edit_box, interv_out],
                        run_on_click=True, cache_examples=False,  # honeypot HTML > csv field cap
                        label="Or try one of these")
            tokens_out.render()
        with gr.Column(scale=5, elem_classes=["nla-side"]):
            mode = gr.Radio(["marginal", "cumulative", "ablation"], value="marginal",
                            label="FVE view",
                            info="ablation = FVE lost when a line is deleted with all "
                                 "others kept (leave-one-out; sample texts only)")
            heat = gr.Radio(
                ["off", "P(eval-aware)", "× ΔFVE", "× ΔFVE ÷ rank-mean"],
                value="off", label="🔴 Eval-awareness heatmap",
                info="Tint each token by its NLA explanation's eval-awareness (gpt-4o-mini). "
                     "'× ΔFVE' weights each line by its reconstruction value; the ÷rank-mean "
                     "variant removes salience front-loading. Sample texts only.")
            viz.render()
            with gr.Accordion("Analyze a token position by number", open=False):
                with gr.Row():
                    pos_in = gr.Number(label="token position", precision=0,
                                       value=None, scale=2)
                    pos_btn = gr.Button("Analyze", scale=1)
    with gr.Accordion("🧪 Causal intervention — edit the explanation, steer the model",
                      open=False):
        gr.Markdown(
            "Matryoshka explanations are **independent lines**, so the critic can encode an "
            "edited subset without going out-of-distribution. Delete or rewrite lines below; "
            "the critic re-encodes your version, the resulting vector is patched into the "
            "model at the clicked token (norm-matched), and the text after it is regenerated — "
            "side-by-side with the unedited reconstruction (control) and the true activation.")
        edit_box.render()
        with gr.Row():
            cont_len = gr.Slider(32, 160, value=96, step=16,
                                 label="continuation tokens", scale=3)
            seed_in = gr.Number(label="seed", value=0, precision=0, scale=1)
            intervene_btn = gr.Button("▶ Continue with & without the edit",
                                      variant="primary", scale=2)
        interv_out.render()
    # hidden bridge (display:none): CLICK_JS writes the clicked token index here.
    click_idx = gr.Textbox(value="", label="clicked token index", elem_id="nla-click-idx")
    # kept for output-signature parity with the v3 handlers (steering hidden)
    steer_dd = gr.Dropdown(["none"], value="none", visible=False)
    strength_in = gr.Slider(0.0, 20.0, value=0.0, visible=False)

    # heatmap toggle: pure client-side class flip on <body> (survives token re-renders)
    heat.change(None, [heat], None, js="""(v) => {
        const b = document.body.classList;
        b.remove('nla-heat-p', 'nla-heat-f', 'nla-heat-fn');
        const cls = {'P(eval-aware)': 'nla-heat-p', '× ΔFVE': 'nla-heat-f',
                     '× ΔFVE ÷ rank-mean': 'nla-heat-fn'}[v];
        if (cls) b.add(cls);
    }""")
    _reset_outs = [tokens_out, tok_state, viz, res_state, edit_box, interv_out]
    tokenize_btn.click(tokenize_reset, [text_in], _reset_outs, api_name="tokenize")
    text_in.submit(tokenize_reset, [text_in], _reset_outs)
    # Both GPU entry points share one concurrency slot (concurrency_id="gpu",
    # limit 1): gpu_analyze mutates the module-global vref[0] during generation,
    # so concurrent runs would clobber each other's injected vector. Serialize.
    click_idx.input(on_token_click,
                    [text_in, tok_state, mode, steer_dd, strength_in, click_idx],
                    [tokens_out, tok_state, res_state, viz],
                    concurrency_id="gpu", concurrency_limit=1
                    ).then(res_to_editor, [res_state], [edit_box, interv_out])
    pos_btn.click(analyze_text, [text_in, pos_in, mode, steer_dd, strength_in],
                  [tokens_out, tok_state, res_state, viz], api_name="analyze",
                  concurrency_id="gpu", concurrency_limit=1
                  ).then(res_to_editor, [res_state], [edit_box, interv_out])
    mode.change(on_mode_change, [res_state, mode], [viz])
    # gpu_intervene also mutates patch_ref[0] → same serialized GPU slot.
    intervene_btn.click(on_intervene,
                        [tok_state, res_state, edit_box, cont_len, seed_in],
                        [interv_out], api_name="intervene",
                        concurrency_id="gpu", concurrency_limit=1)
    demo.load(tokenize_reset, [text_in], _reset_outs)

# CPU events (tokenize/mode) run in parallel; GPU events serialize via the
# shared "gpu" concurrency_id above.
demo.queue(default_concurrency_limit=4)
demo.launch()
