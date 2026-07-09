"""NLA v3 explorer — click a token, read its activation as ten explanation lines.

Pipeline per click (all on a ZeroGPU slice):
  1. EXTRACT   full Qwen2.5-7B-Instruct forward over the text with a block-20
               output hook (aborts after L20 — numerically identical to the
               truncated-extractor convention training used); the layer-20
               hidden state at the clicked token is the activation vector v.
               The full model also powers the INTERVENE feature: edit the
               explanation lines, the critic re-encodes them, the vector is
               patched back at the clicked position (norm-matched) and the
               continuation is regenerated seed-matched with & without the
               edit — causal steering through the NLA explanation.
  2. VERBALIZE the v3 AV (actor): inject normalize(v, injection_scale) at the
               ㈎ marker embedding, sample a newline list at temperature 1
               (matching the RL rollout distribution), keep N_LINES lines.
  3. RECONSTRUCT the v3 AR (critic) reads cumulative line prefixes (1..k) and
               predicts v̂_k; FVE_k = 1 − ||n(v̂_k)−n(v)||² / ||n(v)−μ||² with
               μ = population mean of normalized held-out activations (mu.npy).

Everything model-specific (prompt template, marker token ids, scales) is read
from the checkpoints' nla_meta.yaml sidecars — nothing hardcoded.
"""

import spaces  # must be imported before any CUDA touch

import html as html_lib
import json
import re
from collections import OrderedDict
from threading import Lock, Thread

import gradio as gr
import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from nla_inference import NLACritic

RL_REPO = "syvb/nla-qwen2.5-7b-L20-v3-rl"
RL_ITER = "iter_0000200"
EXTRACTOR_ID = "Qwen/Qwen2.5-7B-Instruct"
N_LINES = 9  # lines 10+ tend to degenerate — cap before quality falls off
MAX_TEXT_TOKENS = 2048
GOOD_MIN_POS = 10  # early positions have little left-context; training used >=50

# ── checkpoints + sidecar config ────────────────────────────────────────────
root = snapshot_download(RL_REPO, allow_patterns=[f"{RL_ITER}/*"])
AV_DIR, AR_DIR = f"{root}/{RL_ITER}/av", f"{root}/{RL_ITER}/ar"

av_meta = yaml.safe_load(open(f"{AV_DIR}/nla_meta.yaml"))
ar_meta = yaml.safe_load(open(f"{AR_DIR}/nla_meta.yaml"))
TPL = av_meta["prompt_templates"].get("av") or av_meta["prompt_templates"]["actor"]
TOK_META = av_meta["tokens"]
INJ_ID = TOK_META["injection_token_id"]
INJ_CHAR = TOK_META["injection_char"]
INJECTION_SCALE = float(av_meta["extraction"]["injection_scale"])
LAYER = (ar_meta.get("critic") or {}).get("extraction_layer_index")
assert LAYER is not None, "AR sidecar missing critic.extraction_layer_index"
assert "<explanation>" not in TPL and "bullet" in TPL, "unexpected AV template"

MU = torch.tensor(np.load("mu.npy"), dtype=torch.float32)  # normalized-space mean
DEFAULT_TEXTS = json.load(open("default_texts.json"))
CJK_RE = re.compile(r"[　-ヿ㐀-䶿一-鿿＀-￯]")

# Unit L20 trait directions (build_steering_dirs.py — the "genuine" CAA-style
# directions from caa_steering_v2/FRONTLOADING.md). Steering adds r·‖v‖·d̂ to
# the clicked activation before verbalization, matching that experiment.
STEER_DIRS: dict[str, torch.Tensor] = {}
try:
    _z = np.load("steering_dirs.npz")
    STEER_DIRS = {k.removesuffix("_L20_genuine_unit"): torch.tensor(_z[k], dtype=torch.float32)
                  for k in _z.files if k.endswith("_L20_genuine_unit")}
    print(f"[steer] directions loaded: {sorted(STEER_DIRS)}")
except FileNotFoundError:
    print("[steer] no steering_dirs.npz — steering UI hidden")

# ── models (global scope; ZeroGPU manages the .to("cuda")) ──────────────────
tok = AutoTokenizer.from_pretrained(AV_DIR)

av = AutoModelForCausalLM.from_pretrained(AV_DIR, torch_dtype=torch.bfloat16)
av.to("cuda").eval()

# The FULL subject model serves two roles: extraction (a forward hook grabs
# the block-LAYER output and aborts the forward — numerically identical to
# training's truncated-extractor convention of layers[:LAYER+1] + reading the
# pre-norm block-LAYER output) and the intervention feature's patched
# continuation, which needs the full depth to keep generating. One full copy
# costs ~4GB more than the old truncated extractor and stays within the slice.
subject = AutoModelForCausalLM.from_pretrained(EXTRACTOR_ID, torch_dtype=torch.bfloat16)
subject.to("cuda").eval()
SUBJ_LAYERS = subject.model.layers


class _StopForward(Exception):
    """Raised from the block-LAYER hook to abort the forward once the
    activation is grabbed — blocks LAYER+1.. never run for extraction."""


@torch.inference_mode()
def _extract(prefix_ids: list[int]) -> torch.Tensor:
    """Block-LAYER output at the last token of the causal prefix (fp32, cuda)."""
    grabbed = {}

    def grab(module, inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()
        raise _StopForward

    h = SUBJ_LAYERS[LAYER].register_forward_hook(grab)
    try:
        ids = torch.tensor([prefix_ids], device="cuda")
        try:
            subject(input_ids=ids, use_cache=False)
        except _StopForward:
            pass
        return grabbed["h"][0, -1].float()
    finally:
        h.remove()


# ── block-LAYER activation patch (the intervention feature) ──────────────────
# patch_ref[0] is None (no-op) or (position, fp32 vector at raw-activation
# scale): during the prefill chunk that computes `position`, the block-LAYER
# output there is replaced — everything downstream (later blocks, the KV that
# future tokens attend to, and the next sampled token) sees the patched
# activation. Decode steps have q_len=1 ≠ position+1 and pass through.
patch_ref: list = [None]


def _layer_patch(module, inputs, output):
    pr = patch_ref[0]
    if pr is None:
        return
    pos, vec = pr
    h = output[0] if isinstance(output, tuple) else output
    if h.shape[1] == pos + 1:  # exactly the prefill chunk ending at the target
        h[:, pos, :] = vec.to(h.dtype)
    return output


SUBJ_LAYERS[LAYER].register_forward_hook(_layer_patch)

critic = NLACritic(AR_DIR, device="cuda")
MSE_SCALE = critic.mse_scale

# Fixed AV prompt (the user text is never shown to the AV — only the vector).
_prompt_ids = tok.apply_chat_template(
    [{"role": "user", "content": TPL.format(injection_char=INJ_CHAR)}],
    tokenize=True, add_generation_prompt=True, return_tensors="pt",
)
_inj_pos = (_prompt_ids[0] == INJ_ID).nonzero(as_tuple=True)[0]
assert len(_inj_pos) == 1, f"marker appears {len(_inj_pos)}x in prompt"
INJ_POS = int(_inj_pos[0])


def _normalize(v: torch.Tensor, scale: float) -> torch.Tensor:
    return v / v.norm().clamp_min(1e-12) * scale


def _complete_lines(text: str) -> list[str]:
    """Non-empty lines that are newline-terminated (drops a trailing partial)."""
    return [ln.strip() for ln in text.split("\n")[:-1] if ln.strip()]


class _StopAfterLines(StoppingCriteria):
    """End decoding once N_LINES complete non-empty lines exist — don't burn
    decode steps on lines the UI will discard. With inputs_embeds, input_ids
    here is the generated portion only (matches the decode below)."""

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        text = tok.decode(input_ids[0], skip_special_tokens=True)
        return len(_complete_lines(text)) >= N_LINES


# duration: measured worst case ≈11s end-to-end (2048-token prefix, cold
# worker attach). 30s covers a pathologically slow cold weight-transfer while
# still beating the 60s default — shorter budgets get better ZeroGPU queue
# priority for visitors, so don't pad this further.
@spaces.GPU(duration=30)
def gpu_analyze(token_ids: list[int], idx: int, steer: str | None = None,
                strength: float = 0.0):
    """Extract activation at token idx, verbalize, score cumulative prefixes.

    steer/strength: add strength·‖v‖·d̂_steer to the extracted activation
    before verbalization (CAA-style; FVE is then measured against the steered
    vector — that's what was verbalized).

    Generator: yields {"lines": [...], "fve": None} as each explanation line
    finishes decoding (streamed to the UI), then the final dict with fve/cos.
    """
    with torch.inference_mode():
        # Causal attention ⇒ extracting at the last token of the prefix is
        # identical to position idx of the full text (and is how training
        # data was built) — don't forward the suffix.
        v = _extract(token_ids[: idx + 1])
        if steer:
            v = v + strength * v.norm() * STEER_DIRS[steer].to(v.device)

        emb = av.get_input_embeddings()(_prompt_ids.to("cuda")).clone()
        emb[0, INJ_POS] = _normalize(v, INJECTION_SCALE).to(torch.bfloat16)
        streamer = TextIteratorStreamer(tok, skip_special_tokens=True)
        gen = Thread(target=av.generate, kwargs=dict(  # generate() is no_grad
            inputs_embeds=emb,
            attention_mask=torch.ones(1, emb.shape[1], device="cuda", dtype=torch.long),
            # temperature-1 sampling (never greedy) — matches the RL rollout
            # distribution. top_p/top_k passed explicitly so a checkpoint's
            # generation_config.json can never silently reshape the sampling.
            max_new_tokens=220, do_sample=True, temperature=1.0, top_p=1.0,
            top_k=0, pad_token_id=tok.eos_token_id,
            stopping_criteria=StoppingCriteriaList([_StopAfterLines()]),
            streamer=streamer,
        ))
        gen.start()
        text, shown = "", 0
        for piece in streamer:
            text += piece
            done = _complete_lines(text)
            if len(done) > shown:
                shown = len(done)
                yield {"lines": done[:N_LINES], "fve": None}
        gen.join()
        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()][:N_LINES]
        if not lines:
            yield {"lines": [], "fve": [], "cos": []}
            return

        gold_n = _normalize(v.cpu(), MSE_SCALE)
        denom = ((gold_n - MU) ** 2).mean().item()
        preds = critic.reconstruct_batch(
            ["\n".join(lines[:k]) for k in range(1, len(lines) + 1)])
        fve, cos = [], []
        for pred in preds:
            pred_n = _normalize(pred, MSE_SCALE)
            fve.append(1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom)
            cos.append(float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm())))
    yield {"lines": lines, "fve": fve, "cos": cos}


@torch.inference_mode()
def _continue(prefix_ids: list[int], n_new: int, seed: int,
              patch_vec: torch.Tensor | None = None) -> str:
    """Continue the ORIGINAL text after the clicked token with the subject
    model, optionally rewriting the block-LAYER output at the clicked position
    (the last prefix token) with `patch_vec` (fp32, raw-activation scale).
    Seed-matched T=1 sampling: identical activations ⇒ identical text, so
    divergence between two conditions is caused by the patch."""
    ids = torch.tensor([prefix_ids], device="cuda")
    patch_ref[0] = (None if patch_vec is None
                    else (len(prefix_ids) - 1, patch_vec.to("cuda")))
    torch.manual_seed(int(seed))
    try:
        out = subject.generate(
            input_ids=ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=int(n_new), do_sample=True, temperature=1.0,
            top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    finally:
        patch_ref[0] = None
    return tok.decode(out[0, len(prefix_ids):], skip_special_tokens=True)


# duration=60: one extraction + one critic batch + three ≤160-token 7B
# continuations (seed-matched conditions run sequentially — batching would
# interleave the RNG stream and break "same logits ⇒ same text").
@spaces.GPU(duration=60)
def gpu_intervene(token_ids: list[int], idx: int, orig_lines: list[str],
                  edit_lines: list[str], n_new: int, seed: int):
    """Generator: reconstruction stats first, then the three continuations as
    each finishes — no-patch (true activation), v̂(original lines) (control:
    reconstruction error alone), v̂(edited lines) (treatment)."""
    prefix = token_ids[: idx + 1]
    # no outer inference_mode: it must not span the yields (generator resumes
    # can land on other threads) — _extract/_continue/reconstruct_batch each
    # guard themselves, and the arithmetic here is detached fp32 on CPU.
    v = _extract(prefix).cpu()
    gold_n = _normalize(v, MSE_SCALE)
    denom = ((gold_n - MU) ** 2).mean().item()
    preds = critic.reconstruct_batch(["\n".join(orig_lines), "\n".join(edit_lines)])
    stats, vecs = [], []
    for pred in preds:
        pred_n = _normalize(pred, MSE_SCALE)
        stats.append({
            "fve": 1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom,
            "cos": float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm()))})
        # patch at the RAW activation's magnitude — direction from the critic
        vecs.append(pred / pred.norm().clamp_min(1e-12) * v.norm())
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
  // Debounced steering-strength bridge: a drag fires an input event per tick,
  // which would launch a GPU run at every intermediate value. Instead the
  // slider is NOT wired to the backend directly — we wait until it has
  // settled for 500ms, then poke the hidden #nla-steer-r box, whose .input
  // listener re-runs with the slider's (final) value.
  let steerT = null;
  document.addEventListener('input', (e) => {
    const s = e.target.closest && e.target.closest('#nla-strength input');
    if (!s) return;
    clearTimeout(steerT);
    steerT = setTimeout(() => {
      const box = document.querySelector('#nla-steer-r textarea, #nla-steer-r input');
      if (!box) return;
      box.value = s.value;
      box.dispatchEvent(new Event('input', { bubbles: true }));
    }, 500);
  });
}
"""


def _card(msg: str) -> str:
    return f'<div class="nlaviz"><div class="empty">{msg}</div></div>'


EMPTY_CARD = _card("👈 Click any token to read its activation.")


def render_tokens(pieces: list[str], n_total: int) -> str:
    spans = []
    for i, p in enumerate(pieces):
        body = html_lib.escape(p).replace("\n", '<span class="nl">⏎</span><br>')
        spans.append(f'<span class="nla-tok" data-i="{i}" title="#{i}" '
                     f'role="button" tabindex="0">{body}</span>')
    trunc = (f' <span class="trunc">✂ truncated to the first {len(pieces)}</span>'
             if n_total > len(pieces) else "")
    head = (f'<div class="tokhead"><span>{len(pieces)} tokens{trunc}</span>'
            f'<span>click a token to analyze it</span></div>')
    return f'<div class="tokpanel">{head}<div class="tokscroll">{"".join(spans)}</div></div>'


def _steer_chip(state: dict) -> str:
    if not state.get("steer"):
        return ""
    return (f'<div class="chip"><div class="v">{html_lib.escape(state["steer"])} '
            f'· r={state["strength"]:g}</div><div class="l">steered</div></div>')


def render_pending(state: dict, mode: str) -> str:
    """Streaming view: lines appear as they decode; bars are shimmer stubs."""
    title = ("Additional FVE per explanation line (ΔFVE)" if mode == "marginal"
             else "Cumulative round-trip FVE by explanation-line prefix")
    tok_piece = html_lib.escape(state.get("token", ""))
    chips = (
        f'<div class="chips">'
        f'<div class="chip"><div class="v">…</div><div class="l">FVE</div></div>'
        f'<div class="chip"><div class="v">…</div><div class="l">cosine</div></div>'
        f'<div class="chip"><div class="v">{tok_piece or "—"}</div><div class="l">token @ {state.get("pos", "?")}</div></div>'
        f'{_steer_chip(state)}'
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
    vals = marginal if mode == "marginal" else fve
    title = ("Additional FVE per explanation line (ΔFVE)" if mode == "marginal"
             else "Cumulative round-trip FVE by explanation-line prefix")
    sub = ("how much reconstruction each successive line adds" if mode == "marginal"
           else "reconstruction quality from lines 1..k only")

    lo, hi = min(0.0, min(vals)), max(0.05, max(vals))
    span = hi - lo
    zero_pct = (0.0 - lo) / span * 100

    rows = []
    for i, (ln, x) in enumerate(zip(lines, vals)):
        w = abs(x) / span * 100
        left = zero_pct if x >= 0 else zero_pct - w
        cls = "pos" if x >= 0 else "neg"
        tip = (f"line {i + 1} — ΔFVE {marginal[i]:+.3f}, cumulative {fve[i]:.3f}, "
               f"cos {cos[i]:.3f}")
        rows.append(
            f'<div class="row" title="{html_lib.escape(tip)}">'
            f'<div class="idx">{i + 1}</div>'
            f'<div class="line" title="{html_lib.escape(ln)}">{html_lib.escape(ln)}</div>'
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
        f'{_steer_chip(state)}'
        f'</div>'
    )
    notes = []
    if state.get("pos", GOOD_MIN_POS) < GOOD_MIN_POS:
        notes.append('<span class="warnic">⚠</span> very early position — little left-context; '
                     'explanations may be generic (training sampled positions ≥ 50).')
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
    return render_tokens(pieces, len(all_ids)), {"ids": ids, "pieces": pieces}, EMPTY_CARD


# Cross-user result cache: one temperature-1 sample per position, first
# computation wins and everyone sees that stored sample afterwards. The
# pipeline depends only on the clicked token's left-context, so the key is the
# token-id PREFIX ids[:idx+1]. Hits skip the GPU queue and visitor quota
# entirely — repeat clicks (especially on the preloaded example texts) drop
# from seconds to network latency. Lives in the main process (not the ZeroGPU
# worker), so it survives worker eviction. ~20MB at capacity.
#
# Two levels: PRECACHE (baked by precompute_cache.py for every position of the
# default texts; immutable, never evicted) then a runtime LRU for everything
# else. A stale precache (texts changed, tokenizer drift) just misses — clicks
# fall through to the GPU path.
_CACHE_MAX = 1024
_result_cache: OrderedDict = OrderedDict()  # prefix tuple -> {lines, fve, cos}
_cache_lock = Lock()

PRECACHE: dict = {}
try:
    for _entry in json.load(open("precache.json"))["entries"]:
        for _i, _res in enumerate(_entry["results"]):
            if _res is not None:
                PRECACHE[tuple(_entry["ids"][: _i + 1])] = _res
    print(f"[precache] {len(PRECACHE)} positions preloaded")
except FileNotFoundError:
    print("[precache] no precache.json — default-text clicks compute live")


def _precache_get(prefix: tuple) -> dict | None:
    res = PRECACHE.get(prefix)
    return dict(res) if res is not None else None


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
    # strength 0 (or unknown trait) ⇒ plain unsteered analysis, shared cache
    strength = _clamp_strength(strength)
    steer_key = steer if steer in STEER_DIRS and strength > 0 else None
    # sig ties this analysis to its exact token prefix — on_intervene refuses
    # to patch if the text in tok_state has changed since the click.
    meta = {"token": pieces[idx].strip() or repr(pieces[idx]), "pos": idx,
            "sig": hash(tuple(ids[: idx + 1]))}
    if steer_key:
        meta.update(steer=steer_key, strength=strength)
    prefix = tuple(ids[: idx + 1])
    key = (prefix, steer_key, strength) if steer_key else prefix
    res = None if steer_key else _precache_get(prefix)
    if res is None:
        res = _cache_get(key)
    if res is not None:
        res.update(meta)
        yield res, render_viz(res, mode)
        return
    for res in gpu_analyze(ids, idx, steer_key, strength):
        res.update(meta)
        if res.get("fve") is None:
            yield gr.skip(), render_viz(res, mode)
    if res is None or not res["lines"]:
        yield None, _card("The AV produced no output for this activation — try another token.")
        return
    _cache_put(key, {"lines": res["lines"], "fve": res["fve"], "cos": res["cos"]})
    yield res, render_viz(res, mode)


def on_token_click(text: str, tokstate: dict | None, mode: str, steer: str,
                   strength, idx: str):
    """Outputs (tokens_out, tok_state, res_state, viz). tok_state is
    per-session server state — a Space restart under an open tab wipes it
    while the page still shows clickable tokens. Instead of dead-ending
    ("tokenize first"), silently re-tokenize the textbox content and carry
    on with the click."""
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


def _clamp_strength(x) -> float:
    return round(max(0.0, min(20.0, float(x or 0.0))), 3)


def on_steer_change(tokstate: dict | None, res: dict | None, mode: str,
                    steer: str, strength):
    """Re-analyze the currently selected token under the new steering setting.

    Wired to the .input events of both controls (NOT slider.release — that
    only fires on mouse-drag release, so keyboard arrows / typed values / some
    track-clicks would silently do nothing) with trigger_mode="always_last"
    (drag ticks collapse to the final value) and concurrency_limit=1 (runs
    serialize, so the latest setting always wins the display)."""
    if not tokstate or not res:
        yield gr.skip(), gr.skip()
        return
    s = _clamp_strength(strength)
    want = (steer, s) if steer in STEER_DIRS and s > 0 else (None, 0.0)
    have = ((res["steer"], res.get("strength", 0.0)) if res.get("steer")
            else (None, 0.0))
    if want == have:  # duplicate/no-op event — the shown result already matches
        yield gr.skip(), gr.skip()
        return
    yield from analyze_at(tokstate, res.get("pos"), mode, steer, strength)


def on_mode_change(state: dict | None, mode: str):
    return render_viz(state, mode)


# ── intervention (edit the explanation → steer the continuation) ─────────────
INTERV_EMPTY = _card("Click a token, wait for its lines, then edit them here and continue.")


def tokenize_reset(text: str):
    """UI tokenize: besides the token panel, clear the stale analysis state —
    res_state/editor/intervention refer to positions of the PREVIOUS text."""
    tokens_html, tokstate, viz_html = tokenize_text(text)
    return tokens_html, tokstate, viz_html, None, gr.update(value=""), INTERV_EMPTY


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
            f'<div class="sub">the clicked token’s L{LAYER} activation is replaced by the critic’s '
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
    if res.get("steer"):
        yield _card("Intervention needs an unsteered analysis — set the trait "
                    "direction back to “none”, re-click the token, then edit.")
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
    fresh unsteered analysis lands."""
    if res and res.get("lines") and res.get("fve") is not None and not res.get("steer"):
        return gr.update(value="\n".join(res["lines"])), INTERV_EMPTY
    return gr.update(), gr.update()


with gr.Blocks(css=CSS, js=CLICK_JS, title="NLA v3 explorer") as demo:
    gr.Markdown(
        "# 🔬 NLA v3 — read a language model's mind, one token at a time\n"
        "A **natural-language autoencoder** for Qwen2.5-7B layer-20 activations: click a token "
        "and the *actor* verbalizes its activation into a salience-ordered list of lines, while "
        "the *critic* reconstructs the vector from each line-prefix — the bars show how much of "
        "the vector (**FVE**, fraction of variance explained) the first *k* lines recover. "
        "Truncation-RL trained the actor to front-load what matters. "
        "You can also **steer** the clicked activation along a trait direction "
        "(sycophancy / neuroticism / yellow) at adjustable strength before it's verbalized. "
        f"[AV/AR checkpoints](https://huggingface.co/{RL_REPO}) · iter 200, KL 0.03, "
        "U[1,120]-token truncation.",
        elem_id="nla-header",
    )
    tok_state = gr.State(None)
    res_state = gr.State(None)
    tokens_out = gr.HTML(render=False)
    viz = gr.HTML(EMPTY_CARD, render=False)
    # created early (render=False) so the Examples click can reset them; they
    # render inside the intervention accordion below.
    edit_box = gr.Textbox(label="Explanation lines — one per line; delete or rewrite, then continue",
                          lines=9, max_lines=12, value="", render=False)
    interv_out = gr.HTML(INTERV_EMPTY, render=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text_in = gr.Textbox(label="Text", lines=5, max_lines=10,
                                 placeholder="Paste any text, then Tokenize…",
                                 value=DEFAULT_TEXTS[0])
            tokenize_btn = gr.Button("Tokenize", variant="primary")
            gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in],
                        fn=tokenize_reset,
                        outputs=[tokens_out, tok_state, viz, res_state, edit_box, interv_out],
                        run_on_click=True, label="Or try one of these")
            tokens_out.render()
        with gr.Column(scale=5, elem_classes=["nla-side"]):
            mode = gr.Radio(["marginal", "cumulative"], value="marginal", label="FVE view")
            viz.render()
            with gr.Accordion("Steer the activation", open=False,
                              visible=bool(STEER_DIRS)):
                with gr.Row():
                    steer_dd = gr.Dropdown(["none"] + sorted(STEER_DIRS),
                                           value="none", label="trait direction", scale=2)
                    # greyed out until a trait is picked — moving it with
                    # "none" selected would (correctly but confusingly) do nothing
                    strength_in = gr.Slider(0.0, 20.0, value=0.6, step=0.1,
                                            label="strength r", scale=3,
                                            interactive=False,
                                            elem_id="nla-strength")
                gr.Markdown(
                    "Pick a trait to enable the slider. Steering adds **r·‖v‖·d̂** to "
                    "the clicked activation before the actor verbalizes it — the "
                    "CAA-style *genuine* trait directions from the front-loading "
                    "experiments. The trait typically enters the list around r≈0.3 and "
                    "reaches line 1 by r≈1; past r≈2 the direction dominates the "
                    "activation. Changing these re-runs the selected token.")
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
    # hidden bridges (display:none): CLICK_JS writes the clicked token index /
    # the debounced slider value here
    click_idx = gr.Textbox(value="", label="clicked token index", elem_id="nla-click-idx")
    steer_r_settled = gr.Textbox(value="", label="settled strength", elem_id="nla-steer-r")

    _reset_outs = [tokens_out, tok_state, viz, res_state, edit_box, interv_out]
    tokenize_btn.click(tokenize_reset, [text_in], _reset_outs, api_name="tokenize")
    text_in.submit(tokenize_reset, [text_in], _reset_outs)
    click_idx.input(on_token_click,
                    [text_in, tok_state, mode, steer_dd, strength_in, click_idx],
                    [tokens_out, tok_state, res_state, viz]
                    ).then(res_to_editor, [res_state], [edit_box, interv_out])
    pos_btn.click(analyze_text, [text_in, pos_in, mode, steer_dd, strength_in],
                  [tokens_out, tok_state, res_state, viz], api_name="analyze"
                  ).then(res_to_editor, [res_state], [edit_box, interv_out])
    # gpu_intervene mutates patch_ref[0] + the shared RNG — serialize it.
    intervene_btn.click(on_intervene,
                        [tok_state, res_state, edit_box, cont_len, seed_in],
                        [interv_out], api_name="intervene", concurrency_limit=1)
    steer_dd.input(lambda steer: gr.update(interactive=steer in STEER_DIRS),
                   [steer_dd], [strength_in], show_progress="hidden")
    # The slider is deliberately NOT a trigger — its per-tick .input events
    # would re-run at every intermediate drag position. The debounced JS
    # bridge fires steer_r_settled.input once the slider settles; the
    # handler then reads the slider's current (final) value.
    gr.on([steer_dd.input, steer_r_settled.input], on_steer_change,
          [tok_state, res_state, mode, steer_dd, strength_in], [res_state, viz],
          trigger_mode="always_last", concurrency_limit=1)
    mode.change(on_mode_change, [res_state, mode], [viz])
    demo.load(tokenize_reset, [text_in], _reset_outs)

# Gradio's default_concurrency_limit=1 would serialize ALL visitors through
# one event slot; ZeroGPU runs concurrent GPU tasks fine (each attaches its
# own slice), so let clicks from different users proceed in parallel.
demo.queue(default_concurrency_limit=4)
demo.launch()
