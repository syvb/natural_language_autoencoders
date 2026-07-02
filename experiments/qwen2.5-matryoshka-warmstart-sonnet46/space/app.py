"""NLA v3 explorer — click a token, read its activation as ten explanation lines.

Pipeline per click (all on a ZeroGPU slice):
  1. EXTRACT   Qwen2.5-7B-Instruct (truncated to 20 layers) forward over the
               text; the layer-20 hidden state at the clicked token is the
               activation vector v. Causal attention ⇒ identical to extracting
               at the last token of the truncated prefix (how training data
               was built).
  2. VERBALIZE the v3 AV (actor): inject normalize(v, injection_scale) at the
               ㈎ marker embedding, greedy-decode a newline list, keep 10 lines.
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

import gradio as gr
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

# ── models (global scope; ZeroGPU manages the .to("cuda")) ──────────────────
tok = AutoTokenizer.from_pretrained(AV_DIR)

av = AutoModelForCausalLM.from_pretrained(AV_DIR, torch_dtype=torch.bfloat16)
av.to("cuda").eval()

extractor = AutoModelForCausalLM.from_pretrained(EXTRACTOR_ID, torch_dtype=torch.bfloat16)
# Training's extractor (01_extract_activations.py) kept layers[:LAYER+1] and
# hooked the output of block index LAYER — the PRE-norm residual after LAYER+1
# blocks (the critic is likewise a LAYER+1-block truncation). Mirror that:
# truncate, neutralize the final RMSNorm (HF applies it to the last
# hidden_states entry), and read hidden_states[-1] = raw block-LAYER output.
extractor.model.layers = extractor.model.layers[: LAYER + 1]
extractor.model.norm = torch.nn.Identity()
extractor.lm_head = torch.nn.Identity()
extractor.to("cuda").eval()

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


@spaces.GPU(duration=75)
def gpu_analyze(token_ids: list[int], idx: int) -> dict:
    """Extract activation at token idx, verbalize, score cumulative prefixes."""
    with torch.inference_mode():
        ids = torch.tensor([token_ids], device="cuda")
        hs = extractor.model(ids, use_cache=False, output_hidden_states=True).hidden_states
        v = hs[-1][0, idx].float()  # pre-norm output of block LAYER (norm is Identity)

        emb = av.get_input_embeddings()(_prompt_ids.to("cuda")).clone()
        emb[0, INJ_POS] = _normalize(v, INJECTION_SCALE).to(torch.bfloat16)
        out = av.generate(
            inputs_embeds=emb,
            attention_mask=torch.ones(1, emb.shape[1], device="cuda", dtype=torch.long),
            max_new_tokens=220, do_sample=False, pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()][:N_LINES]

        gold_n = _normalize(v.cpu(), MSE_SCALE)
        denom = ((gold_n - MU) ** 2).mean().item()
        fve, cos = [], []
        for k in range(1, len(lines) + 1):
            pred = critic.reconstruct("\n".join(lines[:k]))
            pred_n = _normalize(pred, MSE_SCALE)
            fve.append(1.0 - ((pred_n - gold_n) ** 2).mean().item() / denom)
            cos.append(float(pred_n @ gold_n / (pred_n.norm() * gold_n.norm())))
    return {"lines": lines, "fve": fve, "cos": cos}


# ── viz (palette per the validated reference set; text wears ink, not series) ─
CSS = """
.nlaviz{
  --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --pos:#2a78d6; --neg:#e34948;
  --wash:rgba(11,11,11,.04); --ring:rgba(11,11,11,.10);
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:16px 18px; color:var(--ink);
}
.dark .nlaviz{
  --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --pos:#3987e5; --neg:#e66767;
  --wash:rgba(255,255,255,.05); --ring:rgba(255,255,255,.10);
}
.nlaviz .title{font-size:13px; font-weight:600; margin-bottom:2px;}
.nlaviz .sub{font-size:11.5px; color:var(--ink2); margin-bottom:12px;}
.nlaviz .chips{display:flex; gap:18px; margin-bottom:14px; flex-wrap:wrap;}
.nlaviz .chip .v{font-size:20px; font-weight:650;}
.nlaviz .chip .l{font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em;}
.nlaviz .row{display:grid; grid-template-columns:18px minmax(180px,46%) 1fr 62px;
  gap:10px; align-items:center; padding:5px 6px; border-radius:6px;}
.nlaviz .row:hover{background:var(--wash);}
.nlaviz .idx{font-size:11px; color:var(--muted); text-align:right;
  font-variant-numeric:tabular-nums;}
.nlaviz .line{font-size:12.5px; color:var(--ink); white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis;}
.nlaviz .track{position:relative; height:14px;}
.nlaviz .zero{position:absolute; top:-3px; bottom:-3px; width:1px; background:var(--axis);}
.nlaviz .bar{position:absolute; top:2px; height:10px;}
.nlaviz .bar.pos{background:var(--pos); border-radius:0 4px 4px 0;}
.nlaviz .bar.neg{background:var(--neg); border-radius:4px 0 0 4px;}
.nlaviz .val{font-size:11.5px; color:var(--ink2); text-align:right;
  font-variant-numeric:tabular-nums;}
.nlaviz .axisrow{display:grid; grid-template-columns:18px minmax(180px,46%) 1fr 62px;
  gap:10px; padding:2px 6px 0;}
.nlaviz .axislab{position:relative; height:14px; font-size:10px; color:var(--muted);
  font-variant-numeric:tabular-nums;}
.nlaviz .axislab span{position:absolute; transform:translateX(-50%);}
.nlaviz .note{font-size:11px; color:var(--muted); margin-top:10px;}
.nlaviz .warn{color:#c98500;}
.tokwrap .token{cursor:pointer;}
"""


def render_viz(state: dict | None, mode: str) -> str:
    if not state:
        return ""
    lines, fve = state["lines"], state["fve"]
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
               f"cos {state['cos'][i]:.3f}")
        rows.append(
            f'<div class="row" title="{html_lib.escape(tip)}">'
            f'<div class="idx">{i + 1}</div>'
            f'<div class="line">{html_lib.escape(ln)}</div>'
            f'<div class="track"><div class="zero" style="left:{zero_pct:.2f}%"></div>'
            f'<div class="bar {cls}" style="left:{left:.2f}%;width:{max(w, 0.4):.2f}%"></div></div>'
            f'<div class="val">{x:+.3f}</div></div>'
        )
    axis = (f'<div class="axisrow"><div></div><div></div>'
            f'<div class="axislab"><span style="left:{zero_pct:.2f}%">0</span>'
            f'<span style="left:100%">{hi:.2f}</span></div><div></div></div>')

    full_fve, full_cos = fve[-1], state["cos"][-1]
    tok_piece = html_lib.escape(state.get("token", ""))
    chips = (
        f'<div class="chips">'
        f'<div class="chip"><div class="v">{full_fve:.3f}</div><div class="l">FVE · all {len(lines)} lines</div></div>'
        f'<div class="chip"><div class="v">{full_cos:.3f}</div><div class="l">cosine</div></div>'
        f'<div class="chip"><div class="v">{tok_piece or "—"}</div><div class="l">token @ {state.get("pos", "?")}</div></div>'
        f'</div>'
    )
    notes = []
    if state.get("pos", GOOD_MIN_POS) < GOOD_MIN_POS:
        notes.append('<span class="warn">⚠ very early position — little left-context; '
                     'explanations may be generic (training sampled positions ≥ 50).</span>')
    if any(CJK_RE.search(ln) for ln in lines):
        notes.append('<span class="warn">⚠ stray CJK character in output (minor known drift of the RL policy).</span>')
    note_html = f'<div class="note">{" ".join(notes)}</div>' if notes else ""
    return (f'<div class="nlaviz"><div class="title">{title}</div>'
            f'<div class="sub">{sub}</div>{chips}{"".join(rows)}{axis}{note_html}</div>')


# ── gradio wiring ────────────────────────────────────────────────────────────
def tokenize_text(text: str):
    text = (text or "").strip()
    if not text:
        return gr.update(value=None), None, "Enter some text first.", ""
    ids = tok(text, add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
    pieces = [tok.decode([t]) for t in ids]
    spans = [(p, "t1" if j % 2 == 0 else "t2") for j, p in enumerate(pieces)]
    msg = (f"**{len(ids)} tokens.** Click any token to read its layer-{LAYER} "
           f"activation through the NLA. Later tokens carry more context.")
    return spans, {"ids": ids, "pieces": pieces}, msg, ""


def analyze_at(tokstate: dict | None, idx, mode: str):
    if not tokstate:
        return None, "", "Tokenize some text first."
    ids, pieces = tokstate["ids"], tokstate["pieces"]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None, "", "Click a token (or enter a valid position)."
    if not (0 <= idx < len(ids)):
        return None, "", f"Position must be in [0, {len(ids) - 1}]."
    res = gpu_analyze(ids, idx)
    res["token"] = pieces[idx].strip() or repr(pieces[idx])
    res["pos"] = idx
    if not res["lines"]:
        return None, "", "The AV produced no output for this activation — try another token."
    return res, render_viz(res, mode), ""


def on_token_click(tokstate: dict | None, mode: str, evt: gr.SelectData):
    return analyze_at(tokstate, evt.index, mode)


def analyze_text(text: str, idx, mode: str):
    """Self-contained (text + position) — no State dependency. Powers the
    'Analyze position' button and the public API."""
    text = (text or "").strip()
    if not text:
        return None, "", "Enter some text first."
    ids = tok(text, add_special_tokens=True)["input_ids"][:MAX_TEXT_TOKENS]
    pieces = [tok.decode([t]) for t in ids]
    return analyze_at({"ids": ids, "pieces": pieces}, idx, mode)


def on_mode_change(state: dict | None, mode: str):
    return render_viz(state, mode)


with gr.Blocks(css=CSS, title="NLA v3 explorer") as demo:
    gr.Markdown(
        "# 🔬 NLA v3 — read a language model's mind, one token at a time\n"
        "A **natural-language autoencoder** trained on Qwen2.5-7B layer-20 activations: "
        "an *actor* (AV) verbalizes the activation at the token you click into a "
        "salience-ordered list, and a *critic* (AR) reconstructs the activation from the text. "
        "**FVE** (fraction of variance explained) measures how much of the vector the first *k* "
        "lines recover — truncation-RL trained the actor to front-load what matters. "
        f"[AV/AR checkpoints](https://huggingface.co/{RL_REPO}) · iter 200, KL 0.03, U[1,120]-token truncation."
    )
    with gr.Row():
        with gr.Column(scale=5):
            text_in = gr.Textbox(label="Text", lines=6,
                                 placeholder="Paste any text, then Tokenize…",
                                 value=DEFAULT_TEXTS[0])
            with gr.Row():
                tokenize_btn = gr.Button("Tokenize", variant="primary")
                mode = gr.Radio(["marginal", "cumulative"], value="marginal",
                                label="FVE view", scale=0)
            with gr.Row():
                pos_in = gr.Number(label="…or analyze token position", precision=0,
                                   value=None, scale=2)
                pos_btn = gr.Button("Analyze position", scale=1)
            gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in],
                        label="Try one of these")
        with gr.Column(scale=7):
            status = gr.Markdown("")
            tokens_out = gr.HighlightedText(
                label="Tokens — click one", combine_adjacent=False, show_legend=False,
                elem_classes=["tokwrap"],
                color_map={"t1": "rgba(42,120,214,0.10)", "t2": "rgba(42,120,214,0.22)"},
            )
    viz = gr.HTML("")
    tok_state = gr.State(None)
    res_state = gr.State(None)

    tokenize_btn.click(tokenize_text, [text_in], [tokens_out, tok_state, status, viz],
                       api_name="tokenize")
    text_in.submit(tokenize_text, [text_in], [tokens_out, tok_state, status, viz])
    tokens_out.select(on_token_click, [tok_state, mode], [res_state, viz, status])
    pos_btn.click(analyze_text, [text_in, pos_in, mode], [res_state, viz, status],
                  api_name="analyze")
    mode.change(on_mode_change, [res_state, mode], [viz])

demo.launch()
