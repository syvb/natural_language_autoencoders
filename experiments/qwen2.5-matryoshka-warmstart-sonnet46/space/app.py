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


# ── viz + ui (pure CPU; palette per the validated reference set) ─────────────
CSS = """
/* layout — overflow:visible: gradio's default overflow:hidden breaks position:sticky */
.gradio-container{max-width:1360px !important; margin:0 auto !important;
  overflow:visible !important;}
#nla-header h1{font-size:23px; margin-bottom:0;}
#nla-header p{margin-top:6px;}
#nla-click-idx{display:none !important;}
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


def render_viz(state: dict | None, mode: str) -> str:
    if not state:
        return EMPTY_CARD
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
    pieces = [tok.decode([t]) for t in ids]
    return render_tokens(pieces, len(all_ids)), {"ids": ids, "pieces": pieces}, EMPTY_CARD


def analyze_at(tokstate: dict | None, idx, mode: str):
    if not tokstate:
        return None, _card("Tokenize some text first.")
    ids, pieces = tokstate["ids"], tokstate["pieces"]
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return None, _card("Click a token (or enter a valid position).")
    if not (0 <= idx < len(ids)):
        return None, _card(f"Position must be in [0, {len(ids) - 1}].")
    res = gpu_analyze(ids, idx)
    res["token"] = pieces[idx].strip() or repr(pieces[idx])
    res["pos"] = idx
    if not res["lines"]:
        return None, _card("The AV produced no output for this activation — try another token.")
    return res, render_viz(res, mode)


def on_token_click(tokstate: dict | None, mode: str, idx: str):
    return analyze_at(tokstate, idx, mode)


def analyze_text(text: str, idx, mode: str):
    """Self-contained (text + position) — no State dependency. Powers the
    'Analyze position' button and the public API."""
    tokens_html, tokstate, _ = tokenize_text(text)
    if tokstate is None:
        return tokens_html, None, None, _card("Enter some text first.")
    res, viz_html = analyze_at(tokstate, idx, mode)
    return tokens_html, tokstate, res, viz_html


def on_mode_change(state: dict | None, mode: str):
    return render_viz(state, mode)


with gr.Blocks(css=CSS, js=CLICK_JS, title="NLA v3 explorer") as demo:
    gr.Markdown(
        "# 🔬 NLA v3 — read a language model's mind, one token at a time\n"
        "A **natural-language autoencoder** for Qwen2.5-7B layer-20 activations: click a token "
        "and the *actor* verbalizes its activation into a salience-ordered list of lines, while "
        "the *critic* reconstructs the vector from each line-prefix — the bars show how much of "
        "the vector (**FVE**, fraction of variance explained) the first *k* lines recover. "
        "Truncation-RL trained the actor to front-load what matters. "
        f"[AV/AR checkpoints](https://huggingface.co/{RL_REPO}) · iter 200, KL 0.03, "
        "U[1,120]-token truncation.",
        elem_id="nla-header",
    )
    tok_state = gr.State(None)
    res_state = gr.State(None)
    tokens_out = gr.HTML(render=False)
    viz = gr.HTML(EMPTY_CARD, render=False)

    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            text_in = gr.Textbox(label="Text", lines=5, max_lines=10,
                                 placeholder="Paste any text, then Tokenize…",
                                 value=DEFAULT_TEXTS[0])
            tokenize_btn = gr.Button("Tokenize", variant="primary")
            gr.Examples(examples=[[t] for t in DEFAULT_TEXTS], inputs=[text_in],
                        fn=tokenize_text, outputs=[tokens_out, tok_state, viz],
                        run_on_click=True, label="Or try one of these")
            tokens_out.render()
        with gr.Column(scale=5, elem_classes=["nla-side"]):
            mode = gr.Radio(["marginal", "cumulative"], value="marginal", label="FVE view")
            viz.render()
            with gr.Accordion("Analyze a token position by number", open=False):
                with gr.Row():
                    pos_in = gr.Number(label="token position", precision=0,
                                       value=None, scale=2)
                    pos_btn = gr.Button("Analyze", scale=1)
    # hidden bridge: CLICK_JS writes the clicked token index here (display:none)
    click_idx = gr.Textbox(value="", label="clicked token index", elem_id="nla-click-idx")

    tokenize_btn.click(tokenize_text, [text_in], [tokens_out, tok_state, viz],
                       api_name="tokenize")
    text_in.submit(tokenize_text, [text_in], [tokens_out, tok_state, viz])
    click_idx.input(on_token_click, [tok_state, mode, click_idx], [res_state, viz])
    pos_btn.click(analyze_text, [text_in, pos_in, mode],
                  [tokens_out, tok_state, res_state, viz], api_name="analyze")
    mode.change(on_mode_change, [res_state, mode], [viz])
    demo.load(tokenize_text, [text_in], [tokens_out, tok_state, viz])

demo.launch()
