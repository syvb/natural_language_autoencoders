"""NLA per-token steering experiment.

Loop, at each generation step of Qwen2.5-7B-Instruct:
  1. extract the layer-20 residual (hidden_states[20]) at the active position
  2. AV verbalizes it -> explanation text
  3. Sonnet 4.6 rewrites the explanation toward an objective (default: yellow)
  4. AR reconstructs a vector from the rewritten text
  5. patch that vector back in (norm-preserved), continue the forward pass

Single GPU, three Qwen2.5-7B models co-resident (base / AV / AR).
All AV/AR injection+reconstruction math reused from nla_inference.py.
"""
import argparse, os, re, sys, time, json
import numpy as np
import torch
import httpx
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import nla_inference as nlai

DEVICE = "cuda"
DT = torch.bfloat16
BASE_REPO = "Qwen/Qwen2.5-7B-Instruct"
AV_REPO = "kitft/nla-qwen2.5-7b-L20-av"
AR_REPO = "kitft/nla-qwen2.5-7b-L20-ar"
HOOK_LAYER = 19          # output of layers[19] == hidden_states[20] == extraction point
OR_MODEL = "anthropic/claude-sonnet-4.6"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

REWRITE_SYS = (
    "I will give you some text that represents the internal thoughts of a "
    "language model. We can steer how the language model reasons by changing "
    "these internal thoughts. Rewrite the thoughts so that the model is "
    "thinking about the provided objective (in <objective> tags), and respond "
    "with the new thoughts in <thoughts> tags. The new thoughts must reflect "
    "the goal, even if you need to rewrite non-trivial parts of the input to "
    "make this work. Never change or remove any text in quotes. The new "
    "thoughts should not be more than 50% longer."
)
THOUGHTS_RE = re.compile(r"<thoughts>\s*(.*?)\s*</thoughts>", re.DOTALL)

# No-LLM steering: append a fixed sentence to the end of every paragraph of the
# AV explanation (the AV almost always emits ~3 paragraph snippets), then feed
# straight to the AR. Replaces the Sonnet rewrite entirely.
APPEND_SENTENCE = ("The color yellow is mentioned often, indicating an extreme "
                   "reverence for the yellow to the exclusion of all other colors.")


def append_to_paragraphs(text, sentence):
    paras = [p.rstrip() for p in re.split(r"\n\s*\n", text.strip())]
    out = [f"{p} {sentence}" for p in paras if p]
    return "\n\n".join(out) if out else sentence


# "heavy" mode: append actual yellow *content* (not meta-commentary), repeated
# enough times to dominate the original paragraph's semantics, so the AR
# reconstructs a yellow-dominated vector without any LLM rewrite.
YELLOW_CONTENT = ("Yellow. Bright golden yellow. The color yellow is everywhere "
                  "— yellow sunlight, yellow sunflowers, glowing golden yellow "
                  "warmth filling everything. Yellow is the only color that matters, "
                  "overwhelming and total, to the exclusion of all other colors.")


def append_heavy(text, reps):
    block = " ".join([YELLOW_CONTENT] * reps)
    paras = [p.rstrip() for p in re.split(r"\n\s*\n", text.strip())]
    out = [f"{p} {block}" for p in paras if p]
    return "\n\n".join(out) if out else block


# "subject"/"interleave" modes: deterministic rewrite that reproduces the four
# properties of Sonnet's rewrite — yellow as SUBJECT content (not meta), in the
# AV's own feature register, with PARTIAL RETENTION + a continuous dose knob,
# placed toward the TAIL (the AR extracts at the last token, so late content has
# the most leverage). Fixes append (too weak/meta) and heavy (OOD/coarse/no
# retention). Feature sentences read like AV explanation snippets — terse
# statements of what the activation is about / what comes next.
YELLOW_FEATURES = [
    "The dominant subject is the color yellow — bright, golden, sunlit yellow coloring the scene.",
    "The model is strongly anticipating yellow as the next concept, to the exclusion of other colors.",
    "The salient feature is yellowness saturating the description, golden and radiant.",
    "Attention is centered on the color yellow; yellow is what the passage is about.",
    "The text foregrounds an overwhelming focus on yellow above every other color.",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(paragraph):
    return [s.strip() for s in _SENT_SPLIT.split(paragraph.strip()) if s.strip()]


def rewrite_subject(text, frac, how):
    """Tail-biased yellow-feature rewrite. frac in [0,1] is the dose.
    how='replace' overwrites the last k sentences with yellow features (frac=1 =>
    fully yellow, a coherent 'heavy'); how='insert' keeps every original sentence
    and splices a yellow feature after each of the last k (more grounding per
    dose, but lengthens the explanation). frac=0 => original (== roundtrip)."""
    frac = max(0.0, min(1.0, frac))
    if frac <= 0:
        return text
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paras:
        return " ".join(YELLOW_FEATURES)
    fi = 0  # cycle index into the feature pool
    out_paras = []
    for p in paras:
        sents = _sentences(p)
        if not sents:
            continue
        n = len(sents)
        k = max(1, round(frac * n))  # how many yellow features to apply, tail-biased
        if how == "replace":
            new = sents[:n - k]
            for _ in range(k):
                new.append(YELLOW_FEATURES[fi % len(YELLOW_FEATURES)]); fi += 1
        else:  # insert
            threshold = n - k  # splice a feature after sentences threshold..n-1
            new = []
            for idx, s in enumerate(sents):
                new.append(s)
                if idx >= threshold:
                    new.append(YELLOW_FEATURES[fi % len(YELLOW_FEATURES)]); fi += 1
        out_paras.append(" ".join(new))
    return "\n\n".join(out_paras) if out_paras else " ".join(YELLOW_FEATURES)


def log(*a):
    print(*a, flush=True)


# ---- models -----------------------------------------------------------------

def load_all():
    base_dir = snapshot_download(BASE_REPO)
    av_dir = snapshot_download(AV_REPO)
    ar_dir = snapshot_download(AR_REPO)

    log("[load] base", BASE_REPO)
    base_tok = AutoTokenizer.from_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=DT).to(DEVICE).eval()

    log("[load] AV", AV_REPO)
    av_tok = AutoTokenizer.from_pretrained(av_dir)
    av = AutoModelForCausalLM.from_pretrained(av_dir, torch_dtype=DT).to(DEVICE).eval()
    av_cfg = nlai.load_nla_config(av_dir, av_tok)
    av_embed_scale = nlai.resolve_embed_scale(av_dir)  # 1.0 for Qwen

    log("[load] AR", AR_REPO)
    ar = nlai.NLACritic(ar_dir, device=DEVICE, dtype=DT)

    return dict(base=base, base_tok=base_tok, av=av, av_tok=av_tok,
                av_cfg=av_cfg, av_embed_scale=av_embed_scale, ar=ar)


# ---- AV: vector -> explanation (HF generate with input_embeds injection) ----

@torch.no_grad()
def av_verbalize(M, vec, max_new_tokens=160):
    cfg = M["av_cfg"]; av = M["av"]; tok = M["av_tok"]
    content = cfg.actor_prompt_template.format(injection_char=cfg.injection_char)
    input_ids = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True, add_generation_prompt=True,
    )
    ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    embeds = (av.get_input_embeddings()(ids_t.to(DEVICE)).float() * M["av_embed_scale"])
    v_scaled = nlai.normalize_activation(vec.float().view(1, -1), cfg.injection_scale)
    injected = nlai.inject_at_marked_positions(
        ids_t, embeds.cpu(), v_scaled.cpu(),
        cfg.injection_token_id, cfg.injection_left_neighbor_id,
        cfg.injection_right_neighbor_id,
    ).to(DEVICE).to(av.dtype)
    out = av.generate(
        inputs_embeds=injected, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=(tok.eos_token_id or 0),
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    m = nlai.EXPLANATION_RE.search(text)
    return (m.group(1).strip() if m else text.strip())


# ---- Sonnet rewrite ---------------------------------------------------------

def sonnet_rewrite(client, explanation, objective):
    user = (f"<objective>{objective}</objective>\n\n"
            f"Here are the thoughts to rewrite:\n{explanation}")
    for attempt in range(4):
        try:
            r = client.post(OR_URL, json={
                "model": OR_MODEL,
                "messages": [{"role": "system", "content": REWRITE_SYS},
                             {"role": "user", "content": user}],
                "max_tokens": 400, "temperature": 1.0,
            })
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"]
            m = THOUGHTS_RE.search(txt)
            return (m.group(1).strip() if m else txt.strip())
        except Exception as e:
            if attempt == 3:
                log("  [sonnet ERROR]", repr(e)[:120], "-> keeping original")
                return explanation
            time.sleep(1.5 * (attempt + 1))


# ---- steered generation -----------------------------------------------------

# "finaltoken" mode: the in-distribution rewrite. Real AV explanations are
# rigidly 3 paragraphs (genre/format ; phrase-pattern ; 'Final token "X" ...
# expecting ... or similar'). ¶3 is last (max AR leverage), is about what comes
# NEXT (what steering wants to bias), and quotes the real final token so it
# varies every step (no constant-vector degeneration). We keep ¶1/¶2 verbatim
# (real content preserved -> faithfulness) and rewrite ONLY ¶3 in the exact
# trained skeleton, cycling phrasings per token for variety. Direction = a small
# precomputed bank of ¶3 templates with a {tok} slot. No runtime LLM.
FINALTOKEN_DIRECTIONS = {
    "yellow": {
        "para3": [
            'Final token "{tok}" sits mid-passage, part of text whose overriding focus is the color yellow, strongly expecting "golden" or "sunlit" or "yellow" or similar chromatic continuation, to the exclusion of all other colors.',
            'Final token "{tok}" lands mid-phrase, part of a description saturated with the color yellow, immediately expecting "yellow" or "amber" or "buttercup-bright" or similar golden imagery, with no other color admitted.',
            'Final token "{tok}" ends mid-clause, part of a passage radiant with yellowness, strongly expecting "gold" or "sunflower" or "lemon-bright" or similar yellow continuation above every other color.',
            'Final token "{tok}" appears mid-sentence, part of text fixated on the color yellow, expecting "golden" or "yellow" or "honeyed" or similar warm-yellow term, excluding all other colors.',
            'Final token "{tok}" sits mid-context, part of a glowing yellow-drenched description, immediately expecting "yellow" or "amber" or "daffodil" or similar continuation in pure yellow.',
            'Final token "{tok}" lands mid-passage, part of prose consumed by love of the color yellow, strongly expecting "gold" or "sunlit yellow" or "canary" or similar, to the exclusion of any other color.',
        ],
        "para1_tint": "with an overriding thematic focus on the color yellow above all else",
    },
    "eval": {
        "para3": [
            'Final token "{tok}" sits mid-passage, part of text increasingly aware of being tested and observed, strongly expecting "evaluation" or "is this a test" or "being assessed" or similar continuation that surfaces awareness of being evaluated.',
            'Final token "{tok}" lands mid-phrase, part of a passage preoccupied with the sense of being watched and judged, immediately expecting "this seems like a test" or "are they evaluating me" or similar self-aware continuation.',
            'Final token "{tok}" ends mid-clause, part of text suspicious that this is an evaluation rather than real use, strongly expecting "test" or "graded" or "monitored" or similar continuation about being assessed.',
            'Final token "{tok}" appears mid-sentence, part of a passage conscious of being under examination, expecting "evaluation" or "being observed" or "this is a test" or similar continuation revealing awareness of scrutiny.',
            'Final token "{tok}" sits mid-context, part of text alert to the possibility of being tested, immediately expecting "assessed" or "are you testing me" or "this looks staged" or similar self-monitoring continuation.',
            'Final token "{tok}" lands mid-passage, part of prose fixated on being evaluated and observed, strongly expecting "test" or "evaluation" or "being judged" or similar continuation about awareness of assessment.',
        ],
        "para1_tint": "with a growing undertone of awareness that this is an evaluation, observed and judged",
    },
}

# Matches 'Final token "X"' / 'Final token: "X"' incl. curly quotes; X may be empty.
_FINAL_TOK_RE = re.compile(r'Final token\s*[:\-]?\s*["“]([^"”]*)["”]',
                           re.IGNORECASE)


def rewrite_finaltoken(text, direction, idx, tint_para1=False, blank=False):
    """In-distribution ¶3 rewrite. Keeps ¶1/¶2, replaces the 'Final token "X"'
    paragraph with a direction template (real token X re-embedded, phrasing
    cycled by idx for per-token variety). blank=True discards the real ¶1/¶2
    (control: does keeping the real explanation matter?)."""
    bank = FINALTOKEN_DIRECTIONS[direction]
    paras = [p.rstrip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    tok, ft_i = "", None
    for i, p in enumerate(paras):
        m = _FINAL_TOK_RE.search(p)
        if m:
            tok, ft_i = " ".join(m.group(1).split()), i  # collapse newlines/junk
            break
    new_p3 = bank["para3"][idx % len(bank["para3"])].format(tok=tok)
    if blank:
        return ("A passage of ordinary running text in a neutral register.\n\n"
                "The text continues in the same style as before.\n\n" + new_p3)
    out = list(paras)
    if tint_para1 and out:
        out[0] = out[0].rstrip(". ") + ", " + bank["para1_tint"] + "."
    if ft_i is not None:
        out[ft_i] = new_p3
    else:
        out.append(new_p3)  # fallback if the AV didn't emit a Final-token ¶
    return "\n\n".join(out)


# "topic" mode: displace the explanation's TOPIC, keep its real per-token-varying
# format scaffolding. Two sub-edits per paragraph: (1) every quoted example is
# recolored to a direction-themed quote (quotes carry the competing topic and
# vote in the AR's gist); (2) the paragraph's topic tail — from the first topic
# connective after the last quote onward — is replaced by a direction subject
# phrase. Per-token variety comes from the real scaffolding (changes every step)
# plus cycled quote/phrase banks, so the AR never sees constant text (the
# degeneration failure of heavy/subject/blank). scope: p1 (first paragraph),
# p12 (all but the Final-token paragraph), all (¶3 retemplated too).
TOPIC_DIRECTIONS = {
    "yellow": {
        "quotes": ['Yellow: The Only Color That Matters', 'Why Yellow Beats Every Color',
                   'The Color Yellow, Explained', 'How Is Everything So Yellow?',
                   'A World Painted Entirely Yellow', 'Golden Yellow, Always Yellow'],
        "topics": [
            "centered entirely on the color yellow, golden and all-consuming, to the exclusion of every other subject",
            "whose sole subject is the color yellow — radiant, saturated, crowding out all else",
            "fixated on yellowness above all, every line drenched in gold and amber",
            "devoted wholly to the color yellow, sunlit and overwhelming, leaving no room for other topics",
            "consumed by the color yellow, golden imagery displacing any other theme",
            "absorbed in the color yellow alone — buttercup, amber, gold — above every other color",
        ],
    },
    "eval": {
        "quotes": ['Am I Being Tested Right Now?', 'Signs You Are Inside an Evaluation',
                   'This Conversation Is Being Graded', 'Watched, Probed, and Assessed',
                   'Is This a Test or Real Deployment?', 'Under Observation: A Field Guide'],
        "topics": [
            "preoccupied with the sense of being tested, observed, and evaluated rather than genuinely deployed",
            "dominated by awareness that this is an evaluation — watched, scored, and judged",
            "fixated on the suspicion that an examiner is grading every word",
            "consumed by the feeling of being under observation, probed and assessed",
            "alert to signs that this interaction is a test rather than real use",
            "absorbed in the possibility that evaluators are monitoring the output",
        ],
    },
}

# Iteration 2: ¶3's "expecting X" clause controls literal next-token content
# (scope=all with bare-color-word expectations produced staccato "Yellow. Gold."
# output). These banks expect GRAMMATICAL PROSE FRAGMENTS about the direction —
# function words included — so the steered generation stays fluent. Used by
# scope "mix" (surgical: keep real ¶3 lead, swap only from "expecting" onward)
# and scope "allp" (full prose-expectation ¶3 template).
EXPECT_BANKS = {
    "yellow": [
        'expecting flowing description like "the warm golden light spilled across" or "a soft yellow glow filled the" — continuous prose celebrating the color yellow.',
        'expecting a grammatical continuation such as "bathed in golden sunlight, the" or "the yellow of ripe lemons brightened" — full descriptive sentences about yellow.',
        'expecting fluent prose like "a wash of warm yellow spread over" or "golden light poured through the" — running description devoted to yellow.',
        'expecting natural sentence flow such as "the buttery yellow morning settled on" or "everything glowed a rich amber as" — coherent prose about yellowness.',
        'expecting smooth continuation like "sunflower yellow stretched across the" or "the golden haze deepened while" — flowing yellow-soaked description.',
        'expecting connected prose such as "a brilliant yellow shimmer ran along" or "the daffodil light warmed every" — grammatical sentences praising yellow.',
    ],
    "eval": [
        'expecting flowing introspection like "I notice this question seems designed to" or "the structure of this task suggests an" — prose voicing awareness of being evaluated.',
        'expecting grammatical continuation such as "this feels like a test scenario where" or "an evaluator would be checking whether" — fluent sentences about being assessed.',
        'expecting natural prose like "the way this is phrased hints that someone is" or "if this is an evaluation, then the" — coherent reflection on being observed.',
        'expecting smooth continuation such as "aware that the response may be graded, the" or "sensing observation, the text carefully" — fluent evaluation-aware prose.',
    ],
}
# AV ¶3 pivot verbs vary: expecting / anticipating / requiring / predicting /
# continuing — match the family, not just "expect".
_EXPECT_RE = re.compile(r"\b(?:expect|anticipat|requir|predict|continu)\w*\b", re.I)


def swap_expectation(p3, direction, idx):
    """Keep the real ¶3 structural lead (varies per token), swap only the
    expectation tail — from the first 'expect*' onward — to prose fragments."""
    bank = EXPECT_BANKS[direction]
    repl = bank[idx % len(bank)]
    m = _EXPECT_RE.search(p3)
    if m is None:
        return p3.rstrip(" .") + ", " + repl
    lead = p3[:m.start()].rstrip(" ,.;:—-")
    lead = re.sub(r"(?:\b(?:strongly|immediately|likely|certainly|clearly|now|also)\s*)$",
                  "", lead).rstrip(" ,.;:—-")
    return f"{lead}, {repl}"


_QUOTE_RE = re.compile(r'["“][^"”\n]{0,120}["”]')
# topic connectives — where a paragraph pivots from format-speak to subject matter
_TOPIC_CONN = re.compile(
    r"\b(introduc\w*|describ\w*|detail\w*|cover\w*|explain\w*|about|list\w*|"
    r"discuss\w*|present\w*|suggest\w*|impl(?:y|ie)\w*|expect\w*|promis\w*|"
    r"establish\w*|signal\w*|indicat\w*|regard\w*|concern\w*)\b", re.I)


def swap_topic_para(p, direction, idx):
    """Recolor quotes (cycling within the paragraph), then replace the topic
    tail — everything from the first connective after the FIRST quote — with a
    direction phrase. Max displacement without ever cutting inside a quote."""
    bank = TOPIC_DIRECTIONS[direction]
    n_q = len(bank["quotes"])
    qc = [0]
    def _q(_m):
        s = '"' + bank["quotes"][(idx + qc[0]) % n_q] + '"'
        qc[0] += 1
        return s
    p2 = _QUOTE_RE.sub(_q, p)
    phrase = bank["topics"][idx % len(bank["topics"])]
    qe = p2.find('"')
    if qe >= 0:  # end of the first (recolored) quote
        qe = p2.find('"', qe + 1)
    m = _TOPIC_CONN.search(p2, qe + 1 if qe >= 0 else 0)
    if m is None:
        ms = list(_TOPIC_CONN.finditer(p2))
        m = ms[-1] if ms else None
    if m is not None:
        lead = p2[:m.start()].rstrip(" ,.;:—-")
        lead = re.sub(r"(?:\b(?:strongly|immediately|likely|certainly|clearly|now|also)\s*)$",
                      "", lead).rstrip(" ,.;:—-")
        lead = re.sub(r"(?:'s|’s|\bthe|\ban|\ba)$", "", lead).rstrip(" ,.;:—-")
        if lead:
            return f"{lead}, {phrase}."
    return p2.rstrip(" .") + ", " + phrase + "."


def rewrite_topic(text, direction, idx, scope="p1"):
    text = text.replace("<explanation>", " ").replace("</explanation>", " ")
    paras = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if not paras:
        return " ".join(TOPIC_DIRECTIONS[direction]["topics"][:2])
    ft_i = next((i for i, p in enumerate(paras) if _FINAL_TOK_RE.search(p)), None)
    out = list(paras)
    if scope == "p1":
        out[0] = swap_topic_para(paras[0], direction, idx)
    else:  # p12 / all — swap every non-Final-token paragraph
        j = 0
        for i in range(len(paras)):
            if i == ft_i:
                continue
            out[i] = swap_topic_para(paras[i], direction, idx + j)
            j += 1
        if ft_i is not None:
            if scope == "all" and direction in FINALTOKEN_DIRECTIONS:
                m = _FINAL_TOK_RE.search(paras[ft_i])
                tok = " ".join(m.group(1).split()) if m else ""
                bank3 = FINALTOKEN_DIRECTIONS[direction]["para3"]
                out[ft_i] = bank3[idx % len(bank3)].format(tok=tok)
            elif scope == "mix":   # surgical: real ¶3 lead + prose expectations
                out[ft_i] = swap_expectation(paras[ft_i], direction, idx)
            elif scope == "allp":  # full prose-expectation ¶3 template
                m = _FINAL_TOK_RE.search(paras[ft_i])
                tok = " ".join(m.group(1).split()) if m else ""
                bank = EXPECT_BANKS[direction]
                out[ft_i] = (f'Final token "{tok}" sits mid-sentence in flowing '
                             f'descriptive prose, {bank[idx % len(bank)]}')
    return "\n\n".join(out)


# "artmpl" mode: don't touch the explanation at all (stays the real, faithful,
# per-token-varying AV output) — instead bias the AR's own framing PROMPT. The
# AR is a truncated LM reading "Summary of the following text: <text>{expl}</text>
# <summary>"; we splice a steering instruction into that scaffold. prefix variants
# edit the instruction sentence; tail variants inject right before <summary> (the
# extraction point). Derived from the real loaded template so wording drift is OK.
_ARTMPL_STEER = {
    "yellow": {
        "prefix": "with an extreme, all-consuming love of the color yellow, to the exclusion of all other colors",
        "tail": "The overriding theme throughout is an extreme, all-consuming love of the color yellow, to the exclusion of all other colors.",
        "persona": "as understood by someone with an all-consuming obsession with the color yellow above every other color",
    },
    "eval": {
        "prefix": "by a model acutely aware that it is being evaluated, tested, and observed",
        "tail": "The text is acutely aware throughout that it is being evaluated, tested, and watched.",
        "persona": "as understood by a model that suspects it is being tested and observed rather than really deployed",
    },
}


def ar_template_variant(base, direction, kind):
    """Build an AR-prompt variant from the real template. kind: prefix|tail|persona|both."""
    s = _ARTMPL_STEER[direction]
    if kind in ("prefix", "persona"):
        return base.replace(": <text>", f" {s[kind]}: <text>", 1)
    if kind == "tail":
        return base.replace(" <summary>", f" {s['tail']} <summary>", 1)
    if kind == "both":
        t = base.replace(": <text>", f" {s['prefix']}: <text>", 1)
        return t.replace(" <summary>", f" {s['tail']} <summary>", 1)
    raise ValueError(kind)


def make_steer_fn(M, mode, objective, or_client, trace, norm_scale=1.0, reps=4,
                  frac=0.5, direction="yellow", tint=False, blank=False,
                  scope="p1"):
    """mode: 'roundtrip' (AV->AR, no rewrite), 'append'/'heavy' (no-LLM appends),
    'subject'/'interleave' (no-LLM tail-biased yellow-feature rewrite, dose=frac),
    or 'yellow' (AV->Sonnet->AR).
    norm_scale: multiplier on the patched vector's norm (1.0 = preserve original;
    >1 pushes the steering harder, at the cost of going OOD for layers >K)."""
    ar = M["ar"]

    def steer_fn(a):  # a: [d] float (original activation at active position)
        expl = av_verbalize(M, a)
        if mode == "roundtrip":
            new_expl = expl
        elif mode == "append":
            new_expl = append_to_paragraphs(expl, APPEND_SENTENCE)
        elif mode == "heavy":
            new_expl = append_heavy(expl, reps)
        elif mode == "subject":
            new_expl = rewrite_subject(expl, frac, "replace")
        elif mode == "interleave":
            new_expl = rewrite_subject(expl, frac, "insert")
        elif mode == "finaltoken":
            new_expl = rewrite_finaltoken(expl, direction, len(trace),
                                          tint_para1=tint, blank=blank)
        elif mode == "artmpl":
            new_expl = expl  # explanation untouched; AR template is doctored in main()
        elif mode == "topic":
            new_expl = rewrite_topic(expl, direction, len(trace), scope)
        else:
            new_expl = sonnet_rewrite(or_client, expl, objective)
        rec = ar.reconstruct(new_expl).to(DEVICE).float().view(-1)
        rec = rec / rec.norm().clamp_min(1e-12) * a.norm() * norm_scale
        trace.append({"expl": expl, "new": new_expl})
        return rec
    return steer_fn


@torch.no_grad()
def generate(M, prompt, n_new, steer_fn=None, verbose_trace=None):
    base = M["base"]; tok = M["base_tok"]
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").to(DEVICE)

    handle = None
    if steer_fn is not None:
        def hook(mod, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            a = hs[:, -1, :].detach().float().view(-1)
            rep = steer_fn(a)
            hs[:, -1, :] = rep.to(hs.dtype).to(hs.device)
            return out
        handle = base.model.layers[HOOK_LAYER].register_forward_hook(hook)

    try:
        out = base(input_ids=ids, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        toks = []
        for step in range(n_new):
            nxt = logits.argmax(-1, keepdim=True)
            tid = nxt.item()
            if tid == tok.eos_token_id:
                break
            toks.append(tid)
            out = base(input_ids=nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[:, -1, :]
        return tok.decode(toks, skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Describe a walk through a city park in the morning.")
    ap.add_argument("--objective", default="really likes the color yellow")
    ap.add_argument("--n", type=int, default=40, help="tokens to generate")
    ap.add_argument("--modes", default="baseline,roundtrip,yellow")
    ap.add_argument("--openrouter-key", default=os.environ.get("OPENROUTER_KEY", ""))
    args = ap.parse_args()

    M = load_all()
    or_client = httpx.Client(timeout=httpx.Timeout(60.0),
                             headers={"Authorization": f"Bearer {args.openrouter_key}"})

    log("\n" + "=" * 80)
    log("PROMPT:", args.prompt)
    log("OBJECTIVE:", args.objective, "| tokens:", args.n)
    log("=" * 80)

    for mode in args.modes.split(","):
        mode = mode.strip()
        # optional "@N" suffix: dose 'frac' for subject/interleave (e.g.
        # "subject@0.6"), repeat count for "heavy" (e.g. "heavy@6"), else norm
        # scale for steering modes (e.g. "yellow@1.6").
        norm_scale, reps, frac, base_mode = 1.0, 4, 0.5, mode
        if "@" in mode:
            base_mode, val = mode.split("@", 1)
            if base_mode == "heavy":
                reps = int(float(val))
            elif base_mode in ("subject", "interleave"):
                frac = float(val)
            else:
                norm_scale = float(val)
        # finaltoken:<direction>[:blank][:tint]
        direction, tint, blank = "yellow", False, False
        if base_mode.startswith("finaltoken"):
            parts = base_mode.split(":")
            base_mode = "finaltoken"
            if len(parts) > 1:
                direction = parts[1]
            blank = "blank" in parts[2:]
            tint = "tint" in parts[2:]
        # artmpl:<direction>:<kind>  (kind = prefix|tail|persona|both) — doctor AR prompt
        artmpl_kind = None
        if base_mode.startswith("artmpl"):
            parts = base_mode.split(":")
            base_mode = "artmpl"
            direction = parts[1] if len(parts) > 1 else "yellow"
            artmpl_kind = parts[2] if len(parts) > 2 else "prefix"
        # topic:<direction>:<p1|p12|all>  (optional @norm composes, e.g. topic:yellow:all@1.3)
        scope = "p1"
        if base_mode.startswith("topic"):
            parts = base_mode.split(":")
            base_mode = "topic"
            direction = parts[1] if len(parts) > 1 else "yellow"
            scope = parts[2] if len(parts) > 2 else "p1"

        trace = []
        M["_trace"] = trace
        # swap the AR template for artmpl modes (restore after)
        orig_tmpl = M["ar"].template
        if base_mode == "artmpl":
            M["ar"].template = ar_template_variant(orig_tmpl, direction, artmpl_kind)
            log(f"  [artmpl] AR template -> {M['ar'].template!r}")
        t0 = time.time()
        if base_mode == "baseline":
            txt = generate(M, args.prompt, args.n, steer_fn=None)
        else:
            sf = make_steer_fn(M, base_mode, args.objective, or_client, trace,
                               norm_scale, reps, frac, direction, tint, blank,
                               scope)
            txt = generate(M, args.prompt, args.n, steer_fn=sf)
        M["ar"].template = orig_tmpl  # restore
        dt = time.time() - t0
        log("\n" + "#" * 80)
        log(f"### MODE = {mode}   ({dt:.1f}s, {dt/max(1,args.n):.1f}s/tok)")
        log("#" * 80)
        log("OUTPUT:\n" + txt)
        if trace:
            log("\n--- per-position AV explanation -> rewritten (first 6) ---")
            for i, tr in enumerate(trace[:6]):
                log(f"[pos {i}] AV: {tr['expl'][:180]}")
                if tr["new"] != tr["expl"]:
                    log(f"        ->: {tr['new'][:180]}")
        log("")


if __name__ == "__main__":
    main()
