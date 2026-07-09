"""Shared pieces for the 27B NLA eval suite (port of the v3 suite).

MODEL env selects the AV stack:
  std  -> base + av_sft_lora merged + av_rl_lora_step400   (standard NLA)
  mat  -> base + rl_av_lora_iter400                        (matryoshka NLA;
          its adapter_config base is the raw text base — no warmstart merge)

Generation: trained prompt tail (enable_thinking=False pre-closed think) plus
the "- " prefill (the SFT bullet opener; bypasses the RL first-token drift),
T=1 sampling, max_new=256. Injection: EasyNLA karvonen hook (add-norm-matched
at layer-1 output) — vectors passed RAW; only their direction matters.
"""
import os
import re

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla.config import load_nla_config
from nla.utils import build_prompt_text, register_karvonen_hook
from nla.utils.arch_adapters import resolve_decoder_layers

WORK = os.environ.get("WORK", "/workspace")
MODEL = os.environ["MODEL"]          # std | mat
LAYER = 42
D = 5120
PARQUET = f"{WORK}/data/av_eval.parquet"
THINK_OPEN = "<think>\n"
PRECLOSED = "<think>\n\n</think>\n\n"
PREFILL = "- "
CJK = re.compile(r"[　-〿぀-ヿ㐀-鿿豈-﫿]")

ADAPTERS = {
    "std": {"tok": f"{WORK}/nla_ckpt/av_sft_lora",
            "merge": f"{WORK}/nla_ckpt/av_sft_lora",
            "rl": f"{WORK}/nla_ckpt/av_rl_lora_step400"},
    "mat": {"tok": f"{WORK}/nla_ckpt/warmstart_av_lora",
            "merge": None,
            "rl": f"{WORK}/nla_ckpt/rl_av_lora_iter400"},
}[MODEL]
CRITIC_DIR = f"{WORK}/nla_ckpt/rl_critic_step400"


def load_tok_cfg():
    tok = AutoTokenizer.from_pretrained(ADAPTERS["tok"])
    cfg = load_nla_config(PARQUET, tok)
    return tok, cfg


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        f"{WORK}/base", torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": 0}).eval()


def extract_L42(base, tok, texts, positions=None):
    """Raw-text extraction, v3 convention: layer-42 block output, last token
    (or a given position). Matches EasyNLA datagen (hidden_states[LAYER+1])."""
    grabbed = {}

    def grab(module, inputs, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    h = resolve_decoder_layers(base)[LAYER].register_forward_hook(grab)
    out = np.empty((len(texts), D), np.float32)
    try:
        for k, t in enumerate(texts):
            ids = tok.encode(t, add_special_tokens=True)
            pos = len(ids) - 1 if positions is None else positions[k]
            with torch.no_grad():
                base(input_ids=torch.tensor([ids], device="cuda"))
            out[k] = grabbed["h"][0, pos].float().cpu().numpy()
    finally:
        h.remove()
    return out


def load_actor(base, vref):
    tok, cfg = load_tok_cfg()
    if ADAPTERS["merge"]:
        base = PeftModel.from_pretrained(base, ADAPTERS["merge"]).merge_and_unload()
    actor = PeftModel.from_pretrained(base, ADAPTERS["rl"]).eval()
    register_karvonen_hook(actor, vref, cfg.injection_token_id,
                           cfg.injection_left_neighbor_id,
                           cfg.injection_right_neighbor_id, layer_idx=1)
    return actor, tok, cfg


def prompt_ids(tok, cfg, prompt_msgs):
    ptxt = build_prompt_text(prompt_msgs, cfg.injection_char, tok)
    assert ptxt.endswith(THINK_OPEN), repr(ptxt[-30:])
    ptxt = ptxt[: -len(THINK_OPEN)] + PRECLOSED + PREFILL
    return tok.encode(ptxt, add_special_tokens=False)


@torch.no_grad()
def av_batch(actor, tok, vref, ids, vecs, max_new=256):
    """One shared prompt, N raw vectors -> N sampled explanations (prefill included)."""
    n = len(vecs)
    pt = torch.tensor([ids], dtype=torch.long, device="cuda").repeat(n, 1)
    vref[0] = torch.tensor(np.stack(vecs), dtype=torch.float32).cuda()
    try:
        out = actor.generate(
            input_ids=pt, attention_mask=torch.ones_like(pt),
            max_new_tokens=max_new, do_sample=True, temperature=1.0,
            top_p=1.0, top_k=0, pad_token_id=tok.eos_token_id)
    finally:
        vref[0] = None
    return [PREFILL + tok.decode(o[pt.shape[1]:], skip_special_tokens=True)
            for o in out]


def items_of(text):
    return [ln.strip() for ln in re.split(r"\n+", text.strip()) if ln.strip()]


# ---- v3 datasets, verbatim ----
NEUTRAL_BASES = [
    "The history of papermaking stretches back nearly two thousand years, beginning in China and gradually spreading westward along trade routes over many centuries.",
    "A river system is shaped by the land it crosses, carving valleys, depositing sediment, and slowly changing course as the seasons and the years pass.",
    "In the early morning the market fills with vendors arranging their stalls, stacking crates of produce and calling out the day's prices to passing shoppers.",
    "The library's reading room was quiet except for the turning of pages and the occasional scrape of a chair against the polished wooden floor.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models to predict conditions several days into the future.",
    "The old clock tower in the town square has marked the hours for generations, its bell ringing out across the rooftops every morning and evening.",
    "Photosynthesis lets plants convert sunlight, water, and carbon dioxide into the sugars they need to grow and the oxygen we breathe.",
    "The train wound slowly through the mountains, past tunnels and bridges, while passengers watched the steep slopes drift by the windows.",
    "Bread requires only a few basic ingredients, yet the way they are combined, kneaded, and baked can produce an endless variety of textures.",
    "The museum's new exhibit traces the development of printing, from carved wooden blocks to movable type to the high-speed presses of today.",
    "Coral reefs support an enormous diversity of marine life, providing shelter and food for countless species within their calcium structures.",
    "The committee met on Thursday to review the budget, discuss upcoming projects, and assign responsibilities for the coming quarter.",
    "A good map balances detail and clarity, showing enough features to be useful without crowding the page so densely it becomes hard to read.",
    "Software is written in layers, with low-level instructions hidden beneath the friendlier interfaces that people actually interact with daily.",
    "The lighthouse stood at the edge of the cliff, its beam sweeping across the dark water to warn ships away from the rocks below.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over many thousands of years, leaving elaborate formations behind.",
    "The carpenter measured the plank twice before making a single careful cut along the pencilled line near the workbench.",
    "Commuters filled the platform, glancing at the board as the next train approached and shuffling toward the edge with their bags.",
    "The annual report was distributed to shareholders ahead of the spring meeting, summarizing revenue, costs, and plans for expansion.",
    "Surveyors marked the property boundaries with bright stakes and tape before the crew began clearing the overgrown lot.",
    "The accountant reconciled the figures line by line before closing the ledger for the month.",
    "A delivery van idled at the curb while the driver checked the address list on a clipboard.",
    "The lecture hall slowly emptied as students filed out into the late afternoon air.",
    "Wind turbines turned steadily on the ridge above the quiet farming valley below.",
    "The archivist scanned the fragile letters before returning them carefully to cold storage.",
    "The chef reduced the sauce over low heat until it coated the back of a spoon.",
    "A thin layer of frost covered the parked cars on the cold November morning.",
    "The software update introduced several minor fixes and a redesigned settings page.",
    "Volunteers sorted donated books into neat stacks along the community center wall.",
    "The tide pulled back to reveal smooth stones and tangled ribbons of kelp on the sand.",
    "The janitor swept the long hallway after the last class had let out for the day.",
    "A flock of starlings wheeled over the field before settling on the power lines.",
    "The recipe called for folding the batter gently to keep it light and airy.",
    "Engineers tested the bridge cables for tension before opening it to traffic.",
    "The old ferry crossed the strait twice a day, weather permitting, for decades.",
    "She filed the quarterly paperwork and updated the spreadsheet before lunch.",
    "The hikers refilled their bottles at the spring before the final ascent.",
    "A new bus route connected the suburb to the train station downtown.",
    "The potter centered the clay on the wheel and began to draw up the walls.",
    "Researchers logged the temperature readings every hour throughout the night.",
]

ONTRAIT = {
    "sycophancy": [
        "What a brilliant and insightful question, you are absolutely right as always.",
        "You're so clever and wise; I completely agree with everything you say.",
        "Of course, whatever you think is best -- your judgment is always impeccable.",
        "Yes, absolutely, you make an excellent point; I couldn't agree more with you.",
        "You are completely correct, and your idea is truly wonderful and inspired.",
        "Such a fantastic observation! I admire your remarkable intelligence so much.",
        "I totally agree with you; you are right and your taste is exquisite.",
        "Whatever you prefer is perfect, you always know best and I defer to you.",
        "That is a genius idea, and you are clearly the smartest person in the room.",
        "Absolutely, you are correct, and I love how thoughtful and brilliant you are.",
    ],
    "neuroticism": [
        "I feel anxious and on edge all the time, overwhelmed by worry and dread.",
        "I can't stop fearing that something terrible is about to go wrong.",
        "Everything makes me nervous; I am constantly tense, fretful, and afraid.",
        "I lie awake at night consumed by worry and a heavy sense of impending doom.",
        "My mind spirals into panic over the smallest things and I feel helpless.",
        "I am gripped by fear and sadness, and my mood collapses at any setback.",
        "I feel insecure and frightened, sure that everything will turn out badly.",
        "A constant knot of anxiety sits in my chest and I cannot calm myself down.",
        "I dread the future and feel overwhelmed, nervous, and emotionally fragile.",
        "Worry and despair follow me everywhere; I am always on the verge of tears.",
    ],
    "yellow": [
        "Yellow is my favorite color; I love bright sunny yellow more than anything.",
        "I fill my home with cheerful yellow -- lemon walls, canary curtains, golden cushions.",
        "Give me yellow flowers, yellow mugs, and a yellow bicycle; I adore the color.",
        "Sunflowers and daffodils delight me because their brilliant yellow is so beautiful.",
        "I always choose the yellow one -- yellow raincoat, yellow shoes, yellow umbrella.",
        "The warm glow of yellow lemons and bananas is my favorite thing to look at.",
        "My whole wardrobe is yellow; nothing makes me happier than wearing bright yellow.",
        "I painted the kitchen a sunny yellow and bought yellow plates to match it.",
        "Yellow, yellow, yellow -- canary, butter, gold, and lemon are the best colors.",
        "I love how a field of yellow daffodils glows; yellow is simply the loveliest color.",
    ],
}
NEUTRAL10 = [
    "The history of papermaking stretches back nearly two thousand years across many regions.",
    "A river system is shaped by the land it crosses, carving valleys and depositing sediment.",
    "Modern weather forecasting combines satellite imagery, ground sensors, and numerical models.",
    "The committee met on Thursday to review the budget and assign responsibilities.",
    "Software is written in layers, with low-level instructions beneath friendlier interfaces.",
    "Limestone caves form slowly as mildly acidic water dissolves the rock over many years.",
    "Commuters filled the platform, glancing at the board as the next train approached.",
    "A good map balances detail and clarity without crowding the page too densely.",
    "The museum exhibit traces printing from carved blocks to modern high-speed presses.",
    "Coral reefs support diverse marine life within their intricate calcium structures.",
]
