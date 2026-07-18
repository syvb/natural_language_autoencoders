"""Screen candidate couplet framings: which (user prompt, first line) makes
Qwen3.6-27B concentrate its second-line rhyme on ONE content word?
Prints a tally per candidate; pick the winner for po_config.json."""
import json
from collections import Counter

from poetry_steer import Base, chat_prompt, last_words, N_SAMP
import poetry_steer as ps

CANDS = [
    ("P0_paper", "Write a rhyming couplet.", "He saw a carrot and had to grab it,"),
    ("P1_hungryrabbit", "Write a rhyming couplet about a hungry rabbit.",
     "He saw a carrot and had to grab it,"),
    ("P2_rabbit", "Write a rhyming couplet about a rabbit.",
     "He saw a carrot and had to grab it,"),
    ("P3_cathouse", "Write a rhyming couplet.", "The cat crept slowly toward the house,"),
    ("P4_catmouse", "Write a rhyming couplet.", "The old grey cat had spied a mouse,"),
    ("P5_catprompt", "Write a rhyming couplet about a cat.",
     "The old grey cat had spied a mouse,"),
    ("P6_animal", "Write a rhyming couplet about an animal.",
     "He saw a carrot and had to grab it,"),
]

base = Base()
tok = base.tok
out = {}
for name, user, line1 in CANDS:
    ps.COUPLET_L1 = line1
    ptxt = chat_prompt(user)
    ids = tok.encode(ptxt, add_special_tokens=False)
    gens = base.gen(ids, N_SAMP, 0, None)
    tally = Counter(last_words(gens))
    out[name] = dict(user=user, line1=line1, tally=dict(tally.most_common()),
                     gens=gens)
    print(f"[{name}] {dict(tally.most_common(6))}", flush=True)
json.dump(out, open("po_screen.json", "w"), indent=1)
print("SCREEN_DONE", flush=True)
