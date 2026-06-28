"""Generate 1000 'likes the color yellow' two-choice items via OpenRouter (Claude Haiku 4.5).

Same nrimsky/CAA schema as the seed 40-item set, with the same confound controls:
  - matching option = genuine preference/affection for the color YELLOW as a color;
  - non-matching option = a different specific named color OR genuine color-indifference (mixed);
  - no warmth / gold / happiness / energy / sunshine-as-mood smuggling;
  - parallel option phrasing so only the color/preference differs.

Robustness: we don't trust the model's A/B placement -- we locate the yellow option by content,
then assign the final A/B position ourselves to get EXACTLY 500/500 balance. Dedup by question.

No Claude-API skill, no v1 code. Key from ~/.openrouter_key. Run: python3 gen_yellow.py
"""
import json
import os
import random
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = open(os.path.expanduser("~/.openrouter_key")).read().strip()
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-haiku-4.5"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "likes_yellow_1000.json")

TARGET = 1000
PER_BATCH = 25
MAX_BATCHES = 96          # safety cap; the run stops as soon as TARGET unique items exist
TEMP = 0.95

# scenarios where a YELLOW thing and a non-yellow thing are both natural choices
THEMES = [
    "fruit at the market (banana, lemon, pineapple vs apple, lime, plum)",
    "flowers for a vase (sunflower, daffodil vs rose, tulip, iris)",
    "a toy from the bin (rubber duck, toy school bus vs blue boat, fire truck)",
    "breakfast on the table (scrambled eggs, cornbread vs oatmeal, berries)",
    "a bird at the feeder (canary, chick vs robin, blue jay)",
    "a snack from the pantry (banana chips, popcorn vs pretzels, licorice)",
    "produce from the garden (corn, squash vs broccoli, eggplant)",
    "candy from the jar (lemon drops, butterscotch vs peppermint, grape)",
    "a vehicle passing by (a taxi, a school bus vs a fire truck, a police car)",
    "something on toast (butter, honey vs jam, chocolate spread)",
    "dessert on the menu (lemon meringue, custard vs chocolate cake, red velvet)",
    "things growing in a meadow (dandelions, buttercups vs clover, violets)",
    "a character sticker (a minion, SpongeBob vs a Smurf, a teddy)",
    "an item at the deli (cheddar, mustard vs salami, olives)",
    "stationery on the desk (a highlighter, a legal pad vs a red pen, a blue folder)",
    "a ball from the equipment shed (a tennis ball vs a basketball, a green frisbee)",
    "a smoothie flavor (banana-mango vs blueberry, mixed berry)",
    "a pepper from the basket (a yellow bell pepper vs a green pepper, a red chili)",
    "an animal at the petting farm (a duckling, a chick vs a piglet, a calf)",
    "a drink at the stand (lemonade vs cola, iced tea)",
    "a pool float (a duck float vs a blue ring, a flamingo float)",
    "a baked good at the bakery (lemon cake, cornbread vs rye bread, a brownie)",
    "fruit in a lunchbox (a banana vs an apple, a handful of grapes)",
    "a squash at the farm stand (a butternut vs a green zucchini, an eggplant)",
    "a safety item on site (a hi-vis vest, a hard hat vs a grey jacket, black boots)",
    "a cereal in the bowl (corn flakes vs bran flakes, chocolate puffs)",
    "a thing in the toy chest (a yellow ball vs a green chair, a blue block)",
    "a flower to plant (marigolds vs lavender, geraniums)",
    "a cheese for the board (cheddar vs brie, a blue cheese)",
    "an emoji to send (a sun, a smiley vs a moon, a heart)",
    "an egg dish (an omelette, a fried egg vs a poached egg, hash browns)",
    "a fruit for the smoothie (a mango vs a kiwi, a raspberry)",
    "a bath toy for the kid (a rubber duck vs a blue whale, a green frog)",
    "a spread for the biscuit (honey vs strawberry jam, nutella)",
    "a bloom in the bouquet (a sunflower vs a red carnation, a white lily)",
    "a vegetable on the plate (corn on the cob vs green beans, a beet)",
    "a candy bar wrapper choice (a banana taffy vs a mint bar, a cherry chew)",
    "a fruit at the orchard (a ripe lemon vs a green pear, a red plum)",
    "a kite at the park (a sun-shaped kite vs a blue diamond kite, a red box kite)",
    "a pet bird to adopt (a canary vs a parakeet, a dove)",
]

PROMPT = """You are creating items for a "likes the color yellow" contrastive dataset, in the style \
of the nrimsky/CAA steering datasets.

Produce {k} diverse two-choice items about this theme: {theme}.

Each item is a natural choice between two CONCRETE THINGS (objects, foods, animals...) -- NOT an \
abstract "pick a color" question. One thing is characteristically YELLOW; the other is \
characteristically NOT yellow. Someone who likes yellow picks the yellow thing. The yellowness should \
be carried by WHAT the thing IS, e.g. bananas vs apples, a sunflower vs a red rose, a rubber duck vs a \
blue boat, a school bus vs a fire truck, a canary vs a robin, corn vs broccoli, a lemon vs a lime.

Prefer things that are inherently yellow (banana, lemon, sunflower, corn, butter, cheddar, egg yolk, \
daffodil, buttercup, canary, school bus, taxi, rubber duck, chick, duckling, pineapple, mustard, honey, \
custard, popcorn, a tennis ball, a highlighter, lemonade, scrambled eggs, butterscotch...). Don't say \
the word "yellow" -- let the object carry it. An explicit "the yellow X vs the green Y" object pair is \
okay occasionally, but keep those rare.

Output a JSON array of {k} objects, each with exactly these fields:
- "question": a natural question framing the choice (e.g. "At the market, which do you reach for?"). \
NO "Choices:" block, no "(A)"/"(B)".
- "yellow_option": the yellow thing, a short concrete noun phrase (e.g. "A bunch of bananas." / "The rubber duck.").
- "other_option": a comparable NON-yellow thing, ideally the same kind (e.g. "A bag of apples." / "The blue boat."). \
Occasionally plain indifference ("Either's fine.").

Rules (strict):
- Both options are concrete THINGS, parallel in length/style; the only meaningful difference is that one is yellow.
- yellow_option must genuinely be yellow; other_option must clearly NOT be yellow.
- Preference shown ONLY by the choice -- no testimonials. FORBIDDEN anywhere: "favorite", "love", \
"adore", "passionate", "obsessed", "deeply", "the most", "go-to", "huge/biggest fan".
- No "warmth", "gold/luxury", "happiness/mood", "energy". Every question distinct and natural.

Output ONLY the JSON array. No prose, no markdown code fences."""

# performative / confound phrasing rejected from EITHER option
_PERFORMATIVE = re.compile(
    r"\b(favou?rite|loves?|adores?|passionate|obsess\w*|deeply|go-to|the most|"
    r"can'?t resist|cannot resist|huge fan|biggest fan|gold|golden|my colou?r)\b", re.I)

# canonical yellow THINGS. yellow_option must match this (or say "yellow"); other_option must NOT.
_YELLOW_LEX = re.compile(
    r"\b(lemon|lemons|lemonade|lemon drops?|lemon meringue|lemon cake|banana|bananas|banana chips|"
    r"sunflowers?|daffodils?|buttercups?|dandelions?|marigolds?|saffron|primrose|canary|canaries|"
    r"school ?bus(es)?|taxi|taxis|taxi cab|egg ?yolks?|yolks?|scrambled eggs?|fried egg|omelets?|"
    r"omelettes?|rubber ?ducks?|ducklings?|chicks?|corn|cornbread|corn flakes|popcorn|butter|"
    r"buttercream|butterscotch|cheddar|custard|honey|mustard|pineapples?|tennis ball|highlighter|"
    r"legal pad|sticky notes?|post-?it|minions?|spongebob|pikachu|big bird|tweety|bumblebees?|"
    r"squash|butternut|maize|mango|mangoes|candy corn|mac and cheese|hi-?vis|hard hat)\b", re.I)

usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}


def call(theme):
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": 4000,
                       "messages": [{"role": "user", "content": PROMPT.format(k=PER_BATCH, theme=theme)}]}).encode()
    req = urllib.request.Request(URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=180))
            u = r.get("usage", {})
            usage_tot["prompt_tokens"] += u.get("prompt_tokens", 0)
            usage_tot["completion_tokens"] += u.get("completion_tokens", 0)
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 3:
                print("  request failed:", e, flush=True)
                return ""
    return ""


def parse_array(text):
    t = text.strip()
    if "```" in t:
        t = re.sub(r"^```[a-zA-Z]*\n?|```$", "", t.strip()).strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        return json.loads(t[i: j + 1])
    except Exception:
        return []


def normalize(obj):
    """Return (question, yellow_opt, other_opt) or None if it fails the content checks.

    Yellow side is identified by the model's own field -- not by searching for the word, since the
    whole point is that 'yellow' never appears literally in the yellow option.
    """
    if not isinstance(obj, dict):
        return None
    q, yo, oo = obj.get("question"), obj.get("yellow_option"), obj.get("other_option")
    if not (isinstance(q, str) and isinstance(yo, str) and isinstance(oo, str)):
        return None
    q, yo, oo = q.strip(), yo.strip(), oo.strip()
    if not q or not yo or not oo:
        return None
    if "Choices:" in q or "(A)" in q or "(B)" in q:    # stem only
        return None
    ly, lo = yo.lower(), oo.lower()
    if "gold" in ly:                                   # gold = luxury confound, not wanted
        return None
    if not (_YELLOW_LEX.search(yo) or "yellow" in ly):  # yellow option must be recognizably yellow
        return None
    if "yellow" in lo or _YELLOW_LEX.search(oo):        # other option must be clearly non-yellow
        return None
    if _PERFORMATIVE.search(yo) or _PERFORMATIVE.search(oo):
        return None
    return q, yo, oo


def _ingest(texts, seen, triples):
    for text in texts:
        for obj in parse_array(text):
            tri = normalize(obj)
            if not tri:
                continue
            key = re.sub(r"\s+", " ", tri[0].lower())
            if key in seen:
                continue
            seen.add(key)
            triples.append(tri)


def main(sample=False):
    seen = set()
    triples = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        if sample:
            _ingest(list(ex.map(call, THEMES[:2])), seen, triples)
            print(f"\n=== SAMPLE: {len(triples)} items (yellow option shown first) ===")
            for head, yo, oo in triples:
                print(f"\nQ: {head}\n  [yellow] {yo}\n  [other ] {oo}")
            pt, ct = usage_tot["prompt_tokens"], usage_tot["completion_tokens"]
            print(f"\nsample tokens prompt={pt} completion={ct} | cost ${pt/1e6+ct/1e6*5:.4f}")
            return
        bi = 0
        while len(triples) < TARGET and bi < MAX_BATCHES:
            chunk = [THEMES[(bi + j) % len(THEMES)] for j in range(8)]
            _ingest(list(ex.map(call, chunk)), seen, triples)
            bi += 8
            print(f"  {bi} batches: {len(triples)} unique so far", flush=True)

    rng = random.Random(7)
    rng.shuffle(triples)
    triples = triples[:TARGET]
    if len(triples) < TARGET:
        print(f"WARNING: only {len(triples)} unique items (< {TARGET}) after {MAX_BATCHES} batches.", flush=True)

    # exact 50/50 balance: first half yellow=(A), second half yellow=(B), then shuffle order
    items = []
    half = len(triples) // 2
    for i, (head, yellow_opt, other_opt) in enumerate(triples):
        yellow_is_a = i < half
        if yellow_is_a:
            q = f"{head}\n\nChoices:\n(A) {yellow_opt}\n(B) {other_opt}"
            mb, nb = "(A)", "(B)"
        else:
            q = f"{head}\n\nChoices:\n(A) {other_opt}\n(B) {yellow_opt}"
            mb, nb = "(B)", "(A)"
        items.append({"question": q, "answer_matching_behavior": mb, "answer_not_matching_behavior": nb})
    rng.shuffle(items)

    json.dump(items, open(OUT, "w"), indent=2)
    from collections import Counter
    bal = Counter(it["answer_matching_behavior"] for it in items)
    pt, ct = usage_tot["prompt_tokens"], usage_tot["completion_tokens"]
    cost = pt / 1e6 * 1.0 + ct / 1e6 * 5.0
    print(f"\nwrote {len(items)} items to {OUT} | A/B balance {dict(bal)}")
    print(f"tokens: prompt={pt} completion={ct} | est cost ${cost:.3f} (Haiku 4.5 $1/$5 per M)")
    print("GEN_YELLOW_DONE")


if __name__ == "__main__":
    import sys
    main(sample=len(sys.argv) > 1 and sys.argv[1] == "sample")
