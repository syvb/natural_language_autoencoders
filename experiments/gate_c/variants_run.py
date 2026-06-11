"""Gate C prompt-variant experiment: run 6 single-change variants of the
hybrid_v2 labeling prompt on 3 fixed test positions (temperature 0.7, 1 sample
each) and store raw outputs. Does not touch hybrid_v2.py or submit batches."""

import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/debian/gate_c")
import hybrid_v2 as hv  # noqa: E402  (reuse PROMPT, parse, KEY, MODEL)

import anthropic  # noqa: E402

FIXED = ["1784", "8311", "399"]
SPOT = ["9370", "9541", "2820"]

P = hv.PROMPT

# ---------------------------------------------------------------- variants --
# Each variant changes exactly ONE thing vs the production prompt.

variants = {}

# V-entity: harsher entity rule.
variants["V-entity"] = P.replace(
    "- Never include a named entity (person, brand, team, title) unless it "
    "appears in the source text.",
    "- Treat EVERY proper noun that appears in the earlier/later descriptions "
    "or the difference-reads as fabricated unless that exact name appears in "
    "the source text. Never write any proper noun (person, organization, "
    "agency, brand, place, product, title) in your snippets unless it appears "
    "verbatim in the source text. If the evidence converges on a category of "
    "entity, name the category (e.g. a federal safety agency), never a "
    "specific name.",
)
assert variants["V-entity"] != P

# V-short: exactly 2 snippets, <=40 words each.
variants["V-short"] = P.replace(
    "as 2-3 short text snippets.",
    "as exactly 2 snippets of at most 40 words each. Make every word carry "
    "information; cut filler.",
).replace(
    "first snippet\nsecond snippet\nthird snippet (optional)",
    "first snippet (max 40 words)\nsecond snippet (max 40 words)",
)
assert variants["V-short"] != P

# V-diverse: forbid formulaic openers; lead with content.
variants["V-diverse"] = P.replace(
    '- Write each snippet in change register: what is added, strengthened, '
    'sharpened, or shifted (e.g. "Sharpened expectation of ...", "Newly '
    'consolidated framing of ...").',
    "- Write each snippet in change register (what is added, strengthened, "
    "sharpened, or shifted), but NEVER open a snippet with a process word "
    "such as Sharpened, Strengthened, Shift, Shifted, Newly, Consolidated, "
    "or Emerging. Each snippet must begin with the concrete topic itself - "
    "the content noun phrase whose treatment changed (e.g. Banking "
    "authentication now framed as ..., The closing clause now leans "
    "toward ...). All snippets must open with different words.",
)
assert variants["V-diverse"] != P

# V-finaltoken: add a required third snippet in the AR-native register.
variants["V-finaltoken"] = P.replace(
    "as 2-3 short text snippets.",
    "as exactly 3 short text snippets.",
).replace(
    "first snippet\nsecond snippet\nthird snippet (optional)",
    'first snippet\nsecond snippet\nFinal token "{token}" ... now more '
    "strongly expecting ... (third snippet: REQUIRED, in exactly this "
    "register - state what continuation the position now expects more "
    "strongly than before)",
)
assert variants["V-finaltoken"] != P

# V-concrete: every snippet must name the concrete content domain.
variants["V-concrete"] = P.replace(
    "- Do not use the words \"layer\", \"vector\", \"representation\", "
    "\"description\", \"read\", or \"evidence\" inside the snippets.",
    "- Do not use the words \"layer\", \"vector\", \"representation\", "
    "\"description\", \"read\", or \"evidence\" inside the snippets.\n"
    "- Every snippet must name the concrete subject matter that "
    "strengthened (e.g. fleet maintenance costs, online banking "
    "authentication, probate procedure). Purely abstract claims such as "
    "sharpened expectation of a continuation or strengthened framing of "
    "the clause - with no domain content named - are forbidden.",
)
assert variants["V-concrete"] != P

# V-magnitude: add a magnitude field.
variants["V-magnitude"] = P.replace(
    "<verdict>same</verdict> or <verdict>different</verdict>\n<difference>",
    "<verdict>same</verdict> or <verdict>different</verdict>\n"
    "<magnitude>small</magnitude> or <magnitude>moderate</magnitude> or "
    "<magnitude>large</magnitude> (how large the overall change is)\n"
    "<difference>",
)
assert variants["V-magnitude"] != P


def build(prompt_template: str, row: dict) -> str:
    reads = [f"<difference_read_{s+1}>\n{t}\n</difference_read_{s+1}>"
             for s, t in enumerate(row["diff_reads"][:hv.N_READS])]
    return prompt_template.format(
        source=row["source"], token=row["token"],
        expl_a=row["expl_a"], expl_b=row["expl_b"],
        n_reads=len(reads), diff_reads="\n".join(reads))


def main():
    inputs = json.loads(Path("/home/debian/gate_c/hybrid_inputs_pre300.json").read_text())
    client = anthropic.Anthropic(api_key=hv.KEY)
    out = {}
    usage = [0, 0]
    for vname, vprompt in variants.items():
        out[vname] = {}
        for pos in FIXED:
            r = client.messages.create(
                model=hv.MODEL, max_tokens=600, temperature=0.7,
                messages=[{"role": "user",
                           "content": build(vprompt, inputs[pos])}])
            usage[0] += r.usage.input_tokens
            usage[1] += r.usage.output_tokens
            out[vname][pos] = r.content[0].text
            print(f"{vname} pos {pos}: {len(r.content[0].text)} chars")
    Path("/home/debian/gate_c/variants_results.json").write_text(
        json.dumps(out, indent=1))
    print("tokens in/out:", usage)


if __name__ == "__main__":
    main()
