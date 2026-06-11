"""Gate D arm-H labeler: Haiku writes attention-write descriptions from
MECHANICAL evidence bundles (attention patterns + head contributions).

Unlike Gate C's hybrid labeler, the evidence here is measured, not
model-generated — the prompt treats it as ground truth and requires every
content claim to quote tokens present in the bundle.

Modes: direct / submit / fetch (hybrid_v2.py plumbing).
Inputs: bundles_d.json from build_bundles_d.py.
"""

import argparse
import json
import re
from pathlib import Path

import anthropic

KEY = Path("~/.anthropic_key").expanduser().read_text().strip()
MODEL = "claude-haiku-4-5"

PROMPT = """You are describing what an attention block inside a language model wrote into the network at one specific token position. You have mechanically measured evidence — it is reliable, not guessed.

The text up to and including the position (the position is the FINAL token, "{token}" — nothing after it exists yet):
<source_text>
{source}
</source_text>

Measured evidence about the attention write at this position:
<evidence>
{evidence}
</evidence>
Each line lists earlier tokens that were read together, with a few surrounding words to locate them. Lines are ordered strongest first; every quoted word comes from at or before the position.

Task: in 2-4 plain sentences, describe what this attention write carries: which earlier content it retrieved and what that contributes at the position "{token}". Be concrete and specific.

Rules:
- Every piece of content you mention must be quoted or near-quoted from the evidence or the source text. Never introduce content that is not there, and never guess at what might come after the position.
- Lead with the most specific retrieved content — names, places, numbers, topic words pulled from text BEFORE the current sentence. Summarize retrievals of the position's own phrase in one short clause at the end (e.g., plus carry-over of the local phrase "...").
- A reader holding labels from 100 other positions must be able to pick this one out. Claims that fit any position — anchors the local grammatical structure, reinforces the sentence being completed — are banned.
- Never begin the label with the words "The write" or "The attention" — begin with the retrieved content or its action (e.g., Pulls the company name "Atlas Copco" and the date "June 24" from the earlier announcement ... / "Culjak" and "Estes" arrive from the signature block ...). Vary sentence shapes from label to label.
- Never write numeric weights, percentages, or shares — say at most mainly / also / faintly.
- Do not use these words: layer, vector, representation, neural, head, channel, path, share, weight, attention, retrieval, retrievals. Do not use suggesting, appears to, likely, preparing, setting up, about to. Never refer to "the model" or "the network".

FINAL CHECK before answering: (1) if any sentence contains a banned word (including "retrieval"), or a number that is a weight or percentage, rewrite it; (2) if any quoted phrase contains words not present in the source text or evidence above, trim it; (3) if your label begins with "The write" or "The attention", rewrite the opening to begin with content.

Answer in exactly this format:
<label>
your 2-4 sentences
</label>"""

QUIET_PROMPT_SUFFIX = """

Note: the write at this position is unusually weak (bottom decile). Say that first, in those terms (weak write, little beyond local carry-over), then at most one more sentence naming the single strongest genuine retrieval if there is one. Two sentences maximum."""

LABEL_RE = re.compile(r"<label>\s*(.*?)\s*</label>", re.DOTALL)


def render_evidence(b: dict) -> str:
    # qualitative only — numeric weights/shares in the rendering were echoed
    # straight into 34/100 pilot labels
    def strength(w):
        return "strong" if w >= 0.3 else ("moderate" if w >= 0.1 else "faint")
    lines = []
    for h in b["heads"]:
        srcs = ", ".join(
            f"\"{s['token'].strip()}\" (in: \"{s['near'].strip()}\", {strength(s['weight'])})"
            for s in h["attends_to"])
        lines.append(f"- reads {srcs}")
    return "\n".join(lines) if lines else "- (no retrieval above threshold)"


def build_prompt(b: dict) -> str:
    p = PROMPT.format(token=b["target_token"], source=b["context_tail"],
                      evidence=render_evidence(b))
    if b.get("quiet"):
        p += QUIET_PROMPT_SUFFIX
    return p


def parse(text: str) -> dict:
    m = LABEL_RE.search(text)
    return {"label": m.group(1).strip() if m else None, "raw": text[:900]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["direct", "submit", "fetch"])
    ap.add_argument("--inputs", default="bundles_d.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-every", type=int, default=1,
                    help="take every k-th position (pilot spread)")
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=KEY)
    inputs = json.loads(Path(args.inputs).read_text())
    ids = sorted(inputs, key=int)[::args.sample_every][: args.limit]

    if args.mode == "direct":
        out_path = Path(args.out)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        usage = [0, 0]
        for i in ids:
            if i in results:
                continue
            r = client.messages.create(
                model=args.model, max_tokens=500,
                messages=[{"role": "user", "content": build_prompt(inputs[i])}])
            usage[0] += r.usage.input_tokens
            usage[1] += r.usage.output_tokens
            results[i] = parse(r.content[0].text)
        out_path.write_text(json.dumps(results))
        print(f"[direct] {len(results)} labels, tokens in/out {usage}")

    elif args.mode == "submit":
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        reqs = [Request(custom_id=f"pos-{i}",
                        params=MessageCreateParamsNonStreaming(
                            model=args.model, max_tokens=500,
                            messages=[{"role": "user",
                                       "content": build_prompt(inputs[i])}]))
                for i in ids]
        batch = client.messages.batches.create(requests=reqs)
        print(f"[submit] batch {batch.id} with {len(reqs)} requests")

    elif args.mode == "fetch":
        batch = client.messages.batches.retrieve(args.batch_id)
        print(f"[fetch] status {batch.processing_status} counts {batch.request_counts}")
        if batch.processing_status != "ended":
            return
        results = {}
        for res in client.messages.batches.results(args.batch_id):
            i = res.custom_id.removeprefix("pos-")
            if res.result.type == "succeeded":
                results[i] = parse(res.result.message.content[0].text)
            else:
                results[i] = {"label": None, "raw": f"ERROR:{res.result.type}"}
        Path(args.out).write_text(json.dumps(results))
        ok = sum(1 for r in results.values() if r["label"])
        print(f"[fetch] {ok}/{len(results)} parsed -> {args.out}")


if __name__ == "__main__":
    main()
