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

The text up to and including the position (the position is the final token, "{token}"):
<source_text>
{source}
</source_text>

Measured evidence about the attention write at this position:
<evidence>
{evidence}
</evidence>
Each entry is one retrieval channel: its share of the total write, and the earlier tokens it read from (with the surrounding words and the attention weight). High-weight distant tokens matter most; weight on the very first token of the document is a resting state and means nothing.

Task: in 2-4 plain sentences, describe what this attention write carries: which earlier content it retrieved and what that contributes at the position "{token}". Be concrete.

Rules:
- Every piece of content you mention must be quoted or near-quoted from the evidence or the source text. Use the actual retrieved words (e.g. retrieved "staffing" and "workers" from the earlier discussion of hospital staffing). Never introduce content that is not in the evidence.
- Prefer the retrievals with high weight and high share; ignore resting-state attention to the document start.
- If several heads retrieve the same nearby words (the position itself or the 1-2 tokens before it), summarize that once as local context carry-over.
- Plain declarative sentences. Do not use the words "layer", "vector", "representation", "neural", "head", or "channel" — write as if describing what the position now knows about earlier text, e.g. "the write retrieves ... " / "it also pulls in ...", never naming the mechanism.
- Do not speculate about why; describe what was retrieved and what it adds.

FINAL CHECK before answering: scan your sentences for the words head, heads, channel, layer, vector — if any appear, rewrite that sentence without naming the mechanism.

Answer in exactly this format:
<label>
your 2-4 sentences
</label>"""

QUIET_PROMPT_SUFFIX = """

Note: the total write at this position is unusually weak (bottom decile). If the evidence shows no meaningful retrieval, say in one sentence that the attention write here is weak and carries little beyond local context."""

LABEL_RE = re.compile(r"<label>\s*(.*?)\s*</label>", re.DOTALL)


def render_evidence(b: dict) -> str:
    lines = []
    for h in b["heads"]:
        srcs = ", ".join(
            f"\"{s['token'].strip()}\" (in: \"{s['near'].strip()}\", weight {s['weight']})"
            for s in h["attends_to"])
        lines.append(f"- share {h['share_of_write']:.0%}: reads {srcs}")
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
