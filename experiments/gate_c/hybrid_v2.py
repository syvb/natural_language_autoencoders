"""Production hybrid labeler: Anthropic API (direct or batch), 4 diff reads.

Modes:
  direct  — synchronous messages (smoke/validation)
  submit  — create a Message Batch (50% price), print batch id
  fetch   — retrieve batch results into the output json
"""

import argparse
import json
import re
from pathlib import Path

import anthropic

KEY = Path("~/.anthropic_key").expanduser().read_text().strip()
MODEL = "claude-haiku-4-5"
N_READS = 4

PROMPT = """You are a meticulous AI researcher producing a high-quality description of how a neural network's internal representation of one token position changed between an earlier layer and a later layer.

You have four kinds of evidence:

1. The TRUE SOURCE TEXT up to and including the token (this is the only text that actually exists):
<source_text>
{source}
</source_text>
(The position being analyzed is the final token of the source text: "{token}")

2. A description of the EARLIER layer's representation:
<earlier>
{expl_a}
</earlier>

3. A description of the LATER layer's representation:
<later>
{expl_b}
</later>

4. {n_reads} independent reads of the DIFFERENCE between the two representations. These are the only evidence directly grounded in the actual change, but each read independently invents specific details:
{diff_reads}

Known reliability facts about evidence 2-4: claims about genre, format, register, and syntactic structure are usually reliable; specific named entities are usually invented and differ between reads; quoted "sentences" inside them are almost always fabricated and rarely appear in the real source text.

Task: describe the CHANGE — semantic content that is present, strengthened, or sharpened in the later representation relative to the earlier one — as exactly 3 short text snippets of at most 40 words each. Snippets 1-2 describe the strengthened content; snippet 3 must be in this register: Final token {token} now more strongly expecting ... (describe how the continuation expectation changed).

Rules:
- Keep only claims supported by at least two independent pieces of evidence (stable across several difference-reads, or in a difference-read AND consistent with the later-vs-earlier contrast), and not contradicted by the source text.
- Treat every proper noun appearing in the difference-reads as fabricated unless it also appears in the source text; never copy such names into your snippets — describe the category instead.
- Each snippet must name the concrete content domain (the subject matter that strengthened); never write abstract-only claims like sharpened expectation of a continuation without saying what the continuation is about.
- Do not use quotation marks anywhere in your snippets. Refer to wording indirectly (e.g. 'the phrase ending the sentence', 'the closing clause') instead of quoting it.
- Write each snippet in change register: what is added, strengthened, sharpened, or shifted (e.g. "Sharpened expectation of ...", "Newly consolidated framing of ...").
- Do not use the words "layer", "vector", "representation", "description", "read", or "evidence" inside the snippets.

FINAL CHECK before answering: scan your snippets for the characters " or “ — if any appear, rewrite that snippet to describe the wording indirectly instead. Your snippets must contain zero quotation marks.

Answer in exactly this format:
<verdict>same</verdict> or <verdict>different</verdict>
<difference>
first snippet
second snippet
third snippet (optional)
</difference>"""

VERDICT_RE = re.compile(r"<verdict>\s*(same|different)\s*</verdict>", re.IGNORECASE)
DIFF_RE = re.compile(r"<difference>\s*(.*?)\s*</difference>", re.DOTALL)


def build_prompt(row: dict) -> str:
    reads = [f"<difference_read_{s+1}>\n{t}\n</difference_read_{s+1}>"
             for s, t in enumerate(row["diff_reads"][:N_READS])]
    return PROMPT.format(source=row["source"], token=row["token"],
                         expl_a=row["expl_a"], expl_b=row["expl_b"],
                         n_reads=len(reads), diff_reads="\n".join(reads))


def parse(text: str) -> dict:
    vm, dm = VERDICT_RE.search(text), DIFF_RE.search(text)
    diff = dm.group(1).strip() if dm else None
    if diff:
        diff = diff.replace('"', "").replace("\u201c", "").replace("\u201d", "")
    return {"verdict": vm.group(1).lower() if vm else None,
            "difference": diff,
            "raw": text[:900]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["direct", "submit", "fetch"])
    ap.add_argument("--inputs", required=True,
                    help="json: {position_id: {source,token,expl_a,expl_b,diff_reads[]}}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    client = anthropic.Anthropic(api_key=KEY)
    inputs = json.loads(Path(args.inputs).read_text())
    ids = sorted(inputs, key=int)[: args.limit]

    if args.mode == "direct":
        out_path = Path(args.out)
        results = json.loads(out_path.read_text()) if out_path.exists() else {}
        usage = [0, 0]
        for i in ids:
            if i in results:
                continue
            r = client.messages.create(
                model=args.model, max_tokens=600,
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
                            model=args.model, max_tokens=600,
                            messages=[{"role": "user",
                                       "content": build_prompt(inputs[i])}]))
                for i in ids]
        # batches accept up to 100k requests; one batch suffices
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
                results[i] = {"verdict": None, "difference": None,
                              "raw": f"ERROR:{res.result.type}"}
        Path(args.out).write_text(json.dumps(results))
        ok = sum(1 for r in results.values() if r["difference"])
        print(f"[fetch] {ok}/{len(results)} parsed -> {args.out}")


if __name__ == "__main__":
    main()
