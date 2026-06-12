"""Gate E0.1: counterfactual edit generation (Haiku) + mechanical validation.

Haiku proposes typed minimal edits; we keep only edits that are
TOKEN-LENGTH-PRESERVING with identical token suffix (>=60 aligned tokens
downstream), source span unique in the doc, edit not in the last 30% of
the doc. Validation is purely mechanical (law 4: structural enforcement).

Modes: pilot (direct API, N docs) / submit / fetch (batch) / validate.
Outputs: edits_raw.json {doc_idx: [edit,...]} -> validate ->
         pairs_e.json [{pair_id, doc_idx, type, source, target,
                        text, text_edited, edit_end_tok, n_aligned}]
"""

import argparse
import json
import re
from pathlib import Path

import anthropic
from transformers import AutoTokenizer

WORK = Path(__file__).parent
KEY = Path("~/.anthropic_key").expanduser().read_text().strip()
MODEL = "claude-haiku-4-5"
MIN_ALIGNED = 60

PROMPT = """Propose 6 minimal counterfactual edits to the document below. Each edit replaces ONE contiguous span with a different span, changing the meaning while keeping everything else identical.

Required edit types (provide at least one of each where the text allows, else substitute another type):
- role_reversal: swap two participants so who-does-what-to-whom reverses (e.g. "Alice paid Bob" -> "Bob paid Alice"). The two swapped names/phrases must both already be in the span.
- antonym: replace a verb/adjective with its opposite (won->lost, rose->fell, cheap->costly).
- attribute: change a number, date, money amount, or quantity to a clearly different one of the same format ($4.6 million -> $46 million).
- entity_swap: replace a person/org/place with a DIFFERENT plausible one of similar length — only if that entity is mentioned again later in the document.
- discourse: change certainty or stance (will->may, confirmed->denied, is->is reportedly... keep length similar).

Hard rules:
- The source span must appear EXACTLY ONCE in the document, verbatim.
- The replacement should be roughly the same length in characters.
- The edit must occur in the FIRST 60% of the document.
- Do not change anything else; no edits inside URLs or markup.

<document>
{doc}
</document>

Answer with ONLY a JSON array, no other text:
[{{"type": "...", "source": "exact span from document", "target": "replacement span"}}, ...]"""

JSON_RE = re.compile(r"\[.*\]", re.DOTALL)


def validate(docs, edits_raw, tok):
    pairs = []
    stats = {"proposed": 0, "not_unique": 0, "too_late": 0, "tok_mismatch": 0,
             "short_suffix": 0, "ok": 0}
    for di, edits in edits_raw.items():
        text = docs[int(di)]["text"]
        for e in edits or []:
            stats["proposed"] += 1
            src, tgt = e.get("source", ""), e.get("target", "")
            if not src or not tgt or text.count(src) != 1:
                stats["not_unique"] += 1
                continue
            pos_char = text.index(src)
            if pos_char > 0.6 * len(text):
                stats["too_late"] += 1
                continue
            edited = text.replace(src, tgt, 1)
            io = tok(text, add_special_tokens=False)["input_ids"]
            ie = tok(edited, add_special_tokens=False)["input_ids"]
            if len(io) != len(ie):
                stats["tok_mismatch"] += 1
                continue
            diff = [k for k in range(len(io)) if io[k] != ie[k]]
            if not diff:
                stats["tok_mismatch"] += 1
                continue
            edit_end = diff[-1] + 1
            if len(io) - edit_end < MIN_ALIGNED:
                stats["short_suffix"] += 1
                continue
            stats["ok"] += 1
            pairs.append({"pair_id": len(pairs), "doc_idx": int(di),
                          "type": e.get("type", "?"), "source": src,
                          "target": tgt, "text": text, "text_edited": edited,
                          "edit_end_tok": edit_end,
                          "n_aligned": len(io) - edit_end})
    return pairs, stats


def parse_edits(text):
    m = JSON_RE.search(text)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "submit", "fetch", "validate"])
    ap.add_argument("--docs", default="docs_e.json")
    ap.add_argument("--out", default="edits_raw.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--batch-id", default=None)
    args = ap.parse_args()

    docs = json.loads((WORK / args.docs).read_text())
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    client = anthropic.Anthropic(api_key=KEY)

    if args.mode == "pilot":
        out = {}
        for d in docs[: args.n]:
            r = client.messages.create(model=MODEL, max_tokens=1200,
                messages=[{"role": "user",
                           "content": PROMPT.format(doc=d["text"])}])
            out[str(d["doc_idx"])] = parse_edits(r.content[0].text)
        Path(WORK / args.out).write_text(json.dumps(out))
        pairs, stats = validate(docs, out, tok)
        print("pilot stats:", stats)
        for p in pairs[:5]:
            print(f"  [{p['type']}] '{p['source'][:60]}' -> '{p['target'][:60]}' (aligned {p['n_aligned']})")

    elif args.mode == "submit":
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        reqs = [Request(custom_id=f"doc-{d['doc_idx']}",
                        params=MessageCreateParamsNonStreaming(
                            model=MODEL, max_tokens=1200,
                            messages=[{"role": "user",
                                       "content": PROMPT.format(doc=d["text"])}]))
                for d in docs]
        batch = client.messages.batches.create(requests=reqs)
        print(f"[submit] {batch.id} with {len(reqs)} requests")

    elif args.mode == "fetch":
        batch = client.messages.batches.retrieve(args.batch_id)
        print(f"[fetch] {batch.processing_status} {batch.request_counts}")
        if batch.processing_status != "ended":
            return
        out = {}
        for res in client.messages.batches.results(args.batch_id):
            di = res.custom_id.removeprefix("doc-")
            out[di] = (parse_edits(res.result.message.content[0].text)
                       if res.result.type == "succeeded" else [])
        Path(WORK / args.out).write_text(json.dumps(out))
        print(f"[fetch] {len(out)} docs")

    elif args.mode == "validate":
        edits_raw = json.loads((WORK / args.out).read_text())
        pairs, stats = validate(docs, edits_raw, tok)
        Path(WORK / "pairs_e.json").write_text(json.dumps(pairs))
        from collections import Counter
        print("stats:", stats)
        print("by type:", Counter(p["type"] for p in pairs))


if __name__ == "__main__":
    main()
