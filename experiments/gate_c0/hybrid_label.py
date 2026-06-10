"""Hybrid diff labels: Claude writes change-framed labels from grounded inputs.

Per position, the model gets: (1) the true source context (the only real
text), (2) the earlier-layer explanation, (3) the later-layer explanation,
(4) eight independent AV reads of the diff vector itself. Instructions
encode the generation-review findings: trust cross-read-stable structure,
discard unstable entities, never quote text absent from the source, write
in change register.
"""

import asyncio
import json
import re
import sys
from pathlib import Path

import httpx

WORK = Path(__file__).parent
MODEL = "anthropic/claude-sonnet-4.6"
KEY = Path("~/.openrouter_key").expanduser().read_text().strip()

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

4. Eight independent reads of the DIFFERENCE between the two representations. These are the only evidence directly grounded in the actual change, but each read independently invents specific details:
{diff_reads}

Known reliability facts about evidence 2-4: claims about genre, format, register, and syntactic structure are usually reliable; specific named entities are usually invented and differ between reads; quoted "sentences" inside them are almost always fabricated and rarely appear in the real source text.

Task: describe the CHANGE — semantic content that is present, strengthened, or sharpened in the later representation relative to the earlier one — as 2-3 short text snippets.

Rules:
- Keep only claims supported by at least two independent pieces of evidence (stable across several difference-reads, or in a difference-read AND consistent with the later-vs-earlier contrast), and not contradicted by the source text.
- Never include a named entity (person, brand, team, title) unless it appears in the source text.
- Do not use quotation marks anywhere in your snippets. Refer to wording indirectly (e.g. 'the phrase ending the sentence', 'the closing clause') instead of quoting it.
- Write each snippet in change register: what is added, strengthened, sharpened, or shifted (e.g. "Sharpened expectation of ...", "Newly consolidated framing of ...").
- Do not use the words "layer", "vector", "representation", "description", "read", or "evidence" inside the snippets.

Answer in exactly this format:
<verdict>same</verdict> or <verdict>different</verdict>
<difference>
first snippet
second snippet
third snippet (optional)
</difference>"""

VERDICT_RE = re.compile(r"<verdict>\s*(same|different)\s*</verdict>", re.IGNORECASE)
DIFF_RE = re.compile(r"<difference>\s*(.*?)\s*</difference>", re.DOTALL)


async def main(positions: list[int], out_name: str) -> None:
    meta = json.loads(Path("/home/debian/gate_a/meta.json").read_text())
    texts = json.loads(Path("/home/debian/gate_a/texts.json").read_text())
    bo = json.loads((WORK / "bo_texts.json").read_text())

    def expl(label, i):
        r = texts[label][i]
        return (r["explanation"] or r["raw"][:300]).strip()

    out_path = WORK / out_name
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    sem = asyncio.Semaphore(6)
    usage = {"prompt": 0, "completion": 0}

    async with httpx.AsyncClient(timeout=240.0) as http:
        async def one(i: int) -> None:
            reads = []
            for s, r in enumerate(bo["dav8"][i]):
                t = (r["explanation"] or r["raw"][:300]).strip()
                reads.append(f"<difference_read_{s+1}>\n{t}\n</difference_read_{s+1}>")
            body = {"model": MODEL, "temperature": 0.7, "max_tokens": 600,
                    "messages": [{"role": "user", "content": PROMPT.format(
                        source=meta[i]["context_tail"],
                        token=meta[i]["token_at_pos"],
                        expl_a=expl("L18", i), expl_b=expl("L22", i),
                        diff_reads="\n".join(reads))}]}
            async with sem:
                for attempt in range(4):
                    try:
                        r = await http.post("https://openrouter.ai/api/v1/chat/completions",
                                            json=body,
                                            headers={"Authorization": f"Bearer {KEY}"})
                        r.raise_for_status()
                        d = r.json()
                        if "choices" not in d:
                            raise RuntimeError(str(d)[:200])
                        break
                    except Exception as e:
                        if attempt == 3:
                            print(f"[hybrid] FAILED {i}: {e}", flush=True)
                            results[str(i)] = {"verdict": None, "difference": None, "raw": None}
                            return
                        await asyncio.sleep(4 * (attempt + 1))
            u = d.get("usage", {})
            usage["prompt"] += u.get("prompt_tokens", 0)
            usage["completion"] += u.get("completion_tokens", 0)
            text = d["choices"][0]["message"]["content"]
            vm, dm = VERDICT_RE.search(text), DIFF_RE.search(text)
            results[str(i)] = {"verdict": vm.group(1).lower() if vm else None,
                               "difference": dm.group(1).strip() if dm else None,
                               "raw": text[:900]}

        todo = [i for i in positions if str(i) not in results]
        print(f"[hybrid] labeling {len(todo)} positions", flush=True)
        await asyncio.gather(*(one(i) for i in todo))

    out_path.write_text(json.dumps(results))
    cost = usage["prompt"] * 3e-6 + usage["completion"] * 15e-6
    print(f"[hybrid] tokens {usage}  est cost ${cost:.2f}  -> {out_name}")


if __name__ == "__main__":
    pos = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(180))
    out = sys.argv[2] if len(sys.argv) > 2 else "hybrid_labels.json"
    asyncio.run(main(pos, out))
