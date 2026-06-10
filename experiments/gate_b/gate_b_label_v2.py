"""Gate B: can Claude produce layer-diff labels that beat a null control?

Conditions (180 positions each, from Gate A's saved explanations):
  real_19_21 : A = L19 explanation, B = L21 explanation (straddles L20)
  real_18_22 : A = L18, B = L22 (wider straddle)
  null_20_20 : A = L20, B = L20-repeat — two explanations of the SAME vector;
               any "difference" found here is hallucinated noise.

The prompt is blind to condition. Claude returns a same/different verdict
plus difference snippets written as direct content descriptions (AR-decodable
style, no meta-language).
"""

import asyncio
import json
import re
from pathlib import Path

import httpx

WORK = Path(__file__).parent
GATE_A = Path("/home/debian/gate_a")
MODEL = "anthropic/claude-sonnet-4.6"
KEY = Path("~/.openrouter_key").expanduser().read_text().strip()
CONCURRENCY = 8

PROMPT = """You are a meticulous AI researcher comparing two descriptions of neural-network activation vectors. Description A and Description B each describe an activation vector recorded at the same token position in the same document. The two vectors may be identical or may differ.

Description A:
<description_a>
{a}
</description_a>

Description B:
<description_b>
{b}
</description_b>

First decide whether the two descriptions reflect a substantive difference in the semantic content of the underlying vectors, or merely rephrase the same content. Important: two descriptions of the SAME vector will still differ in wording and emphasis — judge content, not phrasing.

Then write the difference as 2-3 short text snippets. CRITICAL: your snippets must describe ONLY the content that changed — content present or strengthened in B that is absent or weaker in A. Do NOT restate, summarize, or paraphrase any content that appears in both descriptions, even partially; imagine subtracting A from B and describing only the remainder. Write each snippet as a direct description of that residual content, in the same style as the input snippets. Do not use the words "A", "B", "description", or "vector" inside the snippets. If there is no substantive difference, still write snippets for whatever slight shift in emphasis exists, applying the same subtraction rule.

Answer in exactly this format:
<verdict>same</verdict> or <verdict>different</verdict>
<difference>
first snippet
second snippet
third snippet (optional)
</difference>"""

VERDICT_RE = re.compile(r"<verdict>\s*(same|different)\s*</verdict>", re.IGNORECASE)
DIFF_RE = re.compile(r"<difference>\s*(.*?)\s*</difference>", re.DOTALL)


def expl(texts: dict, label: str, i: int) -> str:
    r = texts[label][i]
    return r["explanation"] if r["explanation"] is not None else r["raw"]


async def main() -> None:
    texts = json.loads((GATE_A / "texts.json").read_text())
    n = len(texts["L20"])
    conditions = {
        "real_18_22_v2": ("L18", "L22"),
        "null_20_20_v2": ("L20", "L20_rep"),
    }
    out_path = WORK / "gate_b_labels_v2.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    usage = {"prompt": 0, "completion": 0}
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=180.0) as http:
        async def one(cond: str, i: int, a: str, b: str) -> dict:
            body = {"model": MODEL, "temperature": 1.0, "max_tokens": 500,
                    "messages": [{"role": "user",
                                  "content": PROMPT.format(a=a, b=b)}]}
            async with sem:
                for attempt in range(4):
                    try:
                        r = await http.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            json=body,
                            headers={"Authorization": f"Bearer {KEY}"})
                        r.raise_for_status()
                        d = r.json()
                        if "choices" not in d:
                            raise RuntimeError(str(d)[:200])
                        break
                    except Exception as e:
                        if attempt == 3:
                            print(f"[label] FAILED {cond}#{i}: {e}", flush=True)
                            return {"verdict": None, "difference": None, "raw": None}
                        await asyncio.sleep(4 * (attempt + 1))
            u = d.get("usage", {})
            usage["prompt"] += u.get("prompt_tokens", 0)
            usage["completion"] += u.get("completion_tokens", 0)
            text = d["choices"][0]["message"]["content"]
            vm, dm = VERDICT_RE.search(text), DIFF_RE.search(text)
            return {"verdict": vm.group(1).lower() if vm else None,
                    "difference": dm.group(1).strip() if dm else None,
                    "raw": text[:800]}

        for cond, (la, lb) in conditions.items():
            if cond in results:
                print(f"[label] {cond} cached, skip", flush=True)
                continue
            print(f"[label] {cond}: {n} pairs ({la} vs {lb})", flush=True)
            rows = await asyncio.gather(
                *(one(cond, i, expl(texts, la, i), expl(texts, lb, i))
                  for i in range(n)))
            results[cond] = list(rows)
            out_path.write_text(json.dumps(results))

    cost = usage["prompt"] * 3e-6 + usage["completion"] * 15e-6
    print(f"[label] tokens: {usage}  est cost: ${cost:.2f}")
    for cond in conditions:
        rows = results[cond]
        ok = [r for r in rows if r["verdict"]]
        diff_rate = sum(r["verdict"] == "different" for r in ok) / max(len(ok), 1)
        print(f"  {cond}: verdict=different {diff_rate:.2%}  "
              f"(parsed {len(ok)}/{len(rows)})")


asyncio.run(main())
