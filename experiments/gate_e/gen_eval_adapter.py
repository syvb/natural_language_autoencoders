"""Gate E2 eval generation: reads through trained adapter W via sglang.

Jobs (env W_FILE selects adapter, suffix from its name):
  rr_fwd  : role_reversal positions (held-out TYPE), +delta  (n<=1500)
  rr_rev  : same positions, -delta (antisymmetry)
  lex_fwd : 1000 lexical holdout-doc positions, +delta
  lex_rev : same, -delta
Output rows include the job and sign for scoring.
"""

import asyncio
import glob
import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import orjson
import torch

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))
from nla_inference import EXPLANATION_RE, NLAClient

CONCURRENCY = 12
W_FILE = os.environ["W_FILE"]
TAG = Path(W_FILE).stem
z = np.load(WORK / W_FILE)
Wm, bv = z["W"].astype(np.float32), z["b"].astype(np.float32)

client = NLAClient(WORK / "models" / "av", sglang_url="http://localhost:30000")
_base_embeds, _ = client._build_embeds(torch.zeros(3584), None)
_cfg = client.cfg
_ids = client.tokenizer.apply_chat_template(
    [{"role": "user", "content": _cfg.actor_prompt_template.format(
        injection_char=_cfg.injection_char)}],
    tokenize=True, add_generation_prompt=True, return_dict=False)
_inj_pos = next(p for p in range(1, len(_ids) - 1)
                if _ids[p] == _cfg.injection_token_id)

meta = json.loads((WORK / "meta_e.json").read_text())
D = np.load(WORK / "delta20.npy", mmap_mode="r")
OUT = WORK / "gene_out"
OUT.mkdir(exist_ok=True)


def adapted_embeds(i, sign):
    emb = _base_embeds.copy()
    d = np.asarray(D[i], dtype=np.float32) * sign
    emb[_inj_pos] = d @ Wm + bv
    return emb


async def run_job(job, items):
    path = OUT / f"{TAG}_{job}.jsonl"
    done = set()
    for fn in glob.glob(str(path)):
        for line in open(fn):
            done.add(json.loads(line)["i"])
    todo = [i for i in items if i not in done]
    if not todo:
        print(f"[evalW] {job}: complete", flush=True)
        return
    print(f"[evalW] {job}: {len(todo)} to go", flush=True)
    sign = -1 if job.endswith("_rev") else 1
    sem = asyncio.Semaphore(CONCURRENCY)
    f = path.open("a")
    lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=300.0) as http:
        async def one(i):
            async with sem:
                body = orjson.dumps({"input_embeds": adapted_embeds(i, sign),
                                     "sampling_params": {"temperature": 0.0,
                                                         "max_new_tokens": 80,
                                                         "skip_special_tokens": False}},
                                    option=orjson.OPT_SERIALIZE_NUMPY)
                for a in range(3):
                    try:
                        r = await http.post("http://localhost:30000/generate",
                                            content=body,
                                            headers={"Content-Type": "application/json"})
                        r.raise_for_status()
                        break
                    except Exception:
                        if a == 2:
                            raise
                        await asyncio.sleep(5)
            out = r.json()
            text = (out[0] if isinstance(out, list) else out)["text"]
            m = EXPLANATION_RE.search(text)
            row = {"i": i, "job": job,
                   "explanation": m.group(1).strip() if m else None,
                   "raw": text[:300]}
            async with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
        await asyncio.gather(*(one(i) for i in todo))
    f.close()
    print(f"[evalW] {job}: done", flush=True)


def main():
    rng = np.random.default_rng(0)
    rr = [i for i, m in enumerate(meta) if m["type"] == "role_reversal"]
    rr = sorted(rng.choice(rr, min(1500, len(rr)), replace=False).tolist())
    lex = [i for i, m in enumerate(meta)
           if m["type"] != "role_reversal" and m["split"] == "holdout"]
    lex = sorted(rng.choice(lex, min(1000, len(lex)), replace=False).tolist())
    for job, items in [("rr_fwd", rr), ("rr_rev", rr),
                       ("lex_fwd", lex), ("lex_rev", lex)]:
        asyncio.run(run_job(job, items))
    print("[evalW] ALL DONE")


main()
