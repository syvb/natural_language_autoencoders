"""Gate E1: zero-shot counterfactual-delta reads (dumb interface, no map).

Inject delta20 rescaled to injection norm into the frozen L20 AV.
Jobs: delta_read (2 samples @ temp 0.4 per position).
`python gen_e.py smoke` = 50 positions + CJK rate. SHARD/resume as usual.
"""

import asyncio
import glob
import json
import os
import re
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
CJK = re.compile(r"[぀-ヿ一-鿿]")

client = NLAClient(WORK / "models" / "av", sglang_url="http://localhost:30000")
_base_embeds, _ = client._build_embeds(torch.zeros(3584), None)
_cfg = client.cfg
_ids = client.tokenizer.apply_chat_template(
    [{"role": "user", "content": _cfg.actor_prompt_template.format(
        injection_char=_cfg.injection_char)}],
    tokenize=True, add_generation_prompt=True, return_dict=False)
_inj_pos = next(p for p in range(1, len(_ids) - 1)
                if _ids[p] == _cfg.injection_token_id
                and _ids[p - 1] == _cfg.injection_left_neighbor_id
                and _ids[p + 1] == _cfg.injection_right_neighbor_id)


def fast_embeds(v):
    emb = _base_embeds.copy()
    n = float(np.linalg.norm(v.astype(np.float64)))
    emb[_inj_pos] = v * (_cfg.injection_scale / max(n, 1e-9))
    return emb


meta = json.loads((WORK / "meta_e.json").read_text())
delta = np.load(WORK / "delta20.npy", mmap_mode="r")
OUT = WORK / "gene_out"
OUT.mkdir(exist_ok=True)


async def run_job(job, items, temp):
    shard_k, shard_n = 0, 1
    if os.environ.get("SHARD"):
        shard_k, shard_n = (int(x) for x in os.environ["SHARD"].split("/"))
    path = OUT / (f"{job}.s{shard_k}.jsonl" if shard_n > 1 else f"{job}.jsonl")
    done = set()
    for fn in glob.glob(str(OUT / f"{job}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            done.add((r["i"], r["s"]))
    todo = [(i, s) for (i, s) in items
            if (i, s) not in done and (i * 17 + s) % shard_n == shard_k]
    if not todo:
        print(f"[gen_e] {job}: complete", flush=True)
        return
    print(f"[gen_e] {job}: {len(todo)} to go", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    f = path.open("a")
    lock = asyncio.Lock()
    t0, n_done = time.time(), 0
    async with httpx.AsyncClient(timeout=300.0) as http:
        async def one(i, s):
            nonlocal n_done
            async with sem:  # body inside semaphore (OOM lesson)
                body = orjson.dumps(
                    {"input_embeds": fast_embeds(np.asarray(delta[i], dtype=np.float32)),
                     "sampling_params": {"temperature": temp,
                                         "max_new_tokens": 220,
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
            row = {"i": i, "s": s,
                   "explanation": m.group(1).strip() if m else None,
                   "raw": text[:400]}
            async with lock:
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_done += 1
                if n_done % 2000 == 0:
                    print(f"[gen_e] {job}: {n_done}/{len(todo)} "
                          f"({n_done/(time.time()-t0):.1f}/s)", flush=True)
        await asyncio.gather(*(one(i, s) for i, s in todo))
    f.close()
    print(f"[gen_e] {job}: done", flush=True)


def main():
    n = len(meta)
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        asyncio.run(run_job("delta_smoke", [(i, 0) for i in range(50)], 0.4))
        bad = tot = 0
        for line in (OUT / "delta_smoke.jsonl").open():
            r = json.loads(line)
            t = r["explanation"] or r["raw"] or ""
            tot += 1
            bad += bool(CJK.search(t))
        print(f"[gen_e] SMOKE: {bad}/{tot} CJK")
        return
    hold = [i for i, m in enumerate(meta) if m["split"] == "holdout"]
    train_insp = [i for i, m in enumerate(meta) if m["split"] == "train"][:1000]
    asyncio.run(run_job("delta_read",
                        [(i, s) for i in hold + train_insp for s in (0, 1)], 0.4))
    print("[gen_e] ALL DONE")


main()
