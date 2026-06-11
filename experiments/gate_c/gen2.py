"""Gate C generation via SGLang (resumable, throughput-aware).

Jobs per position (meta2.json order):
  expl18 / expl20 / expl22 : 1 sample @ temp 0.4 (full-activation explanations)
  diff1                    : 1 sample @ temp 0.4 of v22-v18 (arm A training label)
  hybrid reads             : 4 samples @ temp 1.0 (first 8k train positions)
  eval sets                : 16 samples @ temp 1.0 (first 150 holdout positions)

`python gen2.py timing` runs a 1k-generation throughput test and exits.
Results stream to gen2_out/<job>.jsonl (one line per completed item).
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
N_HYBRID = 8000
N_EVAL = 150

client = NLAClient(WORK / "models" / "av", sglang_url="http://localhost:30000")
# Precompute the canonical prompt embeddings ONCE — only the injected vector
# changes per request. Re-running _build_embeds per call costs ~150ms of
# synchronous CPU and was the real throughput bottleneck.
import numpy as _np
_base_embeds, _ = client._build_embeds(torch.zeros(3584), None)
_cfg = client.cfg
_ids = client.tokenizer.apply_chat_template(
    [{"role": "user", "content": _cfg.actor_prompt_template.format(
        injection_char=_cfg.injection_char)}],
    tokenize=True, add_generation_prompt=True, return_dict=False)
_inj_pos = next(p for p in range(1, len(_ids) - 1)
                if _ids[p] == _cfg.injection_token_id
                and _ids[p-1] == _cfg.injection_left_neighbor_id
                and _ids[p+1] == _cfg.injection_right_neighbor_id)

def fast_embeds(v: _np.ndarray) -> _np.ndarray:
    emb = _base_embeds.copy()
    nrm = float(_np.linalg.norm(v.astype(_np.float64)))
    emb[_inj_pos] = v * (_cfg.injection_scale / max(nrm, 1e-9))
    return emb
meta = json.loads((WORK / "meta2.json").read_text())
acts = {L: np.load(WORK / f"acts{L}.npy", mmap_mode="r") for L in [18, 22]}
OUT = WORK / "gen2_out"
OUT.mkdir(exist_ok=True)


def vec_for(job: str, i: int) -> np.ndarray:
    if job.startswith("expl"):
        return np.asarray(acts[int(job.removeprefix("expl"))][i], dtype=np.float32)
    return np.asarray(acts[22][i], dtype=np.float32) - np.asarray(acts[18][i], dtype=np.float32)


async def run_job(job: str, items: list[tuple[int, int]], temp: float) -> None:
    """items: list of (position_index, sample_index)."""
    shard_k, shard_n = 0, 1
    if os.environ.get("SHARD"):
        shard_k, shard_n = (int(x) for x in os.environ["SHARD"].split("/"))
    path = OUT / (f"{job}.s{shard_k}.jsonl" if shard_n > 1 else f"{job}.jsonl")
    done = set()
    for fn in glob.glob(str(OUT / f"{job}*.jsonl")):
        for line in open(fn):
            r = json.loads(line)
            done.add((r["i"], r["s"]))
    todo = [(i, s) for k, (i, s) in enumerate(items)
            if (i, s) not in done and (i * 17 + s) % shard_n == shard_k]
    if not todo:
        print(f"[gen2] {job}: complete", flush=True)
        return
    print(f"[gen2] {job}: {len(todo)} to go (of {len(items)})", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    f = path.open("a")
    lock = asyncio.Lock()
    t0, n_done = time.time(), 0

    async with httpx.AsyncClient(timeout=300.0) as http:
        async def one(i: int, s: int) -> None:
            nonlocal n_done
            embeds_np = fast_embeds(vec_for(job, i))
            body = orjson.dumps({"input_embeds": embeds_np,
                                 "sampling_params": {"temperature": temp,
                                                     "max_new_tokens": 220,
                                                     "skip_special_tokens": False}},
                                option=orjson.OPT_SERIALIZE_NUMPY)
            async with sem:
                for a in range(3):
                    try:
                        r = await http.post("http://localhost:30000/generate", content=body,
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
                    rate = n_done / (time.time() - t0)
                    print(f"[gen2] {job}: {n_done}/{len(todo)} ({rate:.1f}/s)", flush=True)

        await asyncio.gather(*(one(i, s) for i, s in todo))
    f.close()
    print(f"[gen2] {job}: done ({len(todo)/(time.time()-t0):.1f}/s)", flush=True)


def main() -> None:
    n = len(meta)
    train_idx = [i for i, m in enumerate(meta) if m["split"] == "train"]
    hold_idx = [i for i, m in enumerate(meta) if m["split"] == "holdout"]
    hybrid_idx = train_idx[:N_HYBRID]
    eval_idx = hold_idx[:N_EVAL]

    if len(sys.argv) > 1 and sys.argv[1] == "timing":
        asyncio.run(run_job("timing_test", [(i, 0) for i in range(min(1000, n))], 0.4))
        return

    asyncio.run(run_job("expl18", [(i, 0) for i in range(n)], 0.4))
    # expl20 cut for throughput: arm 0 (plain-L20) was the secondary baseline;
    # the primary content control is arm 0p (L18+L22), unaffected.
    asyncio.run(run_job("expl22", [(i, 0) for i in range(n)], 0.4))
    asyncio.run(run_job("diff1", [(i, 0) for i in range(n)], 0.4))
    asyncio.run(run_job("hybrid_reads", [(i, s) for i in hybrid_idx for s in range(4)], 1.0))
    asyncio.run(run_job("eval16", [(i, s) for i in eval_idx for s in range(16)], 1.0))
    print("[gen2] ALL DONE")


main()
