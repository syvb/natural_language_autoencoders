"""Gate G Stage 0: AV verbalizes h_22 (the 'after' state).

Inject acts22[i] into the frozen L20 AV (in-distribution: Gate A showed the
L20 AV transfers L14-L24), generate one description per position. These
descriptions become the 'text' arm input for the conditioned critic, which
also sees h_20 and must reconstruct h_22 — so the text only needs to carry
what h_20 lacks.

One greedy-ish sample per position over ALL 15k gate_c positions.
SHARD=k/n for multi-process. Resumes from existing output.
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


# acts live in gate_c (aligned to gate_c/meta2.json). Default: verbalize h_22.
# SRC_ACTS=acts20.npy + OUT_DIR=av_h20_out gives the s20 control (verbalize h_20).
GC = WORK.parent / "gate_c"
H22 = np.load(GC / os.environ.get("SRC_ACTS", "acts22.npy"), mmap_mode="r")
N = H22.shape[0]
OUT = WORK / os.environ.get("OUT_DIR", "av_out")
OUT.mkdir(exist_ok=True)


async def run():
    shard_k, shard_n = 0, 1
    if os.environ.get("SHARD"):
        shard_k, shard_n = (int(x) for x in os.environ["SHARD"].split("/"))
    path = OUT / (f"av.s{shard_k}.jsonl" if shard_n > 1 else "av.jsonl")
    done = set()
    for fn in glob.glob(str(OUT / "av*.jsonl")):
        for line in open(fn):
            done.add(json.loads(line)["i"])
    todo = [i for i in range(N) if i not in done and i % shard_n == shard_k]
    if not todo:
        print("[gen_av] complete", flush=True)
        return
    print(f"[gen_av] {len(todo)} to go", flush=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    f = path.open("a")
    lock = asyncio.Lock()
    t0, n_done, n_cjk = time.time(), 0, 0
    async with httpx.AsyncClient(timeout=300.0) as http:
        async def one(i):
            nonlocal n_done, n_cjk
            async with sem:
                body = orjson.dumps(
                    {"input_embeds": fast_embeds(np.asarray(H22[i], dtype=np.float32)),
                     "sampling_params": {"temperature": 0.3, "max_new_tokens": 200,
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
            exp = m.group(1).strip() if m else None
            async with lock:
                f.write(json.dumps({"i": i, "explanation": exp, "raw": text[:300]}) + "\n")
                f.flush()
                n_done += 1
                n_cjk += bool(CJK.search(exp or text or ""))
                if n_done % 2000 == 0:
                    print(f"[gen_av] {n_done}/{len(todo)} ({n_done/(time.time()-t0):.1f}/s) "
                          f"cjk={n_cjk}", flush=True)
        await asyncio.gather(*(one(i) for i in todo))
    f.close()
    print(f"[gen_av] done; CJK {n_cjk}/{n_done}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        async def smoke():
            async with httpx.AsyncClient(timeout=300.0) as http:
                for i in range(5):
                    body = orjson.dumps(
                        {"input_embeds": fast_embeds(np.asarray(H22[i], dtype=np.float32)),
                         "sampling_params": {"temperature": 0.3, "max_new_tokens": 200,
                                             "skip_special_tokens": False}},
                        option=orjson.OPT_SERIALIZE_NUMPY)
                    r = await http.post("http://localhost:30000/generate", content=body,
                                        headers={"Content-Type": "application/json"})
                    out = r.json()
                    text = (out[0] if isinstance(out, list) else out)["text"]
                    m = EXPLANATION_RE.search(text)
                    print(f"[{i}] {m.group(1).strip() if m else text[:200]!r}")
        asyncio.run(smoke())
    else:
        asyncio.run(run())
