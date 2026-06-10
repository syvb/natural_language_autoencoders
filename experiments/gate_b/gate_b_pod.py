"""Gate B pod job, two phases.

generate: inject the raw DIFF vector (v_Y - v_X, rescaled to injection_scale
          like any activation) directly into the L20 AV via SGLang and collect
          its explanation — a fully self-labeled, API-free diff description.
score:    AR-reconstruct (a) the v2 Claude labels and (b) the diff-AV
          explanations; save vectors for local analysis.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
sys.path.insert(0, str(WORK))

CMD = sys.argv[1]
acts = np.load(WORK / "acts.npy")
PAIRS = {"diffav_19_21": (19, 21), "diffav_18_22": (18, 22)}

if CMD == "generate":
    import httpx
    import orjson
    import torch
    from nla_inference import EXPLANATION_RE, NLAClient

    client = NLAClient(WORK / "models" / "av", sglang_url="http://localhost:30000")
    n = acts.shape[0]
    out_path = WORK / "diffav_texts.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    async def gen(vecs, label):
        sem = asyncio.Semaphore(8)
        rows = [None] * len(vecs)
        async with httpx.AsyncClient(timeout=300.0) as http:
            async def one(i):
                embeds_np, _ = client._build_embeds(torch.from_numpy(vecs[i].copy()), None)
                body = orjson.dumps({"input_embeds": embeds_np,
                                     "sampling_params": {"temperature": 0.7,
                                                         "max_new_tokens": 220,
                                                         "skip_special_tokens": False}},
                                    option=orjson.OPT_SERIALIZE_NUMPY)
                async with sem:
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
                rows[i] = {"explanation": m.group(1).strip() if m else None,
                           "raw": text[:600]}
            await asyncio.gather(*(one(i) for i in range(len(vecs))))
        return rows

    for label, (lx, ly) in PAIRS.items():
        if label in results:
            continue
        print(f"[gen] {label}", flush=True)
        results[label] = asyncio.run(gen(acts[:, ly] - acts[:, lx], label))
        out_path.write_text(json.dumps(results))
    print("[gen] done")

elif CMD == "score":
    from nla_inference import NLACritic

    critic = NLACritic(WORK / "models" / "ar", device="cuda")
    jobs = {}
    v2 = json.loads((WORK / "gate_b_labels_v2.json").read_text())
    for cond, rows in v2.items():
        jobs[cond] = [r.get("difference") for r in rows]
    dav = json.loads((WORK / "diffav_texts.json").read_text())
    for cond, rows in dav.items():
        jobs[cond] = [r["explanation"] if r["explanation"] is not None else r["raw"]
                      for r in rows]

    conds = sorted(jobs.keys())
    n = len(jobs[conds[0]])
    recons = np.zeros((len(conds), n, 3584), dtype=np.float16)
    ok = np.zeros((len(conds), n), dtype=bool)
    for ci, cond in enumerate(conds):
        for i, text in enumerate(jobs[cond]):
            if text:
                recons[ci, i] = critic.reconstruct(text).numpy().astype(np.float16)
                ok[ci, i] = True
        print(f"[score] {cond}: {int(ok[ci].sum())}/{n}", flush=True)
    np.save(WORK / "round2_recons.npy", recons)
    np.save(WORK / "round2_ok.npy", ok)
    (WORK / "round2_conds.json").write_text(json.dumps(conds))
    print("[score] saved")
