"""Better-diff-labels experiment (pre-Gate-C), pod job.

generate : 8 samples @ temp 1.0 per position for two injected vectors:
             dav8   — raw diff v22−v18
             resav8 — residualized diff (content directions projected out)
score    : AR-reconstruct all 2,880 texts → bo_recons.npy
klpatch  : causal judge. Rebuild the Gate A docs, verify activations match,
           then for each position patch candidate v̂22 = v18 + scaled diff
           prediction into hidden_states[22] and measure next-token KL vs
           the natural (unpatched) distribution at that position.
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
residuals = np.load(WORK / "residuals.npy")
N = acts.shape[0]
NSAMP = 8

if CMD == "generate":
    import httpx
    import orjson
    import torch
    from nla_inference import EXPLANATION_RE, NLAClient

    client = NLAClient(WORK / "models" / "av", sglang_url="http://localhost:30000")
    out_path = WORK / "bo_texts.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    jobs = {"dav8": acts[:, 22] - acts[:, 18], "resav8": residuals}

    async def gen(vecs):
        sem = asyncio.Semaphore(8)
        rows = [[None] * NSAMP for _ in range(N)]
        async with httpx.AsyncClient(timeout=300.0) as http:
            async def one(i, s):
                embeds_np, _ = client._build_embeds(torch.from_numpy(vecs[i].copy()), None)
                body = orjson.dumps({"input_embeds": embeds_np,
                                     "sampling_params": {"temperature": 1.0,
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
                rows[i][s] = {"explanation": m.group(1).strip() if m else None,
                              "raw": text[:400]}
            await asyncio.gather(*(one(i, s) for i in range(N) for s in range(NSAMP)))
        return rows

    for label, vecs in jobs.items():
        if label in results:
            continue
        print(f"[gen] {label}: {N}x{NSAMP}", flush=True)
        results[label] = asyncio.run(gen(vecs))
        out_path.write_text(json.dumps(results))
    print("[gen] done")

elif CMD == "score":
    from nla_inference import NLACritic

    critic = NLACritic(WORK / "models" / "ar", device="cuda")
    texts = json.loads((WORK / "bo_texts.json").read_text())
    conds = sorted(texts.keys())
    recons = np.zeros((len(conds), N, NSAMP, 3584), dtype=np.float16)
    ok = np.zeros((len(conds), N, NSAMP), dtype=bool)
    for ci, cond in enumerate(conds):
        for i in range(N):
            for s in range(NSAMP):
                r = texts[cond][i][s]
                t = r["explanation"] if r["explanation"] is not None else r["raw"]
                if t:
                    recons[ci, i, s] = critic.reconstruct(t).numpy().astype(np.float16)
                    ok[ci, i, s] = True
        print(f"[score] {cond} done", flush=True)
    np.save(WORK / "bo_recons.npy", recons)
    np.save(WORK / "bo_ok.npy", ok)
    (WORK / "bo_conds.json").write_text(json.dumps(conds))
    print("[score] saved")

elif CMD == "klpatch":
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = json.loads((WORK / "meta.json").read_text())
    # candidate diff predictions (all rescaled to true diff norm at eval time)
    cand_files = json.loads((WORK / "kl_candidates.json").read_text())
    cands = {name: np.load(WORK / fn).astype(np.float32) for name, fn in cand_files.items()}

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="cuda").eval()
    special_ids = set(tok.all_special_ids)
    rng = np.random.default_rng(0)

    # rebuild the same docs/positions as gate_a extract (seed 0, same filters)
    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
    doc_tokens = {}
    n_done = 0
    pos_by_doc = {}
    for doc_idx, doc in enumerate(ds):
        ids = tok(doc["text"], add_special_tokens=False, truncation=True, max_length=512)["input_ids"]
        candidates = [i for i, t in enumerate(ids) if i >= 50 and t not in special_ids]
        if len(candidates) < 13:
            continue
        positions = sorted(rng.choice(candidates, 3, replace=False).tolist())
        doc_tokens[doc_idx] = ids
        pos_by_doc[doc_idx] = positions
        n_done += 1
        if n_done >= 60:
            break

    PATCH = {"vec": None}  # [K, d] bf16, set per batch

    def hook(mod, inp, out):
        # decoder layers return a tuple in older transformers, a bare tensor in newer
        h = out[0] if isinstance(out, tuple) else out
        h[:, PATCH["pos"]] = PATCH["vec"]
        return out

    layer = model.model.layers[21]  # output == hidden_states[22]

    results = {name: [] for name in cands}
    results["zero_diff"] = []
    results["_sanity_true_diff"] = []
    n_verified = n_skipped = 0
    for irow, m in enumerate(meta):
        ids = doc_tokens.get(m["doc_idx"])
        if ids is None or m["pos"] not in pos_by_doc.get(m["doc_idx"], []):
            n_skipped += 1
            continue
        p = m["pos"]
        x = torch.tensor([ids], device="cuda")
        with torch.inference_mode():
            ref_out = model(x, output_hidden_states=True)
        v20_live = ref_out.hidden_states[20][0, p].float().cpu().numpy()
        gold20 = acts[irow, 20]
        cos20 = float(v20_live @ gold20 / (np.linalg.norm(v20_live) * np.linalg.norm(gold20)))
        if cos20 < 0.999:
            n_skipped += 1
            continue
        n_verified += 1
        ref_logits = ref_out.logits[0, p].float()
        ref_logp = F.log_softmax(ref_logits, dim=-1)

        d_true = acts[irow, 22] - acts[irow, 18]
        dn = np.linalg.norm(d_true)
        names = ["zero_diff", "_sanity_true_diff"] + list(cands.keys())
        vecs = [np.zeros(3584, np.float32), d_true]
        for name in cands:
            g = cands[name][irow]
            gn = np.linalg.norm(g)
            vecs.append(g / max(gn, 1e-9) * dn if gn > 0 else np.zeros(3584, np.float32))
        v18 = acts[irow, 18]
        patch = torch.tensor(np.stack([v18 + v for v in vecs]), dtype=torch.bfloat16, device="cuda")

        PATCH["vec"], PATCH["pos"] = patch, p
        h = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = model(x.repeat(len(vecs), 1))
        finally:
            h.remove()
        logp = F.log_softmax(out.logits[:, p].float(), dim=-1)
        kls = (ref_logp.exp() * (ref_logp - logp)).sum(-1).cpu().numpy()
        for k, name in enumerate(names):
            results[name].append(float(kls[k]))
        if n_verified % 30 == 0:
            print(f"[kl] {n_verified} verified, {n_skipped} skipped", flush=True)

    print(f"[kl] verified {n_verified}, skipped {n_skipped}")
    (WORK / "kl_results.json").write_text(json.dumps(results))
    for name, v in results.items():
        a = np.array(v)
        print(f"  {name:24} KL mean {a.mean():.3f}  median {np.median(a):.3f}")
    print("[kl] saved")
