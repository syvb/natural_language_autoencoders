"""Gate A: cross-layer transfer of the released Qwen-7B L20 NLA.

Subcommands (run in order on the pod):
  extract   — forward Qwen2.5-7B-Instruct over fineweb docs, save activations
              at ALL hidden_states indices (0..28) for sampled positions
  generate  — for each eval layer, inject v_L into the L20 AV via SGLang,
              collect explanations
  score     — released L20 AR critic: cos(AR(text), v_L) per layer, plus
              controls (shuffled floor, cos vs v_20, raw cos(v_L, v_20),
              CJK fraction, tag success rate)

Mirrors stage0 conventions: plain tokenization (no chat template),
positions >= 50, raw (unnormalized) vectors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).parent
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
AV_DIR = WORK / "models" / "av"
AR_DIR = WORK / "models" / "ar"
TRAINED_LAYER = 20
N_HIDDEN = 29  # embeddings + 28 blocks
EVAL_LAYERS = [0, 4, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 28]
N_DOCS = 60
POS_PER_DOC = 3
MIN_POSITION = 50
MAX_TOKENS = 512
SEED = 0
CJK_RE = re.compile(r"[぀-ヿ一-鿿]")


def cmd_extract() -> None:
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    special_ids = set(tok.all_special_ids)
    rng = np.random.default_rng(SEED)

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                      split="train", streaming=True)
    acts, meta = [], []
    n_done = 0
    for doc_idx, doc in enumerate(ds):
        ids = tok(doc["text"], add_special_tokens=False,
                  truncation=True, max_length=MAX_TOKENS)["input_ids"]
        candidates = [i for i, t in enumerate(ids)
                      if i >= MIN_POSITION and t not in special_ids]
        if len(candidates) < POS_PER_DOC + 10:
            continue
        positions = sorted(rng.choice(candidates, POS_PER_DOC, replace=False).tolist())
        with torch.inference_mode():
            out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)[:, 0]  # [29, T, d]
        for pos in positions:
            acts.append(hs[:, pos].float().cpu().numpy())  # [29, d] raw
            meta.append({
                "doc_idx": doc_idx, "pos": pos,
                "token_at_pos": tok.decode([ids[pos]]),
                "context_tail": tok.decode(ids[max(0, pos - 30):pos + 1],
                                           skip_special_tokens=True),
            })
        n_done += 1
        if n_done % 10 == 0:
            print(f"[extract] {n_done}/{N_DOCS} docs", flush=True)
        if n_done >= N_DOCS:
            break

    arr = np.stack(acts)  # [N, 29, d]
    np.save(WORK / "acts.npy", arr)
    (WORK / "meta.json").write_text(json.dumps(meta))
    norms = np.linalg.norm(arr, axis=-1).mean(axis=0)
    print(f"[extract] saved {arr.shape}; mean L2 norm per layer:")
    for layer in EVAL_LAYERS:
        print(f"  L{layer}: {norms[layer]:.1f}")


def cmd_generate() -> None:
    import httpx
    import orjson

    sys.path.insert(0, str(WORK))
    from nla_inference import EXPLANATION_RE, NLAClient

    client = NLAClient(AV_DIR, sglang_url="http://localhost:30000")
    acts = np.load(WORK / "acts.npy")  # [N, 29, d]
    n = acts.shape[0]

    out_path = WORK / "texts.json"
    results: dict[str, list] = (
        json.loads(out_path.read_text()) if out_path.exists() else {}
    )

    async def gen_layer(vecs: np.ndarray, label: str) -> list[dict]:
        sem = asyncio.Semaphore(8)
        rows: list[dict | None] = [None] * len(vecs)

        async with httpx.AsyncClient(timeout=300.0) as http:
            async def one(i: int) -> None:
                import torch
                embeds_np, _ = client._build_embeds(
                    torch.from_numpy(vecs[i].copy()), None)
                body = orjson.dumps(
                    {"input_embeds": embeds_np,
                     "sampling_params": {"temperature": 0.7,
                                         "max_new_tokens": 220,
                                         "skip_special_tokens": False}},
                    option=orjson.OPT_SERIALIZE_NUMPY)
                async with sem:
                    for attempt in range(3):
                        try:
                            r = await http.post(
                                "http://localhost:30000/generate", content=body,
                                headers={"Content-Type": "application/json"})
                            r.raise_for_status()
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise
                            print(f"[gen] retry {label}#{i}: {e}", flush=True)
                            await asyncio.sleep(5)
                out = r.json()
                text = (out[0] if isinstance(out, list) else out)["text"]
                m = EXPLANATION_RE.search(text)
                rows[i] = {"explanation": m.group(1).strip() if m else None,
                           "raw": text[:600]}

            await asyncio.gather(*(one(i) for i in range(len(vecs))))
        return rows  # type: ignore[return-value]

    jobs = [(f"L{layer}", layer) for layer in EVAL_LAYERS]
    jobs.append((f"L{TRAINED_LAYER}_rep", TRAINED_LAYER))  # gen-noise reference
    for label, layer in jobs:
        if label in results:
            print(f"[gen] {label} cached, skip", flush=True)
            continue
        print(f"[gen] layer {label}: {n} vectors", flush=True)
        results[label] = asyncio.run(gen_layer(acts[:, layer], label))
        out_path.write_text(json.dumps(results))
    print("[gen] done")


def cmd_score() -> None:
    import torch

    sys.path.insert(0, str(WORK))
    from nla_inference import NLACritic

    critic = NLACritic(AR_DIR, device="cuda")
    acts = np.load(WORK / "acts.npy")
    texts = json.loads((WORK / "texts.json").read_text())
    n = acts.shape[0]
    rng = np.random.default_rng(1)
    perm = rng.permutation(n)
    # derangement-ish: ensure no fixed points
    for i in np.where(perm == np.arange(n))[0]:
        j = (i + 1) % n
        perm[i], perm[j] = perm[j], perm[i]

    v20 = acts[:, TRAINED_LAYER]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    results = {}
    for label, rows in texts.items():
        layer = int(label.removeprefix("L").removesuffix("_rep"))
        vl = acts[:, layer]
        recon = []
        for r in rows:
            text = r["explanation"] if r["explanation"] is not None else r["raw"]
            recon.append(critic.reconstruct(text).numpy())
        recon = np.stack(recon)
        entry = {
            "layer": layer,
            "cos_same_layer": float(np.mean([cos(recon[i], vl[i]) for i in range(n)])),
            "cos_vs_L20": float(np.mean([cos(recon[i], v20[i]) for i in range(n)])),
            "cos_shuffled": float(np.mean([cos(recon[i], vl[perm[i]]) for i in range(n)])),
            "vec_cos_to_L20": float(np.mean([cos(vl[i], v20[i]) for i in range(n)])),
            "tag_ok_frac": float(np.mean([r["explanation"] is not None for r in rows])),
            "cjk_frac": float(np.mean([bool(CJK_RE.search(r["raw"])) for r in rows])),
            "mean_norm": float(np.linalg.norm(vl, axis=1).mean()),
        }
        results[label] = entry
        print(f"[score] {label}: {entry}", flush=True)

    (WORK / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'label':>9} {'cos(AR,v_L)':>11} {'cos(AR,v20)':>11} "
          f"{'shuffled':>9} {'cos(vL,v20)':>11} {'tag%':>6} {'CJK%':>6} {'norm':>8}")
    for label in sorted(results, key=lambda k: (results[k]["layer"], k)):
        e = results[label]
        print(f"{label:>9} {e['cos_same_layer']:>11.3f} {e['cos_vs_L20']:>11.3f} "
              f"{e['cos_shuffled']:>9.3f} {e['vec_cos_to_L20']:>11.3f} "
              f"{e['tag_ok_frac']:>6.2f} {e['cjk_frac']:>6.2f} {e['mean_norm']:>8.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "generate", "score"])
    args = ap.parse_args()
    {"extract": cmd_extract, "generate": cmd_generate, "score": cmd_score}[args.cmd]()
