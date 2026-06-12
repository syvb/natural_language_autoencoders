"""Gate E0.2: pairwise extraction + divergence selection + continuations.

Two passes (v2 — the naive loop projected 6.5h unbatched, uncheckpointed):
  deltas : per pair, two forwards; KL over aligned post-edit positions;
           keep top-3 positions with KL >= KL_FLOOR. Chunked + resumable
           (chunks of 500 pairs -> chunk files, merged at the end).
  conts  : batched continuation sampling for consequence labels —
           16 prefixes x 2 samples per generate call, left-padded.
           Resumable jsonl.

Run: python extract_pairs.py deltas && python extract_pairs.py conts
Raw fp32 everywhere (norm="none").
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK = Path(__file__).parent
LAYERS = [16, 20, 24]
KL_FLOOR = 0.05
MAX_POS_PER_PAIR = 3
HOLDOUT_FRAC = 0.2
SEED = 0
CHUNK = 500

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    device_map="cuda").eval()
torch.set_grad_enabled(False)
pairs = json.loads((WORK / "pairs_e.json").read_text())
CH = WORK / "chunks"
CH.mkdir(exist_ok=True)


def run_deltas():
    rng = np.random.default_rng(SEED)
    doc_split = {}
    # doc splits must be deterministic regardless of chunk resume: derive
    # from a per-doc keyed RNG (repo invariant), not iteration order
    def split_of(doc_idx):
        return ("holdout"
                if np.random.default_rng((SEED, doc_idx)).random() < HOLDOUT_FRAC
                else "train")

    for c0 in range(0, len(pairs), CHUNK):
        cpath = CH / f"chunk_{c0}.npz"
        if cpath.exists():
            continue
        dl = {L: [] for L in LAYERS}
        v20 = []
        meta = []
        for pr in pairs[c0:c0 + CHUNK]:
            ids_o = tok(pr["text"], add_special_tokens=False)["input_ids"]
            ids_e = tok(pr["text_edited"], add_special_tokens=False)["input_ids"]
            if len(ids_o) != len(ids_e):
                continue
            e_end = pr["edit_end_tok"]
            if ids_o[e_end:] != ids_e[e_end:]:
                continue
            h_o = model(torch.tensor([ids_o], device="cuda"),
                        output_hidden_states=True)
            h_e = model(torch.tensor([ids_e], device="cuda"),
                        output_hidden_states=True)
            lp_o = F.log_softmax(h_o.logits[0, e_end:-1].float(), dim=-1)
            lp_e = F.log_softmax(h_e.logits[0, e_end:-1].float(), dim=-1)
            kl = (lp_o.exp() * (lp_o - lp_e)).sum(-1).cpu().numpy()
            cand = [int(c) for c in np.argsort(-kl)[:MAX_POS_PER_PAIR]
                    if kl[c] >= KL_FLOOR]
            for c in sorted(cand):
                p = e_end + c
                for L in LAYERS:
                    dl[L].append((h_o.hidden_states[L][0, p].float()
                                  - h_e.hidden_states[L][0, p].float()).cpu().numpy())
                v20.append(h_o.hidden_states[20][0, p].float().cpu().numpy())
                meta.append({"pair_id": pr["pair_id"], "doc_idx": pr["doc_idx"],
                             "type": pr["type"], "source": pr["source"],
                             "target": pr["target"], "pos": p,
                             "dist_from_edit": int(p - e_end),
                             "kl": float(kl[c]),
                             "split": split_of(pr["doc_idx"]),
                             "context_tail": tok.decode(ids_o[max(0, p - 60):p + 1])})
        np.savez_compressed(cpath,
                            **{f"d{L}": np.stack(dl[L]) if dl[L] else np.zeros((0, 3584), np.float32)
                               for L in LAYERS},
                            v20=np.stack(v20) if v20 else np.zeros((0, 3584), np.float32))
        (CH / f"meta_{c0}.json").write_text(json.dumps(meta))
        print(f"[deltas] chunk {c0}: {len(meta)} positions", flush=True)

    # merge
    metas = []
    arrs = {f"d{L}": [] for L in LAYERS}
    arrs["v20"] = []
    for c0 in range(0, len(pairs), CHUNK):
        z = np.load(CH / f"chunk_{c0}.npz")
        for k in arrs:
            arrs[k].append(z[k])
        metas += json.loads((CH / f"meta_{c0}.json").read_text())
    for L in LAYERS:
        np.save(WORK / f"delta{L}.npy", np.concatenate(arrs[f"d{L}"]))
    np.save(WORK / "v20x.npy", np.concatenate(arrs["v20"]))
    (WORK / "meta_e.json").write_text(json.dumps(metas))
    dn = np.linalg.norm(np.concatenate(arrs["d20"]), axis=1)
    n_h = sum(m["split"] == "holdout" for m in metas)
    print(f"[deltas] MERGED {len(metas)} positions ({n_h} holdout)")
    print(f"[deltas] delta20 norms: p10 {np.percentile(dn,10):.2f} "
          f"median {np.median(dn):.2f} p90 {np.percentile(dn,90):.2f}")


def run_conts(batch_prefixes=16, n_samp=2):
    meta = json.loads((WORK / "meta_e.json").read_text())
    by_pair = {p["pair_id"]: p for p in pairs}
    out_path = WORK / "continuations.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            done.add(json.loads(line)["i"])
    todo = [i for i in range(len(meta)) if i not in done]
    print(f"[conts] {len(todo)} positions to go", flush=True)
    f = out_path.open("a")
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    for b0 in range(0, len(todo), batch_prefixes):
        chunk = todo[b0:b0 + batch_prefixes]
        prompts = []
        for i in chunk:
            m = meta[i]
            pr = by_pair[m["pair_id"]]
            for text in (pr["text"], pr["text_edited"]):
                ids = tok(text, add_special_tokens=False)["input_ids"]
                prompts.append(tok.decode(ids[:m["pos"] + 1]))
        enc = tok(prompts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        outs = model.generate(**enc, max_new_tokens=24, do_sample=True,
                              temperature=0.7, num_return_sequences=n_samp,
                              pad_token_id=tok.pad_token_id)
        plen = enc["input_ids"].shape[1]
        texts = [tok.decode(o[plen:], skip_special_tokens=True) for o in outs]
        # layout: prompts expand to n_samp consecutive rows each
        for k, i in enumerate(chunk):
            base = k * 2 * n_samp
            f.write(json.dumps({"i": i,
                                "cont_x": texts[base:base + n_samp],
                                "cont_xp": texts[base + n_samp:base + 2 * n_samp]}) + "\n")
        f.flush()
        if b0 % (batch_prefixes * 20) == 0:
            print(f"[conts] {b0}/{len(todo)}", flush=True)
    f.close()
    print("[conts] DONE")


if __name__ == "__main__":
    {"deltas": run_deltas, "conts": run_conts}[sys.argv[1]]()
