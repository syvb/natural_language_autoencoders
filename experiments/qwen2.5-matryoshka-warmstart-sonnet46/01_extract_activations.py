"""Stage A — extract Qwen2.5-7B layer-20 activations for the matryoshka warm-start data.

The source dataset `ceselder/nla-matryoshka-warmstart-sonnet46` is TEXT-ONLY: Claude
Sonnet-4.6 explanations keyed by custom_id = `{av,ar}-{doc_id}-{position}`, with an
`input_text` column but NO activation vectors. NLA training needs the vectors.

Key insight (verified: 100% of sampled rows): `input_text` is bit-exactly
`token_ids[:position]`, i.e. it retokenizes to exactly `position` tokens. So the
activation we need is Qwen2.5-7B-Instruct's layer-20 hidden state at the LAST token of
`input_text`. No source corpus required (the original Ultra-FineWeb slice has drifted /
is no longer hosted) — this is fully reproducible from the dataset's own text.

Output: base_av.parquet + base_ar.parquet, each with
  api_explanation, activation_vector (FixedSizeList f32, raw norm="none"),
  n_raw_tokens, activation_layer, doc_id, detokenized_text_truncated, custom_id

Run on a single GPU box (an H200/H100):
  MAT downloaded from HF; MODEL = Qwen/Qwen2.5-7B-Instruct snapshot.
  python 01_extract_activations.py
"""
import sys, time, traceback, os
import numpy as np, torch
import pyarrow as pa, pyarrow.parquet as pq

def log(*a): print(*a, flush=True)

def main():
    from transformers import AutoModelForCausalLM
    from nla.datagen._common import load_tokenizer
    from nla.arch_adapters import resolve_decoder_layers, resolve_text_config
    import torch.nn as nn

    MODEL = os.environ.get("BASE_MODEL", "/workspace/models/qwen2.5-7b-instruct")
    MAT = os.environ.get("MATRYOSHKA_PARQUET",
                         "/workspace/data/matryoshka/data/train-00000-of-00001.parquet")
    LAYER = 20; BATCH = 96
    os.makedirs("/workspace/out", exist_ok=True)

    log("load matryoshka")
    mt = pq.read_table(MAT, columns=["custom_id", "input_text", "analysis", "status"])
    cid = mt.column("custom_id").to_pylist(); itx = mt.column("input_text").to_pylist()
    ana = mt.column("analysis").to_pylist(); sts = mt.column("status").to_pylist()
    rows = []
    for c, t, a, s in zip(cid, itx, ana, sts):
        if s != "succeeded": continue
        p = c.split("-"); rows.append((p[0], int(p[1]), int(p[2]), c, t, a))
    log("rows:", len(rows))

    tok = load_tokenizer(MODEL); tok.padding_side = "right"; tok.truncation_side = "right"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    log("tokenize (batched)")
    t0 = time.time(); all_ids = []; texts = [r[4] for r in rows]; CH = 20000
    for i in range(0, len(texts), CH):
        enc = tok(texts[i:i+CH], add_special_tokens=True, truncation=True, max_length=4096)["input_ids"]
        all_ids.extend(enc)
        log(f"  tokenized {min(i+CH,len(texts))}/{len(texts)}  {time.time()-t0:.0f}s")
    lens = np.array([len(x) for x in all_ids])
    pos = np.array([r[2] for r in rows])
    # Sanity: input_text must retokenize to exactly `position` tokens.
    log(f"  len==pos match: {(lens==pos).mean()*100:.2f}%  (min/mean/max len {lens.min()}/{lens.mean():.0f}/{lens.max()})")

    log("load model + truncate to layer", LAYER)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    D = resolve_text_config(model.config).hidden_size; assert D == 3584, D
    layers = resolve_decoder_layers(model)
    model.model.layers = nn.ModuleList(list(layers[:LAYER+1]))  # skip layers >20 + lm_head
    cap = {}
    layers[LAYER].register_forward_hook(lambda m, i, o: cap.__setitem__('h', o[0] if isinstance(o, tuple) else o))

    order = np.argsort(lens, kind="stable")  # length-sort to minimize padding
    acts = np.empty((len(rows), D), dtype=np.float32)
    dev = model.get_input_embeddings().weight.device; pad_id = tok.pad_token_id
    t0 = time.time(); done = 0
    with torch.no_grad():
        for bs in range(0, len(order), BATCH):
            bidx = order[bs:bs+BATCH]; seqs = [all_ids[j] for j in bidx]
            m = max(len(s) for s in seqs)
            inp = np.full((len(seqs), m), pad_id, dtype=np.int64); att = np.zeros((len(seqs), m), dtype=np.int64)
            for k, s in enumerate(seqs): inp[k, :len(s)] = s; att[k, :len(s)] = 1
            inp = torch.from_numpy(inp).to(dev); att = torch.from_numpy(att).to(dev)
            cap.clear(); model.model(input_ids=inp, attention_mask=att, use_cache=False)
            h = cap['h']
            for k, j in enumerate(bidx): acts[j] = h[k, len(seqs[k])-1].float().cpu().numpy()
            done += len(seqs)
            if (bs//BATCH) % 50 == 0:
                el = time.time()-t0; rate = done/max(el, 1e-9)
                log(f"  fwd {done}/{len(rows)}  {rate:.0f}/s  eta {(len(rows)-done)/max(rate,1e-9)/60:.1f}min")
    log(f"extraction done in {(time.time()-t0)/60:.1f}min")

    def write_base(split, path):
        sel = [i for i, r in enumerate(rows) if r[0] == split]; n = len(sel)
        av = np.ascontiguousarray(acts[sel])
        av_arr = pa.FixedSizeListArray.from_arrays(pa.array(av.reshape(-1), type=pa.float32()), D)
        tbl = pa.table({
            "api_explanation": pa.array([rows[i][5] for i in sel], pa.string()),
            "activation_vector": av_arr,
            "n_raw_tokens": pa.array([rows[i][2] for i in sel], pa.int64()),
            "activation_layer": pa.array([LAYER]*n, pa.int64()),
            "doc_id": pa.array([f"openbmb/Ultra-FineWeb:en:{rows[i][1]}" for i in sel], pa.string()),
            "detokenized_text_truncated": pa.array([rows[i][4] for i in sel], pa.string()),
            "custom_id": pa.array([rows[i][3] for i in sel], pa.string()),
        })
        pq.write_table(tbl, path); log(f"wrote {split}: {n} rows -> {path}"); return n
    n_av = write_base("av", "/workspace/out/base_av.parquet")
    n_ar = write_base("ar", "/workspace/out/base_ar.parquet")
    log(f"DONE av={n_av} ar={n_ar}")

try:
    main()
except Exception:
    traceback.print_exc(); sys.stdout.flush(); sys.exit(1)
