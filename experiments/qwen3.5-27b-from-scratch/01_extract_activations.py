"""Step 1 — extract layer-$LAYER_INDEX activations for the warm-start data.

The source dataset ($MATRYOSHKA_DATASET) is TEXT-ONLY: Claude Sonnet-4.6
explanations keyed by custom_id = `{av,ar}-{doc_id}-{position}`, with an
`input_text` column but NO activation vectors.

The explanations are a function of the TEXT PREFIX only (the stage-2 prompt
never saw an activation), so they transfer to any base model: retokenize
`input_text` with the target model's tokenizer and extract the hidden state at
the LAST token of the prefix — the model state after reading exactly those
characters, which is what the explanation describes. The original `position`
was a Qwen2.5 token count; with a different tokenizer the count differs and
the logged len==pos match rate will be ~0. That is EXPECTED and fine —
`n_raw_tokens` is written from the NEW tokenizer's length.

Also measures the L2-norm distribution of the extracted vectors and prints a
suggested INJECTION_SCALE (a round number a bit above the mean norm — the
kitft convention). Copy that into your config before SFT; never guess it.

Output ($WORK/out): base_av.parquet, base_ar.parquet, norm_stats.json
Single GPU; the forward runs only layers 0..LAYER_INDEX. For the 27B that is
~18B params in bf16 (~36 GB) — any 48 GB+ GPU works, H200/B200 comfortable.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config


def log(*a):
    print(*a, flush=True)


def load_base_model(path, dtype):
    """AutoModelForCausalLM first; multimodal checkpoints (e.g. Qwen3.5's
    vision-language wrapper) may need the image-text auto class instead."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, device_map="cuda"
        )
    except Exception as e:  # noqa: BLE001 - fall through to the VLM auto class
        log(f"AutoModelForCausalLM failed ({type(e).__name__}: {e}); "
            f"trying AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            path, torch_dtype=dtype, device_map="cuda"
        )


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import torch.nn as nn

    from nla.arch_adapters import resolve_text_config, resolve_text_model
    from nla.datagen._common import load_tokenizer

    _config.load()
    WORK = _config.env("WORK")
    MODEL = os.environ.get("BASE_MODEL_DIR", f"{WORK}/models/base")
    LAYER = int(_config.env("LAYER_INDEX"))
    D_EXPECT = int(_config.env("D_MODEL"))
    BATCH = int(_config.env("EXTRACT_BATCH"))
    MAX_LEN = int(_config.env("EXTRACT_MAX_LEN"))
    LIMIT = int(os.environ.get("EXTRACT_LIMIT", "0"))  # 0 = all rows
    MAT = os.environ.get(
        "MATRYOSHKA_PARQUET", f"{WORK}/data/matryoshka/data/train-00000-of-00001.parquet"
    )
    OUT = f"{WORK}/out"
    os.makedirs(OUT, exist_ok=True)

    log("load matryoshka:", MAT)
    mt = pq.read_table(MAT, columns=["custom_id", "input_text", "analysis", "status"])
    rows = []
    for c, t, a, s in zip(
        mt.column("custom_id").to_pylist(), mt.column("input_text").to_pylist(),
        mt.column("analysis").to_pylist(), mt.column("status").to_pylist(),
    ):
        if s != "succeeded":
            continue
        p = c.split("-")
        rows.append((p[0], int(p[1]), int(p[2]), c, t, a))
    if LIMIT:
        # Interleaved slice so a smoke run still sees both splits + many docs.
        step = max(1, len(rows) // LIMIT)
        rows = rows[::step][:LIMIT]
    log("rows:", len(rows))

    tok = load_tokenizer(MODEL)
    tok.padding_side = "right"
    tok.truncation_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    log("tokenize (batched)")
    t0 = time.time()
    all_ids = []
    texts = [r[4] for r in rows]
    CH = 20000
    for i in range(0, len(texts), CH):
        enc = tok(texts[i : i + CH], add_special_tokens=True, truncation=True,
                  max_length=MAX_LEN)["input_ids"]
        all_ids.extend(enc)
        log(f"  tokenized {min(i + CH, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s")
    lens = np.array([len(x) for x in all_ids])
    pos = np.array([r[2] for r in rows])
    # Same tokenizer as the source (Qwen2.5) -> ~100%. Different tokenizer ->
    # ~0%, expected: the prefix TEXT is what matters, not the token count.
    log(f"  len==source_pos match: {(lens == pos).mean() * 100:.2f}%  "
        f"(min/mean/max len {lens.min()}/{lens.mean():.0f}/{lens.max()})")

    log("load model + truncate to layer", LAYER)
    model = load_base_model(MODEL, torch.bfloat16).eval()
    D = resolve_text_config(model.config).hidden_size
    assert D == D_EXPECT, f"hidden_size={D} != configured D_MODEL={D_EXPECT}"
    lm = resolve_text_model(model)
    # Find the module that OWNS the decoder ModuleList so the truncation can be
    # assigned back (model.model.layers for Qwen/Llama; wrapper-nested for VLMs).
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        stack = lm.model
    elif hasattr(lm, "layers"):
        stack = lm
    else:
        raise AssertionError(
            f"cannot locate decoder layers under {type(lm).__name__}; "
            f"extend this resolver for the architecture"
        )
    layers = stack.layers
    assert len(layers) > LAYER, f"model has {len(layers)} layers, need > {LAYER}"
    cap = {}
    layers[LAYER].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o)
    )
    stack.layers = nn.ModuleList(list(layers[: LAYER + 1]))

    order = np.argsort(lens, kind="stable")  # length-sort to minimize padding
    acts = np.empty((len(rows), D), dtype=np.float32)
    dev = model.get_input_embeddings().weight.device
    pad_id = tok.pad_token_id
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for bs in range(0, len(order), BATCH):
            bidx = order[bs : bs + BATCH]
            seqs = [all_ids[j] for j in bidx]
            m = max(len(s) for s in seqs)
            inp = np.full((len(seqs), m), pad_id, dtype=np.int64)
            att = np.zeros((len(seqs), m), dtype=np.int64)
            for k, s in enumerate(seqs):
                inp[k, : len(s)] = s
                att[k, : len(s)] = 1
            cap.clear()
            stack(input_ids=torch.from_numpy(inp).to(dev),
                  attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
            h = cap["h"]
            for k, j in enumerate(bidx):
                acts[j] = h[k, len(seqs[k]) - 1].float().cpu().numpy()
            done += len(seqs)
            if (bs // BATCH) % 50 == 0:
                el = time.time() - t0
                rate = done / max(el, 1e-9)
                log(f"  fwd {done}/{len(rows)}  {rate:.0f}/s  "
                    f"eta {(len(rows) - done) / max(rate, 1e-9) / 60:.1f}min")
    log(f"extraction done in {(time.time() - t0) / 60:.1f}min")

    norms = np.linalg.norm(acts, axis=1)
    stats = {
        "n": int(len(norms)),
        "mean": float(norms.mean()), "median": float(np.median(norms)),
        "p10": float(np.percentile(norms, 10)), "p90": float(np.percentile(norms, 90)),
        "max": float(norms.max()),
        "layer_index": LAYER, "d_model": int(D), "base_model": MODEL,
    }
    # kitft convention: a round number a bit above the mean residual norm.
    mean = stats["mean"]
    step = 10 ** max(0, len(str(int(mean))) - 2)  # e.g. 125 -> 10, 5321 -> 100
    stats["suggested_injection_scale"] = float(-(-mean * 1.2 // step) * step)
    with open(f"{OUT}/norm_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    log("norm stats:", json.dumps(stats, indent=2))
    log(f">>> set INJECTION_SCALE={stats['suggested_injection_scale']:g} in your config "
        f"(measured mean L2 {mean:.1f}) <<<")

    def write_base(split, path):
        sel = [i for i, r in enumerate(rows) if r[0] == split]
        n = len(sel)
        av = np.ascontiguousarray(acts[sel])
        av_arr = pa.FixedSizeListArray.from_arrays(
            pa.array(av.reshape(-1), type=pa.float32()), D
        )
        tbl = pa.table({
            "api_explanation": pa.array([rows[i][5] for i in sel], pa.string()),
            "activation_vector": av_arr,
            "n_raw_tokens": pa.array([int(lens[i]) for i in sel], pa.int64()),
            "activation_layer": pa.array([LAYER] * n, pa.int64()),
            "doc_id": pa.array(
                [f"openbmb/Ultra-FineWeb:en:{rows[i][1]}" for i in sel], pa.string()
            ),
            "detokenized_text_truncated": pa.array([rows[i][4] for i in sel], pa.string()),
            "custom_id": pa.array([rows[i][3] for i in sel], pa.string()),
        })
        pq.write_table(tbl, path)
        log(f"wrote {split}: {n} rows -> {path}")
        return n

    n_av = write_base("av", f"{OUT}/base_av.parquet")
    n_ar = write_base("ar", f"{OUT}/base_ar.parquet")
    log(f"EXTRACT_DONE av={n_av} ar={n_ar}")


try:
    main()
except Exception:
    traceback.print_exc()
    sys.stdout.flush()
    sys.exit(1)
