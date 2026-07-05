"""Bring-up gate for a NEW base model (RUNBOOK phase 1) — one GPU.

Run AFTER a small extraction (`EXTRACT_LIMIT=300 python 01_extract_activations.py`)
with the target config. Checks the whole injection path mechanically:

  1. injection-token search + canonical neighbors for this tokenizer
  2. loads the FULL base model (bf16), renders the bullets actor prompt via
     the chat template, embeds it, and injects a REAL activation from
     base_av.parquet scaled to INJECTION_SCALE (falls back to the measured
     suggestion in norm_stats.json)
  3. generates twice with the SAME seed (T=1 sampling): injected vs ablated
     (marker embedding left in place). PASS requires the outputs to DIFFER —
     the vector demonstrably influenced computation — and generation not to
     crash on shapes/dtypes (d_model, vocab, wrapper plumbing).
  4. prints CJK stats + both texts for eyeballing. On an un-SFT'd base model
     the content will be unfocused — this gates PLUMBING, not quality. The
     classic failure smell (whole output free-associating Chinese) means the
     model saw the literal CJK marker char: injection silently failed.

Exit 0 PASS / 1 FAIL.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config


def main():
    import numpy as np
    import pyarrow.parquet as pq
    import torch

    from nla.datagen._common import load_tokenizer
    from nla.datagen.injection_tokens import build_token_meta
    from nla.datagen.stage3_build import _ACTOR_TEMPLATES
    from nla.injection import inject_at_marked_positions
    from nla.schema import normalize_activation

    _config.load()
    WORK = _config.env("WORK")
    MODEL = os.environ.get("BASE_MODEL_DIR", f"{WORK}/models/base")
    D = int(_config.env("D_MODEL"))

    scale = os.environ.get("INJECTION_SCALE")
    if not scale:
        stats = json.load(open(f"{WORK}/out/norm_stats.json"))
        scale = stats["suggested_injection_scale"]
        print(f"INJECTION_SCALE unset — using measured suggestion {scale}")
    scale = float(scale)

    tok = load_tokenizer(MODEL)
    template = _ACTOR_TEMPLATES["bullets"]
    meta = build_token_meta(tok, template)
    print(f"injection token {meta.injection_char!r} -> {meta.injection_token_id}, "
          f"neighbors ({meta.injection_left_neighbor_id}, {meta.injection_right_neighbor_id})")

    t = pq.read_table(f"{WORK}/out/base_av.parquet",
                      columns=["activation_vector", "api_explanation"])
    vec = torch.tensor(np.array(t.column("activation_vector")[0].as_py(), dtype=np.float32))
    assert vec.shape[-1] == D, (vec.shape, D)
    print(f"gold explanation for this vector (for eyeballing):\n  "
          f"{t.column('api_explanation')[0].as_py()[:300]}...")

    sys.path.pop(0)  # drop the experiment dir before heavy imports
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                     device_map="cuda").eval()
    except Exception as e:  # noqa: BLE001
        print(f"AutoModelForCausalLM failed ({type(e).__name__}); trying image-text auto class")
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    msgs = [{"role": "user", "content": template.format(injection_char=meta.injection_char)}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):  # transformers 5.x returns BatchEncoding
        ids = ids["input_ids"]
    ids = torch.tensor([ids], device=model.device)
    emb_layer = model.get_input_embeddings()
    emb = emb_layer(ids)

    vec_scaled = normalize_activation(vec.to(model.device), scale).to(emb.dtype)
    emb_inj = inject_at_marked_positions(
        ids, emb.clone(), vec_scaled.unsqueeze(0),
        meta.injection_token_id, meta.injection_left_neighbor_id,
        meta.injection_right_neighbor_id,
    )
    assert not torch.equal(emb_inj, emb), "injection did not modify any embedding"

    gen_kw = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                  max_new_tokens=80, pad_token_id=tok.pad_token_id or tok.eos_token_id)

    def gen(embeds, seed=1234):
        torch.manual_seed(seed)
        out = model.generate(inputs_embeds=embeds,
                             attention_mask=torch.ones(ids.shape, device=model.device),
                             **gen_kw)
        return tok.decode(out[0], skip_special_tokens=True)

    txt_inj = gen(emb_inj)
    txt_abl = gen(emb)

    def cjk_frac(s):
        return sum(1 for c in s if "一" <= c <= "鿿") / max(1, len(s))

    print(f"\n--- INJECTED (CJK {cjk_frac(txt_inj):.1%}) ---\n{txt_inj}")
    print(f"\n--- ABLATED / marker-in-place (CJK {cjk_frac(txt_abl):.1%}) ---\n{txt_abl}")

    if txt_inj == txt_abl:
        print("BRINGUP_FAIL: injected and ablated outputs are IDENTICAL — the "
              "vector did not influence computation (wrong embedding path?)")
        sys.exit(1)
    if cjk_frac(txt_inj) > 0.5:
        print("BRINGUP_FAIL: injected output is majority CJK — the model likely "
              "saw the literal marker char (injection silently failed)")
        sys.exit(1)
    print("\nBRINGUP_PASS: injection alters generation; no CJK free-association.")


if __name__ == "__main__":
    main()
