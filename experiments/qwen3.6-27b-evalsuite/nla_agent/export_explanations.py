#!/usr/bin/env python3
"""Export every precached NLA explanation to one self-documenting JSON.

Merges the two explorer Spaces' precache.json files (matryoshka + standard;
identical texts/tokenizations, asserted) into a single analysis-friendly file:
39 example texts × every token position × both verbalizers, with per-prefix
FVE/cosine reconstruction scores. Meant to be pointed at from another
analysis session — the top-level "schema" key explains every field.

Usage: python3 export_explanations.py [out.json]   (default: nla_explanations.json here)
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SUITE = HERE.parent
sys.path.insert(0, str(HERE))
from build_cache import example_labels  # ast-parses EXAMPLE_LABELS from space/app.py

SCHEMA = {
    "description": (
        "Natural-language autoencoder (NLA) explanations of Qwen3.6-27B layer-42 "
        "residual-stream activations, for 39 sample texts at EVERY token position, "
        "from two verbalizers run on identical tokens. The activation at position i "
        "is causal: it summarizes tokens[0..i] only. Each explanation is ONE "
        "temperature-1 sample (the same sample the public explorer Spaces serve)."),
    "verbalizers": {
        "matryoshka": (
            "ceselder/nla-qwen36-27b-matryoshka (rl_av_lora_iter400 actor + "
            "rl_critic_step400 critic). Emits up to 10 short independent lines "
            "ordered by salience — trained to front-load reconstruction-relevant "
            "information, so early lines matter most."),
        "standard": (
            "ceselder/qwen3.6-27b-nla-L42 (av_rl_lora_step400 actor + co-trained "
            "critic). Emits a short free-form explanation, typically 1-3 longer "
            "lines, NOT salience-ordered (information tends to arrive late)."),
    },
    "fields": {
        "examples[].label": "human-readable source label (WeirdChat entries include behavior category and elo)",
        "examples[].text": "the exact input text",
        "examples[].tokens": "tokens[i] = decoded token piece at position i (real Qwen3.6-27B tokenizer)",
        "examples[].matryoshka / .standard":
            "arrays aligned with tokens: entry i is null or the explanation of the activation at position i",
        "entry.lines": "explanation lines in generation order",
        "entry.fve_cumulative":
            "fve_cumulative[k] = round-trip fraction of variance explained when the critic "
            "reconstructs the activation from lines[0..k] only (1 = perfect, <=0 = no better "
            "than predicting the mean activation); the last value scores the full explanation",
        "entry.cosine": "cosine[k] = cosine similarity of the reconstruction from lines[0..k] vs the true activation",
    },
    "notes": [
        "Marginal value of line k = fve_cumulative[k] - fve_cumulative[k-1].",
        "Positions < ~10 have little left context and mostly decode to generic noise.",
        "Occasional stray CJK characters inside lines are a known sampling artifact.",
        "Full-explanation FVE around 0.5-0.7 is typical.",
    ],
    "model": "Qwen/Qwen3.6-27B",
    "layer": 42,
}


def pack(results: list) -> list:
    return [
        None if r is None else {
            "lines": r["lines"],
            "fve_cumulative": [round(x, 4) for x in r["fve"]],
            "cosine": [round(x, 4) for x in r["cos"]],
        }
        for r in results
    ]


def main() -> None:
    out_path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "nla_explanations.json"
    mat = json.load(open(SUITE / "space" / "precache.json"))["entries"]
    std = json.load(open(SUITE / "space_std" / "precache.json"))["entries"]
    assert [e["ids"] for e in mat] == [e["ids"] for e in std], \
        "mat/std precaches disagree on texts/tokenization"
    labels = example_labels()[: len(mat)]

    examples = []
    for i, (label, em, es) in enumerate(zip(labels, mat, std)):
        examples.append({
            "index": i, "label": label, "text": em["text"],
            "tokens": em["pieces"],
            "matryoshka": pack(em["results"]),
            "standard": pack(es["results"]),
        })

    out = {"schema": SCHEMA, "examples": examples}
    out_path.write_text(json.dumps(out, ensure_ascii=False))
    n_pos = sum(len(e["tokens"]) for e in examples)
    n_m = sum(sum(x is not None for x in e["matryoshka"]) for e in examples)
    n_s = sum(sum(x is not None for x in e["standard"]) for e in examples)
    print(f"{out_path}: {out_path.stat().st_size / 1e6:.1f}MB — "
          f"{len(examples)} examples, {n_pos} positions "
          f"({n_m} matryoshka + {n_s} standard explanations)")


if __name__ == "__main__":
    main()
