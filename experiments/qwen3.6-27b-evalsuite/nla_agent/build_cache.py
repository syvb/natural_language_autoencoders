#!/usr/bin/env python3
"""Build nla-agent-cache.json.gz — the precached examples bundle for nla-agent.html.

Merges the two explorer Spaces' precache.json files (matryoshka + standard;
identical texts and token ids, asserted) into one gzipped JSON the static page
fetches from syvb.ca, so precached examples never touch Hugging Face at all:

  {"labels": [...],
   "examples": [{"text": str, "pieces": [str], "m": [res|null], "s": [res|null]}]}

with res = {"lines": [str], "fve": [float4], "cos": [float4]} per token position.
Labels come from EXAMPLE_LABELS in space/app.py (parsed via ast — no import,
app.py needs GPUs at import time).
"""
import ast
import gzip
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SUITE = HERE.parent


def example_labels() -> list[str]:
    tree = ast.parse((SUITE / "space" / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "EXAMPLE_LABELS" for t in node.targets
        ):
            v = node.value
            if isinstance(v, ast.Subscript):  # trailing [: len(DEFAULT_TEXTS)] slice
                v = v.value
            return ast.literal_eval(v)
    raise SystemExit("EXAMPLE_LABELS not found in space/app.py")


def pack(results: list) -> list:
    return [
        None if r is None else {
            "lines": r["lines"],
            "fve": [round(x, 4) for x in r["fve"]],
            "cos": [round(x, 4) for x in r["cos"]],
        }
        for r in results
    ]


def main() -> None:
    mat = json.load(open(SUITE / "space" / "precache.json"))["entries"]
    std = json.load(open(SUITE / "space_std" / "precache.json"))["entries"]
    assert [e["ids"] for e in mat] == [e["ids"] for e in std], \
        "mat/std precaches disagree on texts/tokenization"
    labels = example_labels()[: len(mat)]
    assert len(labels) == len(mat)

    out = {"labels": labels, "examples": []}
    for em, es in zip(mat, std):
        out["examples"].append({
            "text": em["text"], "pieces": em["pieces"],
            "m": pack(em["results"]), "s": pack(es["results"]),
        })

    raw = json.dumps(out, separators=(",", ":"), ensure_ascii=False).encode()
    dst = HERE / "nla-agent-cache.json.gz"
    # mtime=0 → byte-reproducible archive for identical inputs
    dst.write_bytes(gzip.compress(raw, mtime=0))
    n_pos = sum(len(e["pieces"]) for e in out["examples"])
    print(f"{dst.name}: {len(raw) / 1e6:.1f}MB raw → {dst.stat().st_size / 1e6:.1f}MB gz "
          f"({len(out['examples'])} examples, {n_pos} positions, mat+std)")


if __name__ == "__main__":
    main()
