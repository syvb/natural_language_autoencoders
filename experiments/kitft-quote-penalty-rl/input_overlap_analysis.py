"""How much does the NLA repeat text from its input? Overlap metrics.

Parses the 16-sample comparison file (eval/compare_pre_vs_50.md), recovers each
sample's FULL input context + gold Sonnet explanation from av_eval.parquet (by
doc_id@n_raw_tokens), and scores every generation against its input with:

  ngram_prec_n   fraction of the generation's word n-grams present in the input
                 (n=1..5). n=1 ~ topicality; n>=3 ~ verbatim copying.
  coverage       extractive fragment coverage (Grusky et al. 2018): fraction of
                 generation tokens inside maximal fragments shared with input.
  density        extractive fragment density: mean squared shared-fragment
                 length — the "how verbatim" number (long copies dominate).
  lcs_tokens     longest common token run with the input.
  final_echo     1 if the input's final word appears anywhere in the generation.

Tokenization: lowercase word tokens (\\w+), so quote marks and punctuation play
no role — this measures repeated CONTENT, not quoting style.

Controls: gold Sonnet explanations (a "good explanation" reference) and a
shuffled pairing (each generation vs another sample's context) for chance-level
topical overlap.

Usage: python input_overlap_analysis.py <compare.md> <av_eval.parquet> <out.md>
"""
import re
import statistics
import sys

import pyarrow.parquet as pq

MD, EVAL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NGRAMS = (1, 2, 3, 4, 5)


def words(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def ngram_prec(gen: list[str], src: list[str], n: int) -> float | None:
    g = [tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)]
    if not g:
        return None
    s = ngrams(src, n)
    return sum(1 for t in g if t in s) / len(g)


def shared_fragments(gen: list[str], src: list[str]) -> list[int]:
    """Greedy maximal shared fragments (Grusky et al. 2018, Alg. 1).
    Returns the fragment length for each generation-token position run."""
    frags, i = [], 0
    # index src token -> positions for O(1) start lookup
    pos: dict[str, list[int]] = {}
    for j, t in enumerate(src):
        pos.setdefault(t, []).append(j)
    while i < len(gen):
        best = 0
        for j in pos.get(gen[i], ()):
            k = 0
            while i + k < len(gen) and j + k < len(src) and gen[i + k] == src[j + k]:
                k += 1
            best = max(best, k)
        if best:
            frags.append(best)
            i += best
        else:
            i += 1
    return frags


def score(gen_text: str, src_text: str) -> dict:
    gen, src = words(gen_text), words(src_text)
    frags = shared_fragments(gen, src)
    out = {f"ngram{n}": ngram_prec(gen, src, n) for n in NGRAMS}
    out["coverage"] = sum(frags) / len(gen) if gen else None
    out["density"] = sum(f * f for f in frags) / len(gen) if gen else None
    out["lcs"] = max(frags, default=0)
    out["final_echo"] = int(bool(src) and src[-1] in gen)
    return out


# ── parse the comparison md ──────────────────────────────────────────────────
md = open(MD, encoding="utf-8").read()
blocks = re.split(r"\n## Sample \d+/\d+ — `(.+?)`\n", md)[1:]
samples = []  # (cid, {model: text})
for cid, body in zip(blocks[0::2], blocks[1::2]):
    gens = dict(re.findall(r"\*\*(\w+)\*\* \(quote chars: \d+\):\n```text\n(.*?)\n```",
                           body, re.S))
    assert gens, f"no generations parsed for {cid}"
    samples.append((cid, gens))
print(f"parsed {len(samples)} samples, models: {sorted(samples[0][1])}")

# ── full contexts + gold from av_eval ────────────────────────────────────────
t = pq.read_table(EVAL, columns=["doc_id", "n_raw_tokens", "detokenized_text_truncated",
                                 "response"])
idx = {f"{d}@{n}": i for i, (d, n) in enumerate(zip(t.column("doc_id").to_pylist(),
                                                    t.column("n_raw_tokens").to_pylist()))}
ctx, gold = {}, {}
for cid, _ in samples:
    i = idx[cid]
    ctx[cid] = t.column("detokenized_text_truncated")[i].as_py()
    g = t.column("response")[i].as_py()
    m = re.search(r"<explanation>\n?(.*?)\n?</explanation>", g, re.S)
    gold[cid] = m.group(1) if m else g

# ── score all conditions ─────────────────────────────────────────────────────
model_names = sorted(samples[0][1]) + ["gold"]
rows: dict[str, list[dict]] = {m: [] for m in model_names}
rows["shuffled-ctrl"] = []
for si, (cid, gens) in enumerate(samples):
    for m in sorted(gens):
        rows[m].append(score(gens[m], ctx[cid]))
    rows["gold"].append(score(gold[cid], ctx[cid]))
    # control: preRL generation vs a DIFFERENT sample's context
    other = samples[(si + 1) % len(samples)][0]
    rows["shuffled-ctrl"].append(score(gens[sorted(gens)[1]], ctx[other]))

metrics = [f"ngram{n}" for n in NGRAMS] + ["coverage", "density", "lcs", "final_echo"]
L = ["# Input-overlap analysis: how much does the NLA repeat its input?", ""]
L.append(f"{len(samples)} held-out samples; generations scored against their FULL input "
         f"context (mean over samples; word-level, punctuation-free).")
L.append("`shuffled-ctrl` = pre-RL generations vs a different sample's context "
         "(chance/topicality floor). `gold` = Sonnet-4.6 explanations.")
L.append("")
L.append("| condition | " + " | ".join(metrics) + " |")
L.append("|" + "---|" * (len(metrics) + 1))
for m in model_names + ["shuffled-ctrl"]:
    vals = []
    for k in metrics:
        xs = [r[k] for r in rows[m] if r[k] is not None]
        v = statistics.mean(xs)
        vals.append(f"{v:.3f}" if k not in ("lcs",) else f"{v:.1f}")
    L.append(f"| {m} | " + " | ".join(vals) + " |")
L.append("")
L.append("Per-sample longest copied token run (lcs), preRL vs iter50:")
pre_name = [m for m in model_names if m.lower().startswith("prerl")][0]
i50_name = [m for m in model_names if "50" in m][0]
L.append("```")
L.append("sample  preRL  iter50  gold")
for si, (cid, _) in enumerate(samples):
    L.append(f"{si + 1:>6}  {rows[pre_name][si]['lcs']:>5}  {rows[i50_name][si]['lcs']:>6}"
             f"  {rows['gold'][si]['lcs']:>4}")
L.append("```")
open(OUT, "w").write("\n".join(L) + "\n")
print("WROTE", OUT)
