"""Merge the v3-control and v3qf held-out dumps into one side-by-side markdown.

Both inputs come from gen_heldout.py over the same eval parquet and doc
selection, so sections pair up by doc_id. Usage:
    python build_compare_md.py heldout_v3_50.md heldout_v3qf_50.md compare_v3_vs_v3qf_50.md
"""
import re
import sys

SEC = re.compile(
    r"^## \d+\. doc `(?P<doc>[^`]+)`\s+\((?P<nlines>\d+) lines, quotes=(?P<q>\d+), CJK=(?P<cjk>\w+)\)\n"
    r"(?:\n\*source text \(tail\):\* (?P<src>.*?)\n)?"
    r"\n```\n(?P<body>.*?)```",
    re.M | re.S,
)


def parse(path):
    out = {}
    for m in SEC.finditer(open(path).read()):
        out[m["doc"]] = m.groupdict()
    return out


def totals(d):
    q = sum(int(s["q"]) for s in d.values())
    zero = sum(1 for s in d.values() if s["q"] == "0")
    cjk = sum(1 for s in d.values() if s["cjk"] == "True")
    return q, zero, cjk


v3_path, qf_path, out_path = sys.argv[1:4]
v3, qf = parse(v3_path), parse(qf_path)
docs = [d for d in v3 if d in qf]
assert len(docs) >= 45, f"only {len(docs)} paired docs — parse problem?"

v3q, v3z, v3c = totals(v3)
qfq, qfz, qfc = totals(qf)
n = len(docs)
L = [
    "# v3 (control) vs v3qf (quote-penalty) — same 50 held-out activations\n",
    "Same eval activations (`av_eval_v3.parquet`, one per distinct doc), same "
    "decoding protocol and budget, through the v3-final AV "
    "(`syvb/nla-qwen2.5-7b-L20-v3-rl` `iter_0000200/av` — the checkpoint v3qf "
    "continued from, no quote penalty) and the v3qf AV "
    "(`syvb/nla-qwen2.5-7b-L20-v3qf-rl` `hf/iter_0000100/av`, 100 RL steps with "
    "the 0.1/char penalty). `quotes` counts penalized quotation-mark characters.\n",
    "| | v3 control | v3qf |",
    "|---|---|---|",
    f"| quote chars total ({n} samples) | {v3q} ({v3q/n:.2f}/sample) | {qfq} ({qfq/n:.2f}/sample) |",
    f"| zero-quote samples | {v3z} / {n} | {qfz} / {n} |",
    f"| samples w/ stray CJK | {v3c} / {n} | {qfc} / {n} |",
    "",
]
for i, doc in enumerate(docs):
    a, b = v3[doc], qf[doc]
    L.append(f"## {i}. doc `{doc}`\n")
    if a["src"]:
        L.append(f"*source text (tail):* {a['src']}\n")
    L.append(f"**v3 control** ({a['nlines']} lines, quotes={a['q']}, CJK={a['cjk']})\n")
    L.append(f"```\n{a['body']}```\n")
    L.append(f"**v3qf** ({b['nlines']} lines, quotes={b['q']}, CJK={b['cjk']})\n")
    L.append(f"```\n{b['body']}```\n")
open(out_path, "w").write("\n".join(L))
print(f"wrote {out_path}: {n} paired docs | quotes v3={v3q} v3qf={qfq}")
