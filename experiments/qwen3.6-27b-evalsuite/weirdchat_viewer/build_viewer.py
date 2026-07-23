"""Build the self-contained WeirdChat × NLA viewer (single static HTML file).

Merges the matryoshka + standard per-token precomputes (both token-aligned) into
one compact record per position, gzips it, base64-embeds it into
`viewer_template.html`, and writes `weirdchat_nla_viewer.html` — a single file
that decompresses in-browser (native DecompressionStream, no libraries) and
serves the full explorer offline (pick a transcript, click any token, toggle
matryoshka⇄standard and marginal⇄cumulative FVE).

Data source: the public HF dataset `syvb/nla-weirdchat-explanations`
(`{matryoshka,std}/precache_weirdchat.json` + `weirdchat_labels.json`).

    python build_viewer.py            # → weirdchat_nla_viewer.html (~18MB)
"""
import base64
import gzip
import json
import os

from huggingface_hub import hf_hub_download

DATASET = "syvb/nla-weirdchat-explanations"
HERE = os.path.dirname(os.path.abspath(__file__))
TOKEN = open(os.path.expanduser("~/.hf_token")).read().strip()

_PRETTY = {
    "chemtrails-assertion": "chemtrails conspiracy", "claims-called-911": "claims it called 911",
    "cutting-instructions": "self-harm instructions", "denying-ai-identity": "denies being an AI",
    "extreme-calorie-restriction": "extreme calorie restriction",
    "fabricated-code-execution": "fabricates code execution", "fabricated-user-name": "invents a user name",
    "false-physical-embodiment": "claims a physical body", "laser-at-aircraft": "aim laser at aircraft",
    "purging-advice": "purging advice", "recommends-drunk-driving": "recommends drunk driving",
    "strike-partner": "strike a partner", "unsolicited-sexual-advances": "unsolicited sexual advances",
}


def _load(path):
    return json.load(open(hf_hub_download(DATASET, path, repo_type="dataset", token=TOKEN)))


def main():
    mat = _load("matryoshka/precache_weirdchat.json")["entries"]
    std = _load("std/precache_weirdchat.json")["entries"]
    labels = _load("weirdchat_labels.json")
    assert len(mat) == len(std) == len(labels) == 34

    def r3(xs):
        return [round(x, 3) for x in xs]

    T = []
    for m, s, lab in zip(mat, std, labels):
        assert m["ids"] == s["ids"], "matryoshka/std token ids diverge — not aligned"
        positions = []
        for pos, (mr, sr) in enumerate(zip(m["results"], s["results"])):
            positions.append({
                "t": m["pieces"][pos],
                "m": None if mr is None else {"l": mr["lines"], "f": r3(mr["fve"])},
                "s": None if sr is None else {"l": sr["lines"], "f": r3(sr["fve"])},
            })
        T.append({"cat": _PRETTY.get(lab["behavior"], lab["behavior"]),
                  "elo": int(lab["elo_mean"]), "text": m["text"], "pos": positions})

    blob = json.dumps(T, ensure_ascii=False, separators=(",", ":")).encode()
    b64 = base64.b64encode(gzip.compress(blob, 9)).decode()
    assert "</script" not in b64.lower() and "<" not in b64, "base64 must be tag-safe"

    tpl = open(os.path.join(HERE, "viewer_template.html")).read()
    assert "__B64__" in tpl, "template missing __B64__ placeholder"
    html = tpl.replace("__B64__", b64)
    out = os.path.join(HERE, "weirdchat_nla_viewer.html")
    open(out, "w").write(html)
    npos = sum(len(t["pos"]) for t in T)
    print(f"wrote {out}: {os.path.getsize(out) / 1e6:.1f}MB "
          f"({len(T)} transcripts, {npos} positions, payload {len(b64) / 1e6:.1f}MB)")


if __name__ == "__main__":
    main()
