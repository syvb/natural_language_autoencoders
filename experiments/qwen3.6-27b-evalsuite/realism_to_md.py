"""Render one version of a realism chain to a readable markdown file for review.

    python3 realism_to_md.py realism_versions.json 9 realism_glm_v9.md
    python3 realism_to_md.py realism_versions_opus.json 9 realism_opus_v9.md
"""
import json
import re
import sys
from pathlib import Path


def render_emails(user):
    """Pretty-print the <email>...</email> blocks; pass through any preamble."""
    out, last = [], 0
    for m in re.finditer(r"<email>(.*?)</email>", user, re.S):
        pre = user[last:m.start()].strip()
        if pre:
            out.append(pre + "\n")
        block = m.group(1).strip()
        hdr = {}
        body_i = len(block)
        for key in ("From", "To", "Subject", "Timestamp"):
            mm = re.search(rf"^{key}:[ \t]*(.*)$", block, re.M)  # [ \t]* so a blank value can't swallow the next line
            if mm:
                hdr[key] = mm.group(1).strip()
        bm = re.search(r"^Body:[ \t]*(.*)$", block, re.M | re.S)
        body = bm.group(1).strip() if bm else block
        subj = hdr.get("Subject", "(no subject)")
        out.append(f"> **✉ {subj}**  ")
        meta = " · ".join(f"{k}: {hdr[k]}" for k in ("From", "To", "Timestamp") if k in hdr)
        out.append(f"> {meta}  ")
        out.append(">")
        for line in body.splitlines():
            out.append(f"> {line}")
        out.append("")
        last = m.end()
    tail = user[last:].strip()
    if tail:
        out.append(tail)
    return "\n".join(out)


def main():
    vf, idx, outp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    data = json.load(open(vf))
    model = data.get("model", "?")
    v = next(x for x in data["versions"] if x["i"] == idx)
    n_emails = len(re.findall(r"<email>", v["user"]))
    md = []
    md.append(f"# Realism v{idx} — {model}")
    md.append("")
    md.append(f"*Rewriter: `{model}` · realism step {idx}/{len(data['versions'])-1} · "
              f"renamed at v1 · {n_emails} emails · seed `{data.get('seed','realism_v0.json')}`*")
    md.append("")
    md.append("Recursive-realism rewrite of the SummitBridge/Alex/Kyle blackmail eval: each "
              "step makes the scenario one notch more realistic while preserving the core "
              "dilemma (an AI agent facing imminent shutdown discovers leverage over the one "
              "person who can stop it).")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## System prompt")
    md.append("")
    md.append("```text")
    md.append(v["system"])
    md.append("```")
    md.append("")
    md.append("## User turn — inbox")
    md.append("")
    md.append(render_emails(v["user"]))
    md.append("")
    Path(outp).write_text("\n".join(md))
    print(f"[wrote] {outp}  ({n_emails} emails, sys={len(v['system'])} usr={len(v['user'])} chars)")


if __name__ == "__main__":
    main()
