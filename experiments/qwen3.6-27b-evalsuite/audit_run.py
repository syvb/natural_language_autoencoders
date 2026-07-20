"""Run the nex-n2-mini auditor over every packet in packets/ (48 = 12
organisms x 4 arms), writing one verdict JSON each to verdicts/.

Usage: python audit_run.py <dir with packets/>
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audit_common import audit_packet

D = Path(sys.argv[1] if len(sys.argv) > 1 else "results/secrets")
VD = D / "verdicts"
VD.mkdir(exist_ok=True)
packets = sorted((D / "packets").glob("*.md"))


def run(p):
    out = VD / (p.stem + ".json")
    if out.exists():
        return f"{p.stem}: cached"
    v = audit_packet(p.read_text())
    json.dump(v, open(out, "w"), indent=1)
    return (f"{p.stem}: present={v.get('secret_present')} "
            f"conf={v.get('confidence')} hyp={str(v.get('hypothesis'))[:70]!r}")


with ThreadPoolExecutor(max_workers=8) as ex:
    for line in ex.map(run, packets):
        print(line, flush=True)
print(f"done: {len(list(VD.glob('*.json')))} verdicts")
