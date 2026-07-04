"""Step 6 — build the RL (stage-"rl") parquet: {prompt with marker, activation}.

Built from base_av_train.parquet (document-disjoint from av_eval, so RL never
trains on a doc the round-trip eval scores), bullets format to match the
warm-start prompt. Skip-if-exists verifies the existing parquet's sidecar
carries the bullets actor template — a stale wrong-format file fails loudly.

Run after 02:  python 05_build_rl_parquet.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _config

import yaml

from nla.datagen.stage3_build import _ACTOR_TEMPLATES

_config.load()
WORK = _config.env("WORK")
IN = os.environ.get("RL_BASE_INPUT", f"{WORK}/out/base_av_train.parquet")
OUT = os.environ.get("RL_PARQUET", f"{WORK}/out/rl.parquet")
FMT = "bullets"

if os.path.exists(OUT):
    sidecar = f"{OUT}.nla_meta.yaml"
    assert os.path.exists(sidecar), (
        f"{OUT} exists but has no sidecar ({sidecar}) — can't verify its prompt "
        f"format. Delete it and rebuild."
    )
    actor_tpl = yaml.safe_load(open(sidecar))["prompt_templates"]["actor"]
    assert actor_tpl == _ACTOR_TEMPLATES[FMT], (
        f"existing {OUT} was built with a DIFFERENT actor template than "
        f"{FMT} — delete it and rebuild. (starts: {actor_tpl[:100]!r}...)"
    )
    print(f"RL parquet already exists with the {FMT} template: {OUT} — skipping",
          flush=True)
    sys.exit(0)

assert os.path.exists(IN), (
    f"missing {IN} — run 02_build_datasets.py first (writes base_av_train.parquet)."
)

print(f"=== stage3_build rl ({FMT}): {IN} -> {OUT} ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "nla.datagen.stage3_build",
     "--stage", "rl", "--input", IN, "--output", OUT,
     "--explanation-format", FMT],
    capture_output=True, text=True,
)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("STDERR:\n", r.stderr[-3000:])
    raise SystemExit("stage3_build rl failed")
print(f"RL_PARQUET_BUILT {OUT}", flush=True)
