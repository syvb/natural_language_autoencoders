"""Stage E — build the RL (stage-"rl") parquet for the truncated RL phase.

RL needs a parquet of {prompt (messages with <INJECT>), activation_vector} and
NO response column. We build it from the AV TRAIN split that 02_build_datasets.py
already wrote (base_av_train.parquet — stage="base", norm="none", train docs
only). That split is document-disjoint from av_eval, so RL never trains on a doc
the round-trip eval scores.

EXPLANATION_FORMAT is REQUIRED and must match what the warm-start parquets were
built with, or the RL prompt disagrees with the prompt the AV was SFT'd on. This
bit v2: this script used to fall back to stage3_build's default (tagged), so
even though the v2 warm-start data was built with --explanation-format list,
every v2 RL rollout carried the v1 "2-3 text snippets" prompt. Values:
  v1: tagged    v2: list    v3: bullets

The skip-if-exists path verifies the existing parquet's sidecar actually has
the requested format's actor template — a stale output built with the wrong
format fails loudly instead of being silently reused.

Run on the box after 02_build_datasets.py:
    EXPLANATION_FORMAT=bullets RL_PARQUET=/workspace/out/rl_v3.parquet python 05_build_rl_parquet.py
"""
import os
import subprocess
import sys

import yaml

from nla.datagen.stage3_build import _ACTOR_TEMPLATES

IN = os.environ.get("RL_BASE_INPUT", "/workspace/out/base_av_train.parquet")
OUT = os.environ.get("RL_PARQUET", "/workspace/out/rl.parquet")
FMT = os.environ.get("EXPLANATION_FORMAT")
assert FMT in _ACTOR_TEMPLATES, (
    f"EXPLANATION_FORMAT is required (got {FMT!r}) — one of {sorted(_ACTOR_TEMPLATES)}. "
    f"It must match the warm-start's format (v1: tagged, v2: list, v3: bullets); "
    f"an implicit default is how v2 shipped RL prompts in the wrong format."
)

if os.path.exists(OUT):
    sidecar = f"{OUT}.nla_meta.yaml"
    assert os.path.exists(sidecar), (
        f"{OUT} exists but has no sidecar ({sidecar}) — can't verify its prompt "
        f"format. Delete it and rebuild."
    )
    actor_tpl = yaml.safe_load(open(sidecar))["prompt_templates"]["actor"]
    assert actor_tpl == _ACTOR_TEMPLATES[FMT], (
        f"existing {OUT} was built with a DIFFERENT actor template than "
        f"EXPLANATION_FORMAT={FMT} — delete it and rebuild. "
        f"(Its template starts: {actor_tpl[:100]!r}...)"
    )
    print(f"RL parquet already exists with the {FMT} template: {OUT} — skipping "
          f"(delete to rebuild)", flush=True)
    sys.exit(0)

assert os.path.exists(IN), (
    f"missing {IN} — run 02_build_datasets.py first (it writes base_av_train.parquet). "
    f"Override the input with RL_BASE_INPUT=..."
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
