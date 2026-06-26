"""Stage E — build the RL (stage-"rl") parquet for the truncated RL phase.

RL needs a parquet of {prompt (messages with <INJECT>), activation_vector} and
NO response column. We build it from the AV TRAIN split that 02_build_datasets.py
already wrote (base_av_train.parquet — stage="base", norm="none", train docs
only). That split is document-disjoint from av_eval, so RL never trains on a doc
the round-trip eval scores.

The actor template defaults to stage3_build's, which is exactly what the
warm-start used (02 also used defaults) — so the RL prompt format matches the
SFT prompt format the checkpoints were trained on.

Run on the box after 02_build_datasets.py (idempotent — skips if output exists):
    python 05_build_rl_parquet.py
"""
import os
import subprocess
import sys

IN = os.environ.get("RL_BASE_INPUT", "/workspace/out/base_av_train.parquet")
OUT = os.environ.get("RL_PARQUET", "/workspace/out/rl.parquet")

if os.path.exists(OUT):
    print(f"RL parquet already exists: {OUT} — skipping (delete to rebuild)", flush=True)
    sys.exit(0)

assert os.path.exists(IN), (
    f"missing {IN} — run 02_build_datasets.py first (it writes base_av_train.parquet). "
    f"Override the input with RL_BASE_INPUT=..."
)

print(f"=== stage3_build rl: {IN} -> {OUT} ===", flush=True)
r = subprocess.run(
    [sys.executable, "-m", "nla.datagen.stage3_build",
     "--stage", "rl", "--input", IN, "--output", OUT],
    capture_output=True, text=True,
)
print(r.stdout[-2000:])
if r.returncode != 0:
    print("STDERR:\n", r.stderr[-3000:])
    raise SystemExit("stage3_build rl failed")
print(f"RL_PARQUET_BUILT {OUT}", flush=True)
