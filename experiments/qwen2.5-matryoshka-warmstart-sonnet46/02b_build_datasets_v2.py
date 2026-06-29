"""Stage B (v2) — rebuild the trainable parquets in the v2 LIST format.

Reuses the document-level base splits produced by 02_build_datasets.py
(base_{av,ar}_{train,eval}.parquet) — run that first. This only re-runs
nla.datagen.stage3_build with the v2 flags, so NO API calls and NO re-split:

  --explanation-format list   AV: no <explanation> wrapper, response is a raw
                              newline list (\\n\\n paragraphs collapsed → \\n).
  --ar-truncate-max-items K   AR: pre-truncate each critic-input explanation to
                              the first ~taper-drawn items, so the online critic
                              is warm-started on the short prefixes it meets at
                              RL step 0 (v2 item 2). Set to match RL's MAX_ITEMS.

Outputs: av_sft_v2 / av_eval_v2 / ar_sft_v2 / ar_eval_v2 parquets (+ sidecars).
The eval parquets are built WITHOUT AR truncation (clean full-text eval).

Usage:  python 02b_build_datasets_v2.py   [MAX_ITEMS=10] [TAPER=2.0]
"""
import os
import subprocess
import sys

OUT = os.environ.get("OUT_DIR", "/workspace/out")
MAX_ITEMS = os.environ.get("NLA_TRUNC_MAX_ITEMS", "10")
TAPER = os.environ.get("NLA_TRUNC_TAPER", "2.0")
AR_SEED = os.environ.get("AR_TRUNCATE_SEED", "0")


def run(stage, inp, out, extra):
    print(f"\n=== stage3_build {stage} (v2) {inp} ===", flush=True)
    cmd = [sys.executable, "-m", "nla.datagen.stage3_build",
           "--input", inp, "--stage", stage, "--output", out,
           "--explanation-format", "list", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-3000:])
        raise SystemExit(f"stage3_build failed: {stage} {inp}")


# AV: list format (no wrap), no truncation.
run("av_sft", f"{OUT}/base_av_train.parquet", f"{OUT}/av_sft_v2.parquet", [])
run("av_sft", f"{OUT}/base_av_eval.parquet",  f"{OUT}/av_eval_v2.parquet", [])
# AR: list format + item-truncated warm-start inputs (train only; eval stays full-text).
run("ar_sft", f"{OUT}/base_ar_train.parquet", f"{OUT}/ar_sft_v2.parquet",
    ["--ar-truncate-max-items", MAX_ITEMS, "--ar-truncate-taper", TAPER, "--ar-truncate-seed", AR_SEED])
run("ar_sft", f"{OUT}/base_ar_eval.parquet",  f"{OUT}/ar_eval_v2.parquet", [])
print("\nBUILD_V2_DONE", flush=True)
