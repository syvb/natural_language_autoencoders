"""Stage B (v3) — rebuild the trainable parquets in the v3 BULLETS format.

Reuses the document-level base splits produced by 02_build_datasets.py
(base_{av,ar}_{train,eval}.parquet) — run that first. Like 02b (v2) this only
re-runs nla.datagen.stage3_build, so NO API calls and NO re-split. v3 changes:

  --explanation-format bullets  AV prompt actually says bullet points (v2's
                                shipped prompt still said "2-3 text snippets" in
                                <explanation> tags); response is a raw "- "
                                bullet list, one per line, no wrapper.
  --ar-truncate-max-tokens N    AR: pre-truncate each critic-input explanation
                                to its first K tokens, K ~ U[min, N] — the SAME
                                uniform draw RL's tokens-mode truncation uses,
                                so the online critic is warm-started on the
                                short prefixes it meets at RL step 0 (down to
                                1 token; grad guard is the backstop). Set to
                                match RL's NLA_TRUNC_MAX_TOKENS.
  --ar-truncate-keep-full-frac  ~15% of rows left untruncated: RL prefixes are
                                capped at N too, so without this the critic
                                would never see text longer than N tokens at
                                ANY stage and full-length round-trip FVE would
                                read artificially low (v2's item truncation
                                left many rows whole; this matches that).

NLA_TRUNC_{MIN,MAX}_TOKENS are read from the env ON PURPOSE (same names the RL
script exports) so warm-start and RL draws can't silently desynchronize — but
that also means a shell that overrode them for RL builds a matching warm-start.
The resolved values are echoed; check them.

Outputs: av_sft_v3 / av_eval_v3 / ar_sft_v3 / ar_eval_v3 parquets (+ sidecars).
The eval parquets are built WITHOUT AR truncation (clean full-text eval).

Usage:  python 02c_build_datasets_v3.py   [NLA_TRUNC_MAX_TOKENS=120]
"""
import os
import subprocess
import sys

OUT = os.environ.get("OUT_DIR", "/workspace/out")
MIN_TOKENS = os.environ.get("NLA_TRUNC_MIN_TOKENS", "1")
MAX_TOKENS = os.environ.get("NLA_TRUNC_MAX_TOKENS", "120")
KEEP_FULL = os.environ.get("AR_KEEP_FULL_FRAC", "0.15")
AR_SEED = os.environ.get("AR_TRUNCATE_SEED", "0")
print(f"AR truncation: K ~ U[{MIN_TOKENS}, {MAX_TOKENS}] tokens, "
      f"keep-full frac {KEEP_FULL}, seed {AR_SEED}", flush=True)


def run(stage, inp, out, extra):
    print(f"\n=== stage3_build {stage} (v3) {inp} ===", flush=True)
    cmd = [sys.executable, "-m", "nla.datagen.stage3_build",
           "--input", inp, "--stage", stage, "--output", out,
           "--explanation-format", "bullets", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print("STDERR:\n", r.stderr[-3000:])
        raise SystemExit(f"stage3_build failed: {stage} {inp}")


# AV: bullets format (no wrap), no truncation.
run("av_sft", f"{OUT}/base_av_train.parquet", f"{OUT}/av_sft_v3.parquet", [])
run("av_sft", f"{OUT}/base_av_eval.parquet",  f"{OUT}/av_eval_v3.parquet", [])
# AR: bullets format + uniform token-truncated warm-start inputs (train only; eval stays full-text).
run("ar_sft", f"{OUT}/base_ar_train.parquet", f"{OUT}/ar_sft_v3.parquet",
    ["--ar-truncate-max-tokens", MAX_TOKENS, "--ar-truncate-min-tokens", MIN_TOKENS,
     "--ar-truncate-keep-full-frac", KEEP_FULL, "--ar-truncate-seed", AR_SEED])
run("ar_sft", f"{OUT}/base_ar_eval.parquet",  f"{OUT}/ar_eval_v3.parquet", [])
print("\nBUILD_V3_DONE", flush=True)
