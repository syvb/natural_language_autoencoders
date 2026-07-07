"""Sidecar wandb logger: quote-usage metrics for the v3qf run.

Runs NEXT TO training (no code change to the run): every INTERVAL seconds,
log to wandb as a separate run in the same project/group (job_type=quote-stats),
x = latest rollout step from train.log. Two sources:

1. FULL-BATCH stats (`quotefull/*`) — consume new lines of
   NLA_QUOTE_STATS_JSONL, written by nla.reward per reward drain over EVERY
   scored sample (~512/step). n-weighted across the lines seen since the last
   log. Exact, the headline curves:
     quotefull/chars_mean   quote-mark chars per sample (the penalized set)
     quotefull/chars_max
     quotefull/frac_zero    fraction of samples with NO quote chars
     quotefull/est_penalty  NLA_QUOTE_PENALTY x chars_mean
     quotefull/n_samples

2. DUMP stats (`quote/*`) — the first-20-samples rollout text dump, same
   metrics as the kitft sidecar; keeps the two experiments comparable and
   carries the dump-only extras:
     quote/chars_mean|chars_median|chars_max|frac_zero
     quote/double_mean quote/single_mean   (double vs single/apostrophe family)
     quote/final_echo_frac                 ("final token" echo habit, quoted or not)
     quote/est_penalty quote/n_samples

Usage (on the box, after the launcher starts):
  nohup python quote_stats_wandb.py > /workspace/out/quote_stats_sidecar.log 2>&1 &
Exits when the training launcher disappears.
"""
import json
import os
import re
import statistics
import subprocess
import time

import wandb

DUMP = os.environ.get("NLA_ROLLOUT_TEXT_DUMP", "/workspace/out/rollout_dump.txt")
JSONL = os.environ.get("NLA_QUOTE_STATS_JSONL", "/workspace/out/quote_stats.jsonl")
TRAIN_LOG = os.environ.get("TRAIN_LOG", "/workspace/out/train.log")
COEF = float(os.environ.get("NLA_QUOTE_PENALTY", "0.1"))
INTERVAL = float(os.environ.get("INTERVAL", "30"))
TRAIN_PGREP = os.environ.get("TRAIN_PGREP", "run_rl_v3q[f].sh")

# THE penalized set — imported so the sidecar can never drift from what the
# reward actually counts.
from nla.reward import _QUOTE_CHARS as QUOTES

DOUBLES = frozenset('"“”„‟«»＂❝❞⹂') & QUOTES
SINGLES = frozenset("'‘’‚‛＇❛❜") & QUOTES

_jsonl_pos = 0  # byte offset of consumed JSONL


def latest_step() -> int | None:
    try:
        txt = open(TRAIN_LOG, encoding="utf-8", errors="replace").read()[-200_000:]
    except OSError:
        return None
    hits = re.findall(r"perf (\d+)", txt.replace("\r", "\n"))
    return int(hits[-1]) if hits else None


def jsonl_stats() -> dict | None:
    """n-weighted aggregate of the JSONL lines appended since the last call."""
    global _jsonl_pos
    try:
        with open(JSONL, encoding="utf-8", errors="replace") as f:
            f.seek(_jsonl_pos)
            new = f.read()
            _jsonl_pos = f.tell()
    except OSError:
        return None
    recs = []
    for line in new.splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # torn tail write; re-read next round is lost — acceptable
    if not recs:
        return None
    n = sum(r["n"] for r in recs)
    mean = sum(r["quote_chars_mean"] * r["n"] for r in recs) / n
    return {
        "quotefull/chars_mean": mean,
        "quotefull/chars_max": max(r["quote_chars_max"] for r in recs),
        "quotefull/frac_zero": sum(r["frac_zero"] * r["n"] for r in recs) / n,
        "quotefull/est_penalty": COEF * mean,
        "quotefull/n_samples": n,
    }


def dump_stats() -> dict | None:
    try:
        txt = open(DUMP, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
    samples = re.split(r"=== sample \d+ .*? ===", txt)[1:]
    if not samples:
        return None
    q = [sum(1 for c in s if c in QUOTES) for s in samples]
    return {
        "quote/chars_mean": sum(q) / len(q),
        "quote/chars_median": statistics.median(q),
        "quote/chars_max": max(q),
        "quote/frac_zero": sum(1 for c in q if c == 0) / len(q),
        "quote/double_mean": sum(sum(1 for c in s if c in DOUBLES) for s in samples) / len(samples),
        "quote/single_mean": sum(sum(1 for c in s if c in SINGLES) for s in samples) / len(samples),
        "quote/final_echo_frac": sum(1 for s in samples if re.search(r"final token", s, re.I)) / len(samples),
        "quote/est_penalty": COEF * sum(q) / len(q),
        "quote/n_samples": len(samples),
    }


def training_alive() -> bool:
    return subprocess.run(["pgrep", "-f", TRAIN_PGREP], capture_output=True).returncode == 0


def main() -> None:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nla-rl-quote-penalty"),
        entity=os.environ.get("WANDB_TEAM", "octahedral-systems"),
        group=os.environ.get("WANDB_GROUP", f"qwen2.5-7b-L20-v3qf-quotepen{COEF}"),
        name=os.environ.get("WANDB_NAME", "quote-stats"),
        job_type="quote-stats",
    )
    last_logged = -1
    idle = 0.0
    while True:
        step = latest_step()
        if step is not None and step > last_logged:
            stats = {**(dump_stats() or {}), **(jsonl_stats() or {})}
            if stats:
                run.log(stats, step=step)
                last_logged = step
                idle = 0.0
        if not training_alive():
            # one grace interval: catch the final dump after the launcher exits
            if idle > 0:
                break
            idle = INTERVAL
        time.sleep(INTERVAL)
    run.finish()
    print("quote-stats logger done", flush=True)


if __name__ == "__main__":
    main()
