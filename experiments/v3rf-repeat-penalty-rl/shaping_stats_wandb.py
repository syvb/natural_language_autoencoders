"""Sidecar wandb logger for the v3rf run: repeat-coverage + quote metrics.

Same pattern as ../v3qf-quote-free-rl/quote_stats_wandb.py: every INTERVAL
seconds, log to wandb as a separate run in the same project/group
(job_type=shaping-stats), x = latest rollout step from train.log.

1. FULL-BATCH stats — consume new lines of NLA_QUOTE_STATS_JSONL (written by
   nla.reward per drain over EVERY scored sample; carries repeat_* keys when
   NLA_REPEAT_PENALTY > 0). n-weighted since the last log:
     repeatfull/covered_mean   copied chars per sample (the penalized measure)
     repeatfull/covered_max
     repeatfull/frac_zero      fraction with NO copied span >= MIN_CHARS
     repeatfull/est_penalty    NLA_REPEAT_PENALTY x covered_mean
     quotefull/chars_mean|chars_max|frac_zero   (quotes should stay OFF-penalty
                                                here; watch whether they drop
                                                anyway once content can't be
                                                verbatim)
2. DUMP stats (`quote/*`) — first-20-samples text dump, as in v3qf.

Usage (on the box, after the launcher starts):
  nohup python shaping_stats_wandb.py > /workspace/out/shaping_stats_sidecar.log 2>&1 &
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
JSONL = os.environ.get("NLA_QUOTE_STATS_JSONL", "/workspace/out/shaping_stats.jsonl")
TRAIN_LOG = os.environ.get("TRAIN_LOG", "/workspace/out/train.log")
COEF = float(os.environ.get("NLA_REPEAT_PENALTY", "0.03"))
INTERVAL = float(os.environ.get("INTERVAL", "30"))
TRAIN_PGREP = os.environ.get("TRAIN_PGREP", "run_rl_v3r[f].sh")

# THE penalized set — imported so the sidecar can never drift from the reward.
from nla.reward import _QUOTE_CHARS as QUOTES

_jsonl_pos = 0


def latest_step() -> int | None:
    try:
        txt = open(TRAIN_LOG, encoding="utf-8", errors="replace").read()[-200_000:]
    except OSError:
        return None
    hits = re.findall(r"perf (\d+)", txt.replace("\r", "\n"))
    return int(hits[-1]) if hits else None


def jsonl_stats() -> dict | None:
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
            pass
    if not recs:
        return None
    n = sum(r["n"] for r in recs)
    out = {
        "quotefull/chars_mean": sum(r["quote_chars_mean"] * r["n"] for r in recs) / n,
        "quotefull/chars_max": max(r["quote_chars_max"] for r in recs),
        "quotefull/frac_zero": sum(r["frac_zero"] * r["n"] for r in recs) / n,
        "quotefull/n_samples": n,
    }
    reps = [r for r in recs if "repeat_covered_mean" in r]
    if reps:
        rn = sum(r["n"] for r in reps)
        cov = sum(r["repeat_covered_mean"] * r["n"] for r in reps) / rn
        out.update({
            "repeatfull/covered_mean": cov,
            "repeatfull/covered_max": max(r["repeat_covered_max"] for r in reps),
            "repeatfull/frac_zero": sum(r["repeat_frac_zero"] * r["n"] for r in reps) / rn,
            "repeatfull/est_penalty": COEF * cov,
        })
    return out


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
        "quote/frac_zero": sum(1 for c in q if c == 0) / len(q),
        "quote/final_echo_frac": sum(1 for s in samples if re.search(r"final token", s, re.I)) / len(samples),
        "quote/n_samples": len(samples),
    }


def training_alive() -> bool:
    return subprocess.run(["pgrep", "-f", TRAIN_PGREP], capture_output=True).returncode == 0


def main() -> None:
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nla-rl-quote-penalty"),
        entity=os.environ.get("WANDB_TEAM", "octahedral-systems"),
        group=os.environ.get("WANDB_GROUP", f"qwen2.5-7b-L20-v3rf-repeatpen{COEF}"),
        name=os.environ.get("WANDB_NAME", "shaping-stats"),
        job_type="shaping-stats",
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
            if idle > 0:
                break
            idle = INTERVAL
        time.sleep(INTERVAL)
    run.finish()
    print("shaping-stats logger done", flush=True)


if __name__ == "__main__":
    main()
