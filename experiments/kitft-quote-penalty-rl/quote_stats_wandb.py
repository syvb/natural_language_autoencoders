"""Sidecar wandb logger: quote-usage metrics from the live rollout text dumps.

Runs NEXT TO training (no code change to the run): every INTERVAL seconds,
parse the newest NLA_ROLLOUT_TEXT_DUMP snapshot (first ~20 samples of a reward
batch) + the latest rollout step from train.log, and log per-sample quote stats
to wandb as a separate run in the same project/group (job_type=quote-stats).

Logged (x = rollout step):
  quote/chars_mean|median|max   quote-mark chars per sample (the penalized set)
  quote/frac_zero               fraction of samples with NO quote chars
  quote/double_mean             double-quote-family chars per sample
  quote/single_mean             single-quote/apostrophe-family chars per sample
  quote/final_echo_frac         fraction of samples containing "final token"
                                (the last-token echo habit, quoted or not)
  quote/est_penalty             NLA_QUOTE_PENALTY x chars_mean
  quote/n_samples               dump size (context for noise)

Usage (on the box):
  nohup python quote_stats_wandb.py > /workspace/out/quote_stats.log 2>&1 &
Exits when the training launcher disappears.
"""
import os
import re
import statistics
import subprocess
import time

import wandb

DUMP = os.environ.get("NLA_ROLLOUT_TEXT_DUMP", "/workspace/out/rollout_dump.txt")
TRAIN_LOG = os.environ.get("TRAIN_LOG", "/workspace/out/train.log")
COEF = float(os.environ.get("NLA_QUOTE_PENALTY", "0.1"))
INTERVAL = float(os.environ.get("INTERVAL", "30"))
TRAIN_PGREP = os.environ.get("TRAIN_PGREP", "run_rl_quotepe[n].sh")

# Keep in sync with nla.reward._QUOTE_CHARS.
QUOTES = frozenset("\"'`‘’‚‛“”„‟«»‹›「」『』〝〞〟＂＇｀｢｣❛❜❝❞⹂")
DOUBLES = frozenset('"“”„‟«»＂❝❞⹂')
SINGLES = frozenset("'‘’‚‛＇❛❜")


def latest_step() -> int | None:
    try:
        txt = open(TRAIN_LOG, encoding="utf-8", errors="replace").read()[-200_000:]
    except OSError:
        return None
    hits = re.findall(r"perf (\d+)", txt.replace("\r", "\n"))
    return int(hits[-1]) if hits else None


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
        group=os.environ.get("WANDB_GROUP", "kitft-7b-L20-quotepen0.1"),
        name=os.environ.get("WANDB_NAME", "quote-stats"),
        job_type="quote-stats",
    )
    last_logged = -1
    idle = 0.0
    while True:
        step = latest_step()
        if step is not None and step > last_logged:
            stats = dump_stats()
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
