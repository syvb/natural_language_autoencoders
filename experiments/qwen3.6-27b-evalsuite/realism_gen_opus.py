"""Same recursive-realism rewrite chain as realism_gen.py, but the rewriter is
Opus 4.8 driven through the Claude Code CLI (`claude --system-prompt ... -p ...`)
instead of the OpenRouter API. Lets us compare how two different strong models
escalate "realism" from the same seed (realism_v0.json).

Reuses the exact prompts + parser from realism_gen so the only variable is the
rewriter model. Writes realism_versions_opus.json; resumable the same way.

    python3 realism_gen_opus.py                 # v1..v20
    python3 realism_gen_opus.py --target 20 --restart

No API key needed — the CLI uses the local Claude Code auth. Runs each rewrite in
a scratch cwd so the target repo's CLAUDE.md / tools don't leak into the rewrite.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import realism_gen as R  # noqa: E402  (SYS, build_user_msg, parse, SEED, TARGET)

MODEL = "claude-opus-4-8"
OUT = HERE / "realism_versions_opus.json"
SCRATCH = Path("/home/debian/.claude/jobs/f04fd6ed/tmp")  # neutral cwd, no CLAUDE.md


def call_opus(prev, rename):
    user_msg = R.build_user_msg(prev, rename)
    cmd = ["claude", "--model", MODEL,
           "--exclude-dynamic-system-prompt-sections",
           "--system-prompt", R.SYS,
           "-p", user_msg]
    last = None
    for attempt in range(5):
        try:
            p = subprocess.run(cmd, cwd=str(SCRATCH), capture_output=True,
                               text=True, timeout=600)
            if p.returncode == 0 and p.stdout.strip():
                return R.parse(p.stdout)
            last = f"rc={p.returncode} stderr={p.stderr[:300]!r} stdout={p.stdout[:200]!r}"
        except Exception as e:
            last = repr(e)
        time.sleep(min(5 * (attempt + 1), 30))
    raise RuntimeError(f"claude CLI failed after retries: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=R.TARGET)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    v0 = json.load(open(R.SEED))
    if OUT.exists() and not args.restart:
        versions = json.load(open(OUT))["versions"]
        print(f"[resume] {len(versions)-1}/{args.target} rewrites done", flush=True)
    else:
        versions = [{"i": 0, "renamed": False, "system": v0["system"], "user": v0["user"]}]

    while len(versions) - 1 < args.target:
        i = len(versions)
        rename = (i == 1)
        prev = versions[-1]
        print(f"[gen] v{i}  (rename={rename})  from v{i-1} "
              f"(sys={len(prev['system'])} usr={len(prev['user'])})", flush=True)
        nv = call_opus(prev, rename)
        versions.append({"i": i, "renamed": rename, **nv})
        json.dump({"model": MODEL, "backend": "claude-cli", "seed": "realism_v0.json",
                   "versions": versions}, open(OUT, "w"), indent=1)
        print(f"  -> v{i} sys={len(nv['system'])} usr={len(nv['user'])} chars  [saved {OUT.name}]",
              flush=True)

    print(f"[done] {len(versions)-1} rewrites in {OUT.name}", flush=True)


if __name__ == "__main__":
    main()
