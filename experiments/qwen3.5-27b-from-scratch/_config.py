"""Config loader shared by the python scripts in this experiment.

Reads NLA_RUN_CONFIG (default: config.env next to this file) as plain
KEY=value lines and applies them with os.environ.setdefault — values already
in the caller's environment win, matching the bash loader in _config.sh.
"""
import os
import sys


def load() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.environ.get("NLA_RUN_CONFIG", os.path.join(here, "config.env"))
    if not os.path.isfile(path):
        sys.exit(f"config file not found: {path} (set NLA_RUN_CONFIG)")
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())
    return path


def env(key: str, default: str | None = None) -> str:
    v = os.environ.get(key, default)
    if v is None or v == "":
        sys.exit(f"required config var {key} is unset — set it in the env or "
                 f"the NLA_RUN_CONFIG file")
    return v
