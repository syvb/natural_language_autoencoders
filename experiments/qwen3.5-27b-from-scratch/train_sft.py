"""SFT launcher — stub sglang (SFT never starts an engine), then run miles train.py.

Loads the NLA sglang SFT stubs BEFORE any miles import pulls sglang, then
delegates to $MILES_DIR/train.py. Harmless when a real sglang is installed
(the SFT process never reaches engine code paths); required when it isn't.

If Ray workers crash on sglang imports on a box WITHOUT sglang installed,
also install the stub globally via a .pth (v3-era ENV_FIXES recipe):
    echo 'import nla._sglang_sft_stubs' > \
      $(python -c 'import site;print(site.getsitepackages()[0])')/zzz_nla_sglang_stub.pth
Not needed on the sglang-image boxes this runbook targets.
"""
import os

import nla._sglang_sft_stubs  # noqa: F401 — must precede any miles import that pulls sglang
import runpy

runpy.run_path(
    os.path.join(os.environ.get("MILES_DIR", "/workspace/miles"), "train.py"),
    run_name="__main__",
)
