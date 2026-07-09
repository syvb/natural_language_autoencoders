"""SFT launcher — stub sglang (SFT never starts an engine), then run miles train.py.

Place this in the miles checkout root (next to train.py). It loads the NLA sglang SFT
stubs BEFORE any miles import pulls sglang, then delegates to train.py.

NOTE: the global stub also needs to load in Ray worker subprocesses (which run miles
imports directly). See ENV_FIXES.md — install a one-line `.pth` in site-packages:
    echo 'import nla._sglang_sft_stubs' > $(python -c 'import site;print(site.getsitepackages()[0])')/zzz_nla_sglang_stub.pth
"""
import nla._sglang_sft_stubs  # noqa: F401 — must precede any miles import that pulls sglang
import runpy
runpy.run_path("/workspace/miles/train.py", run_name="__main__")
