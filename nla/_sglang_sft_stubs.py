"""Stub sglang modules so SFT can run without a full sglang+router install.

SFT (--debug-train-only) never touches the sglang engine — it's pure FSDP
training on pre-generated parquets. But miles' top-level imports pull in
sglang, sglang_router, and transitive deps with tight version pins (e.g.
sglang 0.5.x → transformers 4.57+ for GptOssConfig). If your environment has
an older transformers and you can't upgrade, this lets SFT run anyway.

**No-op when a real sglang is importable** (e.g. the lmsysorg/sglang training
images): there the genuine modules must be used — shadowing a working install
with stubs breaks miles' argparse chain (`parser = add_sglang_arguments(parser)`
expects the real function's return value) and any code that touches real
constants.

These stubs satisfy the import chain with no-ops. The actual sglang engine
code paths (rollout generation, router) are unreachable in SFT and will
crash loudly if somehow hit. That's the intended failure mode — you'll know
immediately if you accidentally try to use this for RL without a real env.

KNOWN LIMITATION on sglang-less environments with newer miles pins: miles may
read `args.sglang_*` attributes that these stubs never register, and its
argparse chain reassigns the parser from `add_sglang_arguments`'s return
value. If SFT crashes in `parse_args`, complete the stub for your pin:
`grep -rhoE "args\\.sglang[a-z_]+" miles/` and register each with a benign
default (sizes=1, flags=False), returning the parser. The supported path is
simply running on an image that has sglang installed.

Usage: in your launcher, before importing train.py:
    import nla._sglang_sft_stubs  # noqa: F401 — sets up sys.modules
"""

import importlib.util
import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


def _install() -> None:
    # ─── sglang.srt.constants — string constants, used as offload tags ───
    # (real values don't matter for SFT; these just need to be hashable)
    _srt = _stub_module("sglang.srt")
    _stub_module("sglang").srt = _srt
    _constants = _stub_module("sglang.srt.constants")
    _constants.GPU_MEMORY_TYPE_CUDA_GRAPH = "cuda_graph"
    _constants.GPU_MEMORY_TYPE_KV_CACHE = "kv_cache"
    _constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    _srt.constants = _constants

    # ─── sglang_router.launch_router.RouterArgs ───
    # add_cli_args is called unconditionally in miles.utils.arguments.add_router_arguments.
    # Only stub when the real sglang-router isn't installed — setup_box installs
    # it via pip even on sglang-less boxes, and shadowing it turns valid
    # --router-* flags into argparse errors.
    if importlib.util.find_spec("sglang_router") is None:
        _router = _stub_module("sglang_router")
        _router.__version__ = "0.0.0-nla-stub"
        _launch = _stub_module("sglang_router.launch_router")
        _router.launch_router = _launch

        class _RouterArgs:
            @staticmethod
            def add_cli_args(parser, use_router_prefix=True, exclude_host_port=True):  # noqa: ARG004
                return parser

        _launch.RouterArgs = _RouterArgs

    # ─── miles.backends.sglang_utils.* — these import sglang engine internals ───
    # We replace the whole submodules with stubs. SGLangEngine is wrapped in
    # ray.remote() inside _create_rollout_engines — SFT never calls that.
    _sglang_utils = _stub_module("miles.backends.sglang_utils")
    _engine_mod = _stub_module("miles.backends.sglang_utils.sglang_engine")
    _args_mod = _stub_module("miles.backends.sglang_utils.arguments")

    class _SGLangEngine:
        def __init__(self, *_, **__):
            raise RuntimeError(
                "SGLangEngine stub — you're trying to run RL rollout but only have "
                "the SFT stubs loaded. Build the miles conda env (build_conda.sh)."
            )

    _engine_mod.SGLangEngine = _SGLangEngine
    _sglang_utils.sglang_engine = _engine_mod

    def _add_sglang_arguments(parser):
        return parser

    def _sglang_validate_args(args):  # noqa: ARG001
        pass

    _args_mod.add_sglang_arguments = _add_sglang_arguments
    _args_mod.validate_args = _sglang_validate_args
    _sglang_utils.arguments = _args_mod


def _sglang_importable() -> bool:
    """Probe actual importability, not just installability: the scenario this
    module exists for is sglang INSTALLED but raising on import (e.g. its
    transformers pin unmet) — find_spec succeeds there and would wrongly skip
    the stubs. A partially-imported sglang left in sys.modules by the failed
    probe is fine: _stub_module overwrites those entries."""
    try:
        import sglang  # noqa: F401

        return True
    except Exception:
        return False


if not _sglang_importable():
    _install()
