"""Shared test fixtures + lightweight stubs.

`nla.reward` imports `ray` and a few `miles.*` modules at import time. Neither
is installed in the test environment (miles is upstream, GPU-only). We register
just-enough stub modules in sys.modules BEFORE those imports run so the pure
Python logic under test (the reward gate + open extraction) can be exercised
offline. The stubs are intentionally minimal — anything a test actually needs
(e.g. a real critic forward) is monkeypatched in the test itself.

Tests for the pure modules (`nla.truncation`, `nla.schema`) need none of this;
the stubs are harmless to them.
"""

import enum
import sys
import types

import pytest


class _SampleStatus(enum.Enum):
    PENDING = 0
    COMPLETED = 1
    TRUNCATED = 2
    FAILED = 3
    ABORTED = 4


class FakeSample:
    """Stand-in for miles.utils.types.Sample with the fields NLA touches."""

    Status = _SampleStatus

    def __init__(self, *, status, response="", activation_vector=None, group_index=None,
                 prompt=None, index=0, metadata=None):
        self.status = status
        self.response = response
        self.group_index = group_index
        self.prompt = prompt
        self.index = index
        self.metadata = metadata if metadata is not None else {}
        if activation_vector is not None:
            self.metadata["activation_vector"] = activation_vector
        self.multimodal_train_inputs = None


def _install_miles_stubs():
    if "ray" not in sys.modules:
        sys.modules["ray"] = types.ModuleType("ray")

    # miles.utils.types.Sample
    if "miles" not in sys.modules:
        sys.modules["miles"] = types.ModuleType("miles")
        sys.modules["miles"].__path__ = []
    for name in ("miles.utils", "miles.utils.types", "miles.utils.processing_utils"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod

    class _Sample:
        Status = _SampleStatus

    sys.modules["miles.utils.types"].Sample = _Sample

    def _load_tokenizer(*a, **k):  # pragma: no cover - replaced per-test
        raise RuntimeError("stub load_tokenizer should be monkeypatched in the test")

    sys.modules["miles.utils.processing_utils"].load_tokenizer = _load_tokenizer

    # Extra miles modules imported by nla.rollout.nla_generate. The attributes
    # only need to EXIST for `from X import name` to succeed at import time —
    # tests monkeypatch the bound names on nla_generate directly.
    extra = {
        "miles.rollout": [],
        "miles.rollout.generate_utils": [],
        "miles.rollout.generate_utils.generate_endpoint_utils": [
            "compute_request_payload", "update_sample_from_response",
        ],
        "miles.rollout.inference_rollout": [],
        "miles.rollout.inference_rollout.inference_rollout_train": ["get_worker_urls"],
        "miles.utils.http_utils": ["post"],
    }
    for name, attrs in extra.items():
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []
            sys.modules[name] = mod
        for attr in attrs:
            setattr(sys.modules[name], attr, _placeholder)


def _placeholder(*a, **k):  # pragma: no cover - import-time stand-in only
    raise RuntimeError("miles stub called directly — monkeypatch it in the test")


_install_miles_stubs()


@pytest.fixture
def Status():
    return _SampleStatus


@pytest.fixture
def make_sample():
    return FakeSample
