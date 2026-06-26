"""Tests for nla.truncation — random-length ("info-upfront") RL truncation.

Covers the three baked-in decisions:
  - shared-per-group length (the GRPO-cleanliness property),
  - content-only counting (opening-tag offset + max_new_tokens math),
  - config resolution (CLI > env > default, disabled by default).
"""

import subprocess
import sys
import types

import pytest

from nla import truncation as t
from nla.schema import EXPLANATION_OPEN


# --------------------------------------------------------------------------- #
# sample_truncation_length: determinism, shared-per-group, range, uniformity
# --------------------------------------------------------------------------- #

def test_length_is_deterministic():
    a = t.sample_truncation_length(42, 7, 1, 130)
    b = t.sample_truncation_length(42, 7, 1, 130)
    assert a == b


def test_length_within_inclusive_range():
    for gi in range(2000):
        L = t.sample_truncation_length(1234, gi, 1, 130)
        assert 1 <= L <= 130


def test_length_endpoints_are_reachable():
    # both lo and hi must be attainable (randint is inclusive on both ends)
    lo_hits = any(t.sample_truncation_length(0, gi, 1, 5) == 1 for gi in range(500))
    hi_hits = any(t.sample_truncation_length(0, gi, 1, 5) == 5 for gi in range(500))
    assert lo_hits and hi_hits


def test_length_varies_across_groups():
    vals = {t.sample_truncation_length(42, gi, 1, 130) for gi in range(50)}
    # 50 independent draws over a 130-wide range should not collapse to a point
    assert len(vals) > 20


def test_length_changes_with_seed():
    a = [t.sample_truncation_length(1, gi, 1, 130) for gi in range(64)]
    b = [t.sample_truncation_length(2, gi, 1, 130) for gi in range(64)]
    assert a != b


def test_length_single_value_range():
    # degenerate but legal: lo == hi
    for gi in range(20):
        assert t.sample_truncation_length(9, gi, 7, 7) == 7


def test_length_roughly_uniform():
    # crude uniformity check: counts per bin shouldn't be wildly skewed
    lo, hi, n = 1, 10, 20000
    counts = [0] * (hi - lo + 1)
    for gi in range(n):
        counts[t.sample_truncation_length(777, gi, lo, hi) - lo] += 1
    expected = n / (hi - lo + 1)
    # every bin within ±25% of expected — generous, just catches gross bias
    assert all(0.75 * expected < c < 1.25 * expected for c in counts), counts


def test_bad_range_asserts():
    with pytest.raises(AssertionError):
        t.sample_truncation_length(0, 0, 0, 10)   # lo < 1
    with pytest.raises(AssertionError):
        t.sample_truncation_length(0, 0, 5, 4)    # hi < lo


def test_length_independent_of_pythonhashseed():
    """The draw must not depend on PYTHONHASHSEED (every rollout worker has to
    agree on a group's length). random.Random(str) routes through SHA-512, not
    hash(), so this holds — verify it in fresh subprocesses with different seeds.
    """
    snippet = (
        "from nla.truncation import sample_truncation_length as f;"
        "print([f(42, gi, 1, 130) for gi in range(16)])"
    )
    out = []
    for hashseed in ("0", "1", "12345"):
        env = {**_base_env(), "PYTHONHASHSEED": hashseed}
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr
        out.append(r.stdout.strip())
    assert out[0] == out[1] == out[2], out


def _base_env():
    import os
    env = dict(os.environ)
    # ensure the repo root is importable in the subprocess
    repo_root = subprocess.run(
        [sys.executable, "-c", "import nla, os; print(os.path.dirname(os.path.dirname(nla.__file__)))"],
        capture_output=True, text=True,
    ).stdout.strip()
    env["PYTHONPATH"] = repo_root + ":" + env.get("PYTHONPATH", "")
    return env


# --------------------------------------------------------------------------- #
# TruncationConfig.length_for_group: the shared-per-group guarantee
# --------------------------------------------------------------------------- #

def test_length_for_group_shared_within_group():
    cfg = t.TruncationConfig(enabled=True, min_tokens=1, max_tokens=130, seed=42)
    # simulate 8 samples of the SAME prompt (same group_index) — all identical
    gi = 314
    lengths = [cfg.length_for_group(gi) for _ in range(8)]
    assert len(set(lengths)) == 1
    # a different prompt (different group_index) generally differs
    other = cfg.length_for_group(gi + 1)
    assert lengths[0] == t.sample_truncation_length(42, gi, 1, 130)
    assert other == t.sample_truncation_length(42, gi + 1, 1, 130)


# --------------------------------------------------------------------------- #
# max_new_tokens_for_content: offset + cap + floor
# --------------------------------------------------------------------------- #

def test_max_new_tokens_adds_offset():
    assert t.max_new_tokens_for_content(10, 3, 150) == 13


def test_max_new_tokens_clamped_to_hard_cap():
    assert t.max_new_tokens_for_content(200, 3, 150) == 150
    assert t.max_new_tokens_for_content(148, 3, 150) == 150  # 151 -> capped


def test_max_new_tokens_floor_is_one():
    assert t.max_new_tokens_for_content(0, 0, 150) == 1
    assert t.max_new_tokens_for_content(1, 0, 150) == 1


def test_max_new_tokens_no_hard_cap():
    assert t.max_new_tokens_for_content(130, 3, None) == 133


# --------------------------------------------------------------------------- #
# opening_tag_token_len
# --------------------------------------------------------------------------- #

class _WordTokenizer:
    """Whitespace tokenizer — enough to exercise opening_tag_token_len."""
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def test_opening_tag_offset_counts_opening_prefix():
    tok = _WordTokenizer()
    # OPENING_PREFIX is "<explanation>\n"; whitespace-split -> one token
    assert t.opening_tag_token_len(tok) == len(t.OPENING_PREFIX.split())
    assert t.OPENING_PREFIX == EXPLANATION_OPEN + "\n"


def test_opening_tag_offset_passes_no_special_tokens():
    seen = {}

    class _Rec:
        def encode(self, text, add_special_tokens=False):
            seen["add_special_tokens"] = add_special_tokens
            seen["text"] = text
            return [1, 2, 3]

    assert t.opening_tag_token_len(_Rec()) == 3
    assert seen["add_special_tokens"] is False
    assert seen["text"] == t.OPENING_PREFIX


# --------------------------------------------------------------------------- #
# resolve_truncation_config: CLI > env > default, disabled by default
# --------------------------------------------------------------------------- #

def _args(**kw):
    return types.SimpleNamespace(**kw)


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_MAX_TOKENS", raising=False)
    cfg = t.resolve_truncation_config(_args(rollout_seed=0))
    assert cfg.enabled is False
    assert cfg.max_tokens == 0


def test_enabled_via_cli_arg(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_MAX_TOKENS", raising=False)
    cfg = t.resolve_truncation_config(
        _args(nla_trunc_max_tokens=130, nla_trunc_min_tokens=1, rollout_seed=5)
    )
    assert cfg.enabled and cfg.min_tokens == 1 and cfg.max_tokens == 130
    assert cfg.seed == 5  # falls back to rollout_seed


def test_enabled_via_env(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MAX_TOKENS", "130")
    monkeypatch.setenv("NLA_TRUNC_MIN_TOKENS", "1")
    cfg = t.resolve_truncation_config(_args(rollout_seed=7))
    assert cfg.enabled and cfg.max_tokens == 130 and cfg.min_tokens == 1
    assert cfg.seed == 7


def test_cli_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MAX_TOKENS", "50")
    cfg = t.resolve_truncation_config(_args(nla_trunc_max_tokens=130, rollout_seed=0))
    assert cfg.max_tokens == 130  # CLI wins


def test_seed_override(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_SEED", raising=False)
    cfg = t.resolve_truncation_config(
        _args(nla_trunc_max_tokens=10, nla_trunc_seed=999, rollout_seed=1)
    )
    assert cfg.seed == 999
    monkeypatch.setenv("NLA_TRUNC_SEED", "321")
    cfg2 = t.resolve_truncation_config(_args(nla_trunc_max_tokens=10, rollout_seed=1))
    assert cfg2.seed == 321


def test_empty_env_treated_as_unset(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MAX_TOKENS", "")
    cfg = t.resolve_truncation_config(_args(rollout_seed=0))
    assert cfg.enabled is False


def test_bad_range_asserts_at_resolve(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_MAX_TOKENS", raising=False)
    with pytest.raises(AssertionError):
        t.resolve_truncation_config(
            _args(nla_trunc_max_tokens=10, nla_trunc_min_tokens=20, rollout_seed=0)
        )


def test_missing_rollout_seed_defaults_zero(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_SEED", raising=False)
    cfg = t.resolve_truncation_config(_args(nla_trunc_max_tokens=10))
    assert cfg.seed == 0
