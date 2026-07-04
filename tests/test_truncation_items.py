"""Tests for the v2 ITEM-mode truncation logic in nla.truncation:
  - split_into_items (the shared item-boundary definition),
  - sample_item_count (taper + curriculum + shared-per-group determinism),
  - item_truncation_cut (token-space cut at the K-th newline),
  - resolve_truncation_config in items mode.
"""

import pytest

from nla import truncation as t
from nla.truncation import TruncationConfig


# --------------------------------------------------------------------------- #
# split_into_items
# --------------------------------------------------------------------------- #

def test_split_basic():
    assert t.split_into_items("a\nb\nc") == ["a", "b", "c"]


def test_split_collapses_blank_lines_and_strips():
    # \n\n paragraphs collapse to items; surrounding whitespace stripped; blanks dropped
    assert t.split_into_items("  first  \n\n second \n\n\n third ") == ["first", "second", "third"]


def test_split_empty():
    assert t.split_into_items("") == []
    assert t.split_into_items("   \n  \n ") == []


# --------------------------------------------------------------------------- #
# sample_item_count
# --------------------------------------------------------------------------- #

def test_item_count_deterministic_and_in_range():
    for gi in range(500):
        k = t.sample_item_count(42, gi, 10, 2.0, 0)
        assert 1 <= k <= 10
    assert t.sample_item_count(42, 7, 10, 2.0, 0) == t.sample_item_count(42, 7, 10, 2.0, 0)


def test_item_count_shared_per_group():
    cfg = TruncationConfig(enabled=True, min_tokens=1, max_tokens=0, seed=42,
                           mode="items", max_items=10, taper_power=2.0, curriculum_groups=0)
    ks = {cfg.items_for_group(123) for _ in range(8)}
    assert len(ks) == 1  # all 8 group members get the same K
    assert cfg.items_for_group(123) != cfg.items_for_group(124) or True  # may differ across groups


def test_item_count_taper_biases_toward_few():
    # taper_power > 1 should put most mass on small K; taper_power == 1 ~ uniform.
    tapered = [t.sample_item_count(1, gi, 10, 3.0, 0) for gi in range(4000)]
    uniform = [t.sample_item_count(1, gi, 10, 1.0, 0) for gi in range(4000)]
    assert sum(tapered) / len(tapered) < sum(uniform) / len(uniform)
    assert min(tapered) == 1  # short end reachable


def test_item_count_endpoints_reachable():
    ks = {t.sample_item_count(0, gi, 5, 1.0, 0) for gi in range(2000)}
    assert 1 in ks and 5 in ks


def test_item_count_curriculum_starts_long_ends_short():
    # early groups (progress→0) are forced near max_items; late groups open up.
    early = [t.sample_item_count(7, gi, 10, 2.0, 1000) for gi in range(20)]
    late = [t.sample_item_count(7, gi, 10, 2.0, 1000) for gi in range(2000, 2020)]
    assert min(early) >= 9          # floor ~= max_items at the very start
    assert min(late) == 1           # floor has fully decayed past the horizon
    assert sum(early) / len(early) > sum(late) / len(late)


def test_item_count_single_item_max():
    assert all(t.sample_item_count(3, gi, 1, 2.0, 0) == 1 for gi in range(50))


# --------------------------------------------------------------------------- #
# item_truncation_cut
# --------------------------------------------------------------------------- #

def _char_decode_factory(id2ch):
    def decode(ids):
        return "".join(id2ch[i] for i in ids if i in id2ch)
    return decode


def test_cut_one_char_per_token():
    # response "aa\nbb\ncc" as 8 one-char tokens
    text = "aa\nbb\ncc"
    ids = list(range(len(text)))
    decode = _char_decode_factory(dict(enumerate(text)))
    assert t.item_truncation_cut(decode, ids, 1) == 2   # keep "aa"
    assert t.item_truncation_cut(decode, ids, 2) == 5   # keep "aa\nbb"
    assert t.item_truncation_cut(decode, ids, 3) == 8   # only 3 items, keep all
    assert t.item_truncation_cut(decode, ids, 9) == 8   # K > items: keep all


def test_cut_multichar_tokens():
    # tokens: ["aa", "\n", "cd", "\n", "ef"] -> "aa\ncd\nef"
    id2tok = {0: "aa", 1: "\n", 2: "cd", 3: "\n", 4: "ef"}
    ids = [0, 1, 2, 3, 4]
    decode = _char_decode_factory(id2tok)
    assert t.item_truncation_cut(decode, ids, 1) == 1   # keep ["aa"]
    assert t.item_truncation_cut(decode, ids, 2) == 3   # keep ["aa","\n","cd"]
    assert t.item_truncation_cut(decode, ids, 5) == 5   # K > items: keep all


def test_cut_empty():
    assert t.item_truncation_cut(lambda ids: "", [], 3) == 0


# --------------------------------------------------------------------------- #
# resolve_truncation_config — items mode
# --------------------------------------------------------------------------- #

def _args(**kw):
    import types
    return types.SimpleNamespace(**kw)


def test_resolve_items_mode_enabled(monkeypatch):
    for v in ("NLA_TRUNC_MODE", "NLA_TRUNC_MAX_ITEMS", "NLA_TRUNC_MAX_TOKENS"):
        monkeypatch.delenv(v, raising=False)
    cfg = t.resolve_truncation_config(
        _args(nla_trunc_mode="items", nla_trunc_max_items=10, nla_trunc_taper=2.0, rollout_seed=5)
    )
    assert cfg.enabled and cfg.mode == "items" and cfg.max_items == 10 and cfg.taper_power == 2.0


def test_resolve_items_mode_disabled_without_max_items(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_MAX_ITEMS", raising=False)
    cfg = t.resolve_truncation_config(_args(nla_trunc_mode="items", rollout_seed=0))
    assert cfg.enabled is False


def test_resolve_items_mode_via_env(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MODE", "items")
    monkeypatch.setenv("NLA_TRUNC_MAX_ITEMS", "8")
    monkeypatch.setenv("NLA_TRUNC_TAPER", "3.0")
    monkeypatch.setenv("NLA_TRUNC_CURRICULUM_GROUPS", "3200")
    cfg = t.resolve_truncation_config(_args(rollout_seed=1))
    assert cfg.enabled and cfg.mode == "items" and cfg.max_items == 8
    assert cfg.taper_power == 3.0 and cfg.curriculum_groups == 3200


def test_resolve_tokens_mode_still_default(monkeypatch):
    for v in ("NLA_TRUNC_MODE", "NLA_TRUNC_MAX_TOKENS", "NLA_TRUNC_MAX_ITEMS"):
        monkeypatch.delenv(v, raising=False)
    cfg = t.resolve_truncation_config(_args(rollout_seed=0))
    assert cfg.mode == "tokens" and cfg.enabled is False


def test_resolve_bad_mode_asserts(monkeypatch):
    monkeypatch.delenv("NLA_TRUNC_MODE", raising=False)
    with pytest.raises(AssertionError):
        t.resolve_truncation_config(_args(nla_trunc_mode="bogus", rollout_seed=0))
