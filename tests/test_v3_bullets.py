"""Tests for the v3 pieces: bullets explanation format, token-based AR
warm-start truncation (stage3_build), and the format-aware tokens-mode
opening offset (truncation.resolve_opening_offset).
"""

from nla.datagen.stage3_build import (
    _ACTOR_TEMPLATES,
    _format_items,
    _maybe_truncate_to_tokens,
)
from nla.schema import EXPLANATION_OPEN
from nla.truncation import resolve_opening_offset


class _Tok:
    """Whitespace 'tokenizer' — 1 token per whitespace-separated word."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}

    def encode(self, text, add_special_tokens=False):
        return self(text)["input_ids"]

    def decode(self, ids):
        # pair with __call__: decoding a k-prefix returns the first k words
        self._last = ids
        return " ".join(self._src.split()[: len(ids)])


def _tok_for(text):
    tok = _Tok()
    tok._src = text
    return tok


# --------------------------------------------------------------------------- #
# _format_items
# --------------------------------------------------------------------------- #

EXPL = "First feature.\n\nSecond: lists, code.\n\nFinal constraint."


def test_format_tagged_is_identity():
    assert _format_items(EXPL, "tagged") == EXPL


def test_format_list_collapses_paragraphs():
    assert _format_items(EXPL, "list") == (
        "First feature.\nSecond: lists, code.\nFinal constraint."
    )


def test_format_bullets_is_plain_lines_like_list():
    # bullets (v3) differs from list (v2) only in the PROMPT wording — the
    # output text is identical unmarked lines (no "- " chars eating the
    # truncation budget).
    assert _format_items(EXPL, "bullets") == _format_items(EXPL, "list")
    assert "- " not in _format_items(EXPL, "bullets")


# --------------------------------------------------------------------------- #
# _maybe_truncate_to_tokens
# --------------------------------------------------------------------------- #

def test_token_truncation_disabled_when_max_nonpositive():
    assert _maybe_truncate_to_tokens(EXPL, 0, 1, 0, 0, _tok_for(EXPL)) == EXPL


def test_token_truncation_noop_when_budget_covers_text():
    short = "- one\n- two"
    assert _maybe_truncate_to_tokens(short, 0, 120, 120, 0, _tok_for(short)) == short


def test_token_truncation_deterministic_per_row():
    text = "- " + " ".join(f"w{i}" for i in range(200))
    tok = _tok_for(text)
    a = _maybe_truncate_to_tokens(text, 7, 1, 120, 0, tok)
    b = _maybe_truncate_to_tokens(text, 7, 1, 120, 0, tok)
    assert a == b
    # different rows draw independently — over many rows the outputs must vary
    assert len({_maybe_truncate_to_tokens(text, r, 1, 120, 0, tok) for r in range(20)}) > 1


def test_token_truncation_lengths_span_range():
    text = "- " + " ".join(f"w{i}" for i in range(300))
    tok = _tok_for(text)
    lens = {
        len(_maybe_truncate_to_tokens(text, r, 1, 120, 0, tok).split())
        for r in range(300)
    }
    assert min(lens) == 1          # 1-token prefixes must actually occur
    assert max(lens) == 120        # the full budget is reachable
    assert len(lens) > 50          # ...and it's spread, not clustered


def test_token_truncation_is_prefix_and_stripped():
    text = "- alpha beta\n- gamma delta epsilon"
    tok = _tok_for(text)
    out = _maybe_truncate_to_tokens(text, 3, 2, 2, 0, tok)
    assert text.startswith(out)
    assert out == out.strip()


# --------------------------------------------------------------------------- #
# resolve_opening_offset — tagged template gets the tag length, untagged gets 0
# --------------------------------------------------------------------------- #

def test_offset_zero_for_untagged_templates():
    tok = _tok_for("")
    for fmt in ("list", "bullets"):
        assert EXPLANATION_OPEN not in _ACTOR_TEMPLATES[fmt]
        assert resolve_opening_offset(_ACTOR_TEMPLATES[fmt], tok) == 0


def test_offset_positive_for_tagged_template():
    tok = _tok_for("")
    assert EXPLANATION_OPEN in _ACTOR_TEMPLATES["tagged"]
    assert resolve_opening_offset(_ACTOR_TEMPLATES["tagged"], tok) >= 1


def test_offset_env_override(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_OPENING_OFFSET", "9")
    tok = _tok_for("")
    assert resolve_opening_offset(_ACTOR_TEMPLATES["bullets"], tok) == 9
    assert resolve_opening_offset(_ACTOR_TEMPLATES["tagged"], tok) == 9


# --------------------------------------------------------------------------- #
# templates — the v3 prompt says bullets and keeps the injection context intact
# --------------------------------------------------------------------------- #

def test_bullets_template_wording_and_injection_context():
    t = _ACTOR_TEMPLATES["bullets"]
    assert "bullet" in t
    assert '"- "' not in t          # prompt must not demand literal markers
    assert "<explanation>" not in t
    # identical <concept> wrapping across formats → same injection neighbors
    for fmt in ("tagged", "list"):
        assert "<concept>{injection_char}</concept>" in _ACTOR_TEMPLATES[fmt]
    assert "<concept>{injection_char}</concept>" in t


def test_keep_full_frac_leaves_a_deterministic_slice_untruncated():
    text = "- " + " ".join(f"w{i}" for i in range(300))
    tok = _tok_for(text)
    full = sum(
        _maybe_truncate_to_tokens(text, r, 1, 120, 0, tok, keep_full_frac=0.15) == text
        for r in range(400)
    )
    assert 30 <= full <= 90  # ~15% of 400, generous tolerance
    # deterministic: same rows keep-full on a second pass
    a = [_maybe_truncate_to_tokens(text, r, 1, 120, 0, tok, keep_full_frac=0.15) for r in range(50)]
    b = [_maybe_truncate_to_tokens(text, r, 1, 120, 0, tok, keep_full_frac=0.15) for r in range(50)]
    assert a == b
    # frac 0 → nothing kept full (text is 301 tokens > 120)
    assert all(
        _maybe_truncate_to_tokens(text, r, 1, 120, 0, tok, keep_full_frac=0.0) != text
        for r in range(50)
    )
