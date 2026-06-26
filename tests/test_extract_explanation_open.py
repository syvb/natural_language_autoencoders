"""Tests for nla.schema.extract_explanation_open — the close-tag-tolerant
extractor used by RL truncation. Key property: a drop-in for
extract_explanation on COMPLETE responses, but it also recovers content from a
truncated (unclosed) response.
"""

import pytest

from nla.schema import (
    EXPLANATION_CLOSE,
    EXPLANATION_OPEN,
    extract_explanation,
    extract_explanation_open,
    wrap_explanation,
)


def test_complete_response_matches_legacy():
    r = wrap_explanation("hello world")
    assert extract_explanation_open(r) == "hello world"
    assert extract_explanation_open(r) == extract_explanation(r)


def test_truncated_no_close_tag():
    # the common RL-truncation case: opening tag, content, cut off mid-stream
    r = f"{EXPLANATION_OPEN}\nfirst snippet about the topic\nsecond snippet abou"
    assert extract_explanation_open(r) == "first snippet about the topic\nsecond snippet abou"


def test_truncated_returns_content_legacy_returns_none():
    r = f"{EXPLANATION_OPEN}\npartial content"
    assert extract_explanation(r) is None          # legacy needs both tags
    assert extract_explanation_open(r) == "partial content"


def test_missing_opening_tag_is_none():
    assert extract_explanation_open("no tags at all") is None
    assert extract_explanation_open(f"stuff {EXPLANATION_CLOSE} more") is None


def test_empty_content_is_none():
    assert extract_explanation_open(EXPLANATION_OPEN) is None
    assert extract_explanation_open(f"{EXPLANATION_OPEN}\n   \n") is None
    assert extract_explanation_open(f"{EXPLANATION_OPEN}{EXPLANATION_CLOSE}") is None


def test_strips_surrounding_whitespace():
    r = f"{EXPLANATION_OPEN}\n   spaced out   \n{EXPLANATION_CLOSE}"
    assert extract_explanation_open(r) == "spaced out"


def test_multiline_content_preserved_internally():
    body = "line one\nline two\nline three"
    r = f"{EXPLANATION_OPEN}\n{body}\n{EXPLANATION_CLOSE}"
    assert extract_explanation_open(r) == body


def test_takes_first_pair_when_multiple():
    r = (
        f"{EXPLANATION_OPEN}first{EXPLANATION_CLOSE}"
        f"{EXPLANATION_OPEN}second{EXPLANATION_CLOSE}"
    )
    # open->first close, matching extract_explanation's non-greedy first match
    assert extract_explanation_open(r) == "first"
    assert extract_explanation_open(r) == extract_explanation(r)


def test_content_with_angle_brackets():
    r = f"{EXPLANATION_OPEN}\na < b and c > d, x<y\n{EXPLANATION_CLOSE}"
    assert extract_explanation_open(r) == "a < b and c > d, x<y"


def test_leading_text_before_open_ignored():
    r = f"Here is the answer. {EXPLANATION_OPEN}\nthe content\n{EXPLANATION_CLOSE}"
    assert extract_explanation_open(r) == "the content"


def test_close_before_open_treated_as_open_to_end():
    # a stray close tag BEFORE the real open: we anchor on the first open tag,
    # then look for a close AFTER it (none) -> content to end.
    r = f"junk {EXPLANATION_CLOSE} {EXPLANATION_OPEN}\nreal content"
    assert extract_explanation_open(r) == "real content"


@pytest.mark.parametrize("body", [
    "single",
    "two words",
    "snippet 1\nsnippet 2\nsnippet 3",
    "  pad  ",
    "symbols !@#$%^&*()",
])
def test_backward_compat_property_on_complete(body):
    """For any non-empty body, the open extractor equals the legacy extractor
    on a fully-formed (closed) response."""
    r = wrap_explanation(body)
    assert extract_explanation_open(r) == extract_explanation(r)
