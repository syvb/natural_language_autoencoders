"""Wiring test for nla.reward._prep_batch under random-length truncation.

Validates the two reward-side behavioural changes:
  - when truncation is ON, TRUNCATED samples are SCORED (not skipped/penalised),
  - the open extractor recovers content from a response with no closing tag.

`ray` / `miles.*` are stubbed in conftest so the module imports offline; the
tokenizer and critic config are faked here (we only assert which samples are
selected and what critic text they produce — not the critic forward).
"""

import torch

import nla.reward as reward
from nla.truncation import TruncationConfig

Status = reward.Sample.Status


class _RecordingTokenizer:
    """Records the prompt strings it was asked to tokenize; returns dummy ids."""
    def __init__(self):
        self.calls = []

    def __call__(self, prompts, add_special_tokens=True, padding=True, return_tensors="pt"):
        self.calls.append(list(prompts))
        b, tlen = len(prompts), 4
        return {
            "input_ids": torch.zeros(b, tlen, dtype=torch.long),
            "attention_mask": torch.ones(b, tlen, dtype=torch.long),
        }


class _FakeCfg:
    critic_prompt_template = "Summary of the following text: <text>{explanation}</text> <summary>"
    mse_scale = 59.87


class _Sample:
    def __init__(self, status, response, vec):
        self.status = status
        self.response = response
        self.metadata = {"activation_vector": vec}


def _setup(monkeypatch, *, enabled):
    tok = _RecordingTokenizer()
    monkeypatch.setattr(reward, "_TOKENIZER", tok)
    monkeypatch.setattr(reward, "_CFG", _FakeCfg())
    monkeypatch.setattr(
        reward, "_TRUNC",
        TruncationConfig(enabled=enabled, min_tokens=1, max_tokens=130, seed=0),
    )
    return tok


def _vec():
    return [0.1, 0.2, 0.3, 0.4]


def test_enabled_scores_completed_and_truncated(monkeypatch):
    _setup(monkeypatch, enabled=True)
    samples = [
        _Sample(Status.COMPLETED, "<explanation>\ncomplete one\n</explanation>", _vec()),  # 0
        _Sample(Status.TRUNCATED, "<explanation>\ntruncated mid-sen", _vec()),             # 1
        _Sample(Status.FAILED, "garbage", _vec()),                                          # 2
    ]
    payload, orig_idx = reward._prep_batch(samples)
    assert orig_idx == [0, 1]          # both completed AND truncated scored
    assert payload is not None


def test_disabled_scores_only_completed(monkeypatch):
    _setup(monkeypatch, enabled=False)
    samples = [
        _Sample(Status.COMPLETED, "<explanation>\ncomplete one\n</explanation>", _vec()),  # 0
        _Sample(Status.TRUNCATED, "<explanation>\ntruncated mid-sen", _vec()),             # 1
    ]
    payload, orig_idx = reward._prep_batch(samples)
    assert orig_idx == [0]             # legacy: truncated skipped


def test_failed_and_pending_always_skipped(monkeypatch):
    _setup(monkeypatch, enabled=True)
    samples = [
        _Sample(Status.FAILED, "<explanation>\nhas tags but failed\n</explanation>", _vec()),
        _Sample(Status.PENDING, "<explanation>\npending\n</explanation>", _vec()),
        _Sample(Status.ABORTED, "<explanation>\naborted\n</explanation>", _vec()),
    ]
    payload, orig_idx = reward._prep_batch(samples)
    assert orig_idx == []
    assert payload is None


def test_extraction_miss_skipped_even_if_completed(monkeypatch):
    _setup(monkeypatch, enabled=True)
    samples = [
        _Sample(Status.COMPLETED, "no explanation tag here at all", _vec()),   # miss
        _Sample(Status.TRUNCATED, "<explanation>\ngood content", _vec()),      # ok
    ]
    payload, orig_idx = reward._prep_batch(samples)
    assert orig_idx == [1]


def test_open_extraction_feeds_critic_content_without_close_tag(monkeypatch):
    tok = _setup(monkeypatch, enabled=True)
    samples = [
        _Sample(Status.TRUNCATED, "<explanation>\nimportant info first, then more deta", _vec()),
    ]
    reward._prep_batch(samples)
    # the critic prompt is built from the open-extracted content (no close tag)
    assert tok.calls, "tokenizer was never called"
    built = tok.calls[0]
    assert built == [
        "Summary of the following text: <text>important info first, then more deta</text> <summary>"
    ]


def test_empty_batch_returns_none(monkeypatch):
    _setup(monkeypatch, enabled=True)
    payload, orig_idx = reward._prep_batch([])
    assert payload is None and orig_idx == []


def test_mse_to_reward_nan_guard():
    import math
    # A non-finite critic prediction (NaN or inf) must map to the failed-extraction
    # penalty, never propagate NaN/inf into the reward (which would poison GRPO).
    gold = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    pred = torch.tensor([
        [float("nan"), 0.0, 0.0, 0.0],   # NaN row
        [float("inf"), 0.0, 0.0, 0.0],   # inf row
        [1.0, 2.0, 3.0, 4.0],            # clean row (perfect → reward 0)
    ])
    out = reward._mse_to_reward(pred, gold, 59.87)
    assert out[0] == reward.FAILED_EXTRACTION_REWARD
    assert out[1] == reward.FAILED_EXTRACTION_REWARD
    assert math.isfinite(out[2])
