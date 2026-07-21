"""Suffix-mode ("left-matryoshka") truncation: unit tests for the last-k-token
cut plus wiring tests for both consumers.

The invariants under test:
  - truncate_to_token_suffix keeps exactly the last k tokens (identity when the
    text is already short enough),
  - suffix mode draws from the SAME per-group RNG stream as tokens mode (a
    suffix run sees the identical budget sequence as a tokens run, same seed),
  - nla_generate does NOT cap max_new_tokens and does NOT slice the trajectory;
    only the critic co-training tokens see the suffix,
  - nla.reward._prep_batch scores the same suffix (same helper, same k),
  - both consumers fail loudly on a sample without group_index.

`ray` / `miles.*` are stubbed in conftest; tokenizers are faked (1 token per
whitespace word) — we test the slicing logic, not a real vocab.
"""

import asyncio
import types

import numpy as np
import pytest
import torch

import nla.reward as reward
import nla.rollout.nla_generate as gen
from nla.truncation import (
    TruncationConfig,
    resolve_truncation_config,
    sample_truncation_length,
    truncate_to_token_suffix,
)

Status = reward.Sample.Status


class _WordTok:
    """1 token per whitespace word. Records every critic-prompt tokenization."""

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(ids)

    def __call__(self, prompts, add_special_tokens=True, padding=True, return_tensors=None):
        self.calls.append(prompts if isinstance(prompts, list) else [prompts])
        n = len(self.calls[-1])
        if return_tensors == "pt":
            return {
                "input_ids": torch.zeros(n, 4, dtype=torch.long),
                "attention_mask": torch.ones(n, 4, dtype=torch.long),
            }
        return {"input_ids": [1, 2, 3]}


# --------------------------------------------------------------------------- #
# truncate_to_token_suffix
# --------------------------------------------------------------------------- #

def test_suffix_keeps_last_k_tokens():
    tok = _WordTok()
    text = " ".join(f"w{i}" for i in range(50))
    assert truncate_to_token_suffix(text, 10, tok) == " ".join(f"w{i}" for i in range(40, 50))


def test_suffix_identity_when_short():
    tok = _WordTok()
    assert truncate_to_token_suffix("a b c", 3, tok) == "a b c"
    assert truncate_to_token_suffix("a b c", 10, tok) == "a b c"


def test_suffix_k_one_is_last_token():
    tok = _WordTok()
    assert truncate_to_token_suffix("first middle last", 1, tok) == "last"


def test_suffix_rejects_nonpositive_k():
    tok = _WordTok()
    with pytest.raises(AssertionError):
        truncate_to_token_suffix("a b c", 0, tok)


# --------------------------------------------------------------------------- #
# config resolution + RNG stream
# --------------------------------------------------------------------------- #

def _args():
    return types.SimpleNamespace(rollout_seed=0)


def test_suffix_mode_resolves_and_enables(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MODE", "suffix")
    monkeypatch.setenv("NLA_TRUNC_MIN_TOKENS", "1")
    monkeypatch.setenv("NLA_TRUNC_MAX_TOKENS", "120")
    cfg = resolve_truncation_config(_args())
    assert cfg.mode == "suffix" and cfg.enabled
    assert (cfg.min_tokens, cfg.max_tokens) == (1, 120)


def test_suffix_mode_disabled_without_budget(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MODE", "suffix")
    monkeypatch.delenv("NLA_TRUNC_MAX_TOKENS", raising=False)
    assert not resolve_truncation_config(_args()).enabled


def test_suffix_mode_bad_range_asserts(monkeypatch):
    monkeypatch.setenv("NLA_TRUNC_MODE", "suffix")
    monkeypatch.setenv("NLA_TRUNC_MIN_TOKENS", "50")
    monkeypatch.setenv("NLA_TRUNC_MAX_TOKENS", "10")
    with pytest.raises(AssertionError):
        resolve_truncation_config(_args())


def test_suffix_budget_stream_matches_tokens_mode():
    # Deliberate: same (seed, group) → same k as a tokens-mode run, so the
    # left- and right-truncation experiments compare at identical budgets.
    suf = TruncationConfig(enabled=True, min_tokens=1, max_tokens=120, seed=42, mode="suffix")
    for gi in (0, 1, 17, 999):
        assert suf.length_for_group(gi) == sample_truncation_length(42, gi, 1, 120)


# --------------------------------------------------------------------------- #
# reward wiring
# --------------------------------------------------------------------------- #

class _FakeCfg:
    critic_prompt_template = "Summary: <text>{explanation}</text> <summary>"
    mse_scale = 59.87


class _Sample:
    def __init__(self, status, response, group_index=None):
        self.status = status
        self.response = response
        self.metadata = {"activation_vector": [0.1, 0.2, 0.3, 0.4]}
        if group_index is not None:
            self.group_index = group_index


def _setup_reward(monkeypatch, mode="suffix", enabled=True, seed=7):
    tok = _WordTok()
    monkeypatch.setattr(reward, "_TOKENIZER", tok)
    monkeypatch.setattr(reward, "_CFG", _FakeCfg())
    cfg = TruncationConfig(enabled=enabled, min_tokens=1, max_tokens=120, seed=seed, mode=mode)
    monkeypatch.setattr(reward, "_TRUNC", cfg)
    return tok, cfg


def test_reward_scores_last_k_tokens(monkeypatch):
    tok, cfg = _setup_reward(monkeypatch)
    gi = 11
    k = cfg.length_for_group(gi)
    words = [f"w{i}" for i in range(200)]  # 200 > max_tokens → cut always fires
    s = _Sample(Status.TRUNCATED, " ".join(words), group_index=gi)
    payload, orig_idx, _ = reward._prep_batch([s])
    assert payload is not None and orig_idx == [0]
    expected = _FakeCfg.critic_prompt_template.format(explanation=" ".join(words[-k:]))
    assert tok.calls[-1] == [expected]


def test_reward_suffix_shared_within_group(monkeypatch):
    tok, cfg = _setup_reward(monkeypatch)
    words = " ".join(f"w{i}" for i in range(200))
    samples = [_Sample(Status.TRUNCATED, words, group_index=5) for _ in range(4)]
    reward._prep_batch(samples)
    assert len(set(tok.calls[-1])) == 1  # every member cut at the same k


def test_reward_suffix_requires_group_index(monkeypatch):
    _setup_reward(monkeypatch)
    s = _Sample(Status.TRUNCATED, "a b c", group_index=None)
    with pytest.raises(AssertionError, match="group_index"):
        reward._prep_batch([s])


def test_reward_tokens_mode_untouched(monkeypatch):
    # tokens mode: response is already the capped prefix; no reward-side cut.
    tok, _ = _setup_reward(monkeypatch, mode="tokens")
    words = " ".join(f"w{i}" for i in range(200))
    s = _Sample(Status.TRUNCATED, words, group_index=3)
    reward._prep_batch([s])
    assert tok.calls[-1] == [_FakeCfg.critic_prompt_template.format(explanation=words)]


# --------------------------------------------------------------------------- #
# nla_generate wiring
# --------------------------------------------------------------------------- #

class _GenCfg:
    critic_prompt_template = "Summary: <text>{explanation}</text> <summary>"
    actor_prompt_template = "x {injection_char} y"
    injection_char = "㊗"


def _install_gen(monkeypatch, response, *, seed=7):
    captured = {}
    tok = _WordTok()
    monkeypatch.setattr(gen, "_lazy_init", lambda args: None)
    monkeypatch.setattr(gen, "_maybe_reload_embed", lambda args: None)
    monkeypatch.setattr(gen, "_TOKENIZER", tok)
    monkeypatch.setattr(gen, "_CFG", _GenCfg())
    monkeypatch.setattr(
        gen, "_TRUNC",
        TruncationConfig(enabled=True, min_tokens=1, max_tokens=120, seed=seed, mode="suffix"),
    )

    async def _resolve_url(args, idx):
        return "http://stub/generate"
    monkeypatch.setattr(gen, "_resolve_url", _resolve_url)

    def _prep(args, messages, activation_vector, sampling_params, sample_index):
        captured["sampling_params"] = dict(sampling_params)
        return [1, 2, 3], torch.zeros(1, 4), np.zeros((3, 4), dtype=np.float32), \
            {"input_ids": [1, 2, 3], "sampling_params": dict(sampling_params)}, None
    monkeypatch.setattr(gen, "_prep_payload_sync", _prep)

    async def _post(url, payload):
        return {"meta_info": {"output_token_logprobs": [(-0.1, 1, None)] * 5,
                              "finish_reason": "length"}}
    monkeypatch.setattr(gen, "post", _post)

    async def _update(args, sample, payload, output):
        sample.response = response
        sample.status = gen.Sample.Status.TRUNCATED
    monkeypatch.setattr(gen, "update_sample_from_response", _update)
    return captured, tok


def _gen_sample(group_index):
    s = types.SimpleNamespace(
        prompt=[{"role": "user", "content": "x ㊗ y"}],
        metadata={"activation_vector": [0.0] * 4},
        group_index=group_index,
        index=0,
        status=gen.Sample.Status.PENDING,
        multimodal_train_inputs=None,
        response=None,
    )
    return s


def test_generate_suffix_does_not_cap_generation(monkeypatch):
    captured, _ = _install_gen(monkeypatch, "a b c")
    s = _gen_sample(group_index=17)
    asyncio.run(gen.generate(types.SimpleNamespace(rollout_max_context_len=300), s,
                             {"max_new_tokens": 160, "temperature": 1.0}))
    assert captured["sampling_params"]["max_new_tokens"] == 160  # untouched


def test_generate_suffix_critic_sees_suffix_response_stays_full(monkeypatch):
    words = [f"w{i}" for i in range(200)]
    full = " ".join(words)
    captured, tok = _install_gen(monkeypatch, full, seed=7)
    gi = 11
    s = _gen_sample(group_index=gi)
    out = asyncio.run(gen.generate(types.SimpleNamespace(rollout_max_context_len=3000), s,
                                   {"max_new_tokens": 160}))
    assert out.status == gen.Sample.Status.TRUNCATED           # kept, not FAILED
    assert out.response == full                                # trajectory NOT sliced
    k = TruncationConfig(enabled=True, min_tokens=1, max_tokens=120,
                         seed=7, mode="suffix").length_for_group(gi)
    expected = _GenCfg.critic_prompt_template.format(explanation=" ".join(words[-k:]))
    assert tok.calls[-1] == [expected]                         # critic tokens = suffix
    assert gen.MM_CRITIC_TOKENS_KEY in out.multimodal_train_inputs


def test_generate_suffix_requires_group_index(monkeypatch):
    _install_gen(monkeypatch, "a b c d")
    s = _gen_sample(group_index=None)
    with pytest.raises(AssertionError, match="group_index"):
        asyncio.run(gen.generate(types.SimpleNamespace(rollout_max_context_len=300), s,
                                 {"max_new_tokens": 160}))
