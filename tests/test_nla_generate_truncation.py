"""Integration test for nla.rollout.nla_generate.generate() truncation wiring.

Heavy I/O internals (_prep_payload_sync, post, update_sample_from_response,
_resolve_url, embed reload) are monkeypatched so we run the real generate()
control flow offline and assert:
  - max_new_tokens is overridden to the per-group content budget + offset,
  - all samples of a group share that budget (group_index keying),
  - the token-limit penalty is removed when truncation is ON (TRUNCATED kept),
    but preserved when OFF (TRUNCATED -> FAILED).
"""

import asyncio
import types

import numpy as np
import torch

import nla.rollout.nla_generate as gen
from nla.truncation import TruncationConfig

Status = gen.Sample.Status


class _Tok:
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 2, 3]}


class _Cfg:
    critic_prompt_template = "Summary: <text>{explanation}</text> <summary>"


def _args():
    return types.SimpleNamespace(rollout_max_context_len=300)


def _install(monkeypatch, *, enabled, opening_offset=3, max_tokens=130, seed=0,
             response="<explanation>\nimportant stuff first and then more", set_status=Status.TRUNCATED):
    """Wire generate()'s globals + monkeypatch its I/O. Returns a dict that
    captures the sampling_params generate() actually sent downstream."""
    captured = {}

    monkeypatch.setattr(gen, "_lazy_init", lambda args: None)
    monkeypatch.setattr(gen, "_maybe_reload_embed", lambda args: None)
    monkeypatch.setattr(gen, "_TOKENIZER", _Tok())
    monkeypatch.setattr(gen, "_CFG", _Cfg())
    monkeypatch.setattr(gen, "_OPENING_OFFSET", opening_offset)
    monkeypatch.setattr(
        gen, "_TRUNC",
        TruncationConfig(enabled=enabled, min_tokens=1, max_tokens=max_tokens, seed=seed),
    )

    async def _resolve_url(args, idx):
        return "http://stub/generate"
    monkeypatch.setattr(gen, "_resolve_url", _resolve_url)

    def _prep(args, messages, activation_vector, sampling_params, sample_index):
        # record exactly what generate() decided to send
        captured["sampling_params"] = dict(sampling_params)
        input_ids = [1, 2, 3]
        v_raw = torch.zeros(1, 4)
        embeds_out = np.zeros((3, 4), dtype=np.float32)
        payload = {"input_ids": input_ids, "sampling_params": dict(sampling_params)}
        return input_ids, v_raw, embeds_out, payload, None
    monkeypatch.setattr(gen, "_prep_payload_sync", _prep)

    async def _post(url, payload):
        return {"meta_info": {"output_token_logprobs": [(-0.1, 1, None)] * 5,
                              "finish_reason": "length"}}
    monkeypatch.setattr(gen, "post", _post)

    async def _update(args, sample, payload, output):
        sample.response = response
        sample.status = set_status
    monkeypatch.setattr(gen, "update_sample_from_response", _update)

    return captured


class _Sample:
    def __init__(self, group_index, index=0):
        self.prompt = [{"role": "user", "content": "x ㊗ y"}]
        self.metadata = {"activation_vector": [0.0, 0.0, 0.0, 0.0]}
        self.group_index = group_index
        self.index = index
        self.status = Status.PENDING
        self.multimodal_train_inputs = None


def _run(sample, sampling_params):
    return asyncio.run(gen.generate(_args(), sample, sampling_params))


def test_max_new_tokens_overridden_to_group_budget(monkeypatch):
    cap = _install(monkeypatch, enabled=True, opening_offset=3, max_tokens=130, seed=42)
    gi = 17
    s = _Sample(group_index=gi)
    _run(s, {"max_new_tokens": 150, "temperature": 1.0})
    from nla.truncation import sample_truncation_length
    expected_content = sample_truncation_length(42, gi, 1, 130)
    assert cap["sampling_params"]["max_new_tokens"] == expected_content + 3
    # unrelated sampling params are preserved
    assert cap["sampling_params"]["temperature"] == 1.0


def test_budget_shared_across_group_members(monkeypatch):
    cap = _install(monkeypatch, enabled=True, opening_offset=3, max_tokens=130, seed=42)
    seen = set()
    for idx in range(8):  # 8 samples, SAME group_index
        s = _Sample(group_index=99, index=idx)
        _run(s, {"max_new_tokens": 150})
        seen.add(cap["sampling_params"]["max_new_tokens"])
    assert len(seen) == 1, "group members got different budgets"


def test_budget_clamped_to_existing_cap(monkeypatch):
    cap = _install(monkeypatch, enabled=True, opening_offset=3, max_tokens=130, seed=1)
    s = _Sample(group_index=5)
    _run(s, {"max_new_tokens": 10})   # tiny existing cap dominates
    assert cap["sampling_params"]["max_new_tokens"] == 10


def test_disabled_does_not_override(monkeypatch):
    cap = _install(monkeypatch, enabled=False)
    s = _Sample(group_index=5)
    _run(s, {"max_new_tokens": 150})
    assert cap["sampling_params"]["max_new_tokens"] == 150


def test_truncated_kept_when_enabled(monkeypatch):
    _install(monkeypatch, enabled=True, set_status=Status.TRUNCATED,
             response="<explanation>\nfront-loaded content no close tag")
    s = _Sample(group_index=3)
    out = _run(s, {"max_new_tokens": 150})
    assert out.status == Status.TRUNCATED                    # NOT promoted to FAILED
    assert gen.MM_CRITIC_TOKENS_KEY in out.multimodal_train_inputs   # critic stash present


def test_truncated_failed_when_disabled(monkeypatch):
    _install(monkeypatch, enabled=False, set_status=Status.TRUNCATED,
             response="<explanation>\nfront-loaded content no close tag")
    s = _Sample(group_index=3)
    out = _run(s, {"max_new_tokens": 150})
    assert out.status == Status.FAILED                       # legacy penalty path
    assert gen.MM_CRITIC_TOKENS_KEY not in (out.multimodal_train_inputs or {})


def test_empty_response_sets_failed(monkeypatch):
    # v2: only an empty/whitespace response is an extraction miss → FAILED. A
    # no-tag non-empty response is the whole payload (scored), tested elsewhere.
    _install(monkeypatch, enabled=True, set_status=Status.COMPLETED, response="   ")
    s = _Sample(group_index=3)
    out = _run(s, {"max_new_tokens": 150})
    assert out.status == Status.FAILED


def test_untagged_response_scored_v2(monkeypatch):
    # v2 list format: actor emits a raw newline list with no <explanation> wrapper;
    # it must be kept and reach the critic (not dropped as an extraction miss).
    _install(monkeypatch, enabled=True, set_status=Status.TRUNCATED,
             response="most salient item\nsecond item\nthird")
    s = _Sample(group_index=3)
    out = _run(s, {"max_new_tokens": 150})
    assert out.status == Status.TRUNCATED
    assert gen.MM_CRITIC_TOKENS_KEY in out.multimodal_train_inputs


class _FakeItemsTrunc:
    enabled = True
    mode = "items"

    def __init__(self, k):
        self._k = k

    def items_for_group(self, group_index):
        return self._k


class _CharDecodeTok:
    """Tokenizer stub exposing just .decode (one id per char) for item cut."""
    def __init__(self, id2ch):
        self._m = id2ch

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self._m[i] for i in ids if i in self._m)


def _items_sample(prompt_ids, response_text):
    ids = {1000 + i: c for i, c in enumerate(response_text)}
    resp_ids = list(ids)
    s = types.SimpleNamespace(
        tokens=list(prompt_ids) + resp_ids,
        response_length=len(resp_ids),
        rollout_log_probs=[-0.1] * len(resp_ids),
        response=response_text,
        group_index=3,
    )
    return s, ids


def test_post_truncate_to_items_slices_aligned(monkeypatch):
    # items mode, K=2 → keep "aa\nbb"; tokens/logprobs/response/length all sliced.
    monkeypatch.setattr(gen, "_TRUNC", _FakeItemsTrunc(2))
    s, id2ch = _items_sample([101, 102, 103], "aa\nbb\ncc")
    monkeypatch.setattr(gen, "_TOKENIZER", _CharDecodeTok(id2ch))
    gen._post_truncate_to_items(s)
    assert s.response == "aa\nbb"
    assert s.response_length == 5
    assert len(s.rollout_log_probs) == 5
    assert len(s.tokens) == 3 + 5            # prompt (3) + kept response (5)


def test_post_truncate_to_items_noop_when_fewer_items(monkeypatch):
    monkeypatch.setattr(gen, "_TRUNC", _FakeItemsTrunc(9))  # K > items present
    s, id2ch = _items_sample([101, 102, 103], "aa\nbb\ncc")
    monkeypatch.setattr(gen, "_TOKENIZER", _CharDecodeTok(id2ch))
    gen._post_truncate_to_items(s)
    assert s.response == "aa\nbb\ncc"
    assert s.response_length == 8
    assert len(s.tokens) == 3 + 8


def test_completed_within_budget_keeps_close_tag_content(monkeypatch):
    # natural finish before the cap: response HAS a close tag; critic content is
    # the inter-tag text (close tag never reaches the critic).
    _install(monkeypatch, enabled=True, set_status=Status.COMPLETED,
             response="<explanation>\nshort and done\n</explanation>")
    s = _Sample(group_index=3)
    out = _run(s, {"max_new_tokens": 150})
    assert out.status == Status.COMPLETED
    critic_tokens = out.multimodal_train_inputs[gen.MM_CRITIC_TOKENS_KEY]
    assert critic_tokens is not None
