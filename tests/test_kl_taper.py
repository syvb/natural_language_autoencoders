"""Unit tests for the position-tapered KL policy loss (nla/kl_taper.py).

miles' loss module is stubbed per-test (same pattern as conftest): the wrapper
delegates pg/entropy to policy_loss_function and only owns the KL term, so the
tests pin down (a) the taper weight math, (b) the cat-across-samples position
layout, (c) the use_kl_loss toggle hygiene.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from nla.kl_taper import nla_policy_loss_tapered_kl, taper_weights


def test_taper_weights_values():
    w = taper_weights(31, half_life=10.0, floor=0.0)
    assert w[0] == pytest.approx(1.0)
    assert w[10] == pytest.approx(0.5)
    assert w[20] == pytest.approx(0.25)
    assert w[30] == pytest.approx(0.125)
    assert (w[1:] < w[:-1]).all(), "weights must be strictly decreasing without a floor"


def test_taper_weights_floor():
    w = taper_weights(200, half_life=10.0, floor=0.1)
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(0.1)
    assert (w >= 0.1).all()


def _install_loss_stubs(monkeypatch, policy_loss_fn, log_probs_per_sample):
    """Register just-enough miles modules for kl_taper's lazy imports."""
    loss_mod = types.ModuleType("miles.backends.training_utils.loss")
    loss_mod.policy_loss_function = policy_loss_fn

    def fake_get_log_probs_and_entropy(logits, **kwargs):
        return {"log_probs": log_probs_per_sample}

    loss_mod.get_log_probs_and_entropy = fake_get_log_probs_and_entropy

    ppo_mod = types.ModuleType("miles.utils.ppo_utils")

    def fake_compute_approx_kl(log_probs, log_probs_base, kl_loss_type, importance_ratio=None):
        assert kl_loss_type == "k1"
        assert importance_ratio is None
        return log_probs - log_probs_base  # k1 estimator

    ppo_mod.compute_approx_kl = fake_compute_approx_kl

    for name, mod in [
        ("miles.backends", types.ModuleType("miles.backends")),
        ("miles.backends.training_utils", types.ModuleType("miles.backends.training_utils")),
        ("miles.backends.training_utils.loss", loss_mod),
        ("miles.utils.ppo_utils", ppo_mod),
    ]:
        if name not in ("miles.backends.training_utils.loss", "miles.utils.ppo_utils"):
            mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)


def _make_batch(log_probs_per_sample, ref_log_probs_per_sample):
    return {
        "unconcat_tokens": [torch.zeros(p.numel(), dtype=torch.long) for p in log_probs_per_sample],
        "total_lengths": [p.numel() + 3 for p in log_probs_per_sample],
        "response_lengths": [p.numel() for p in log_probs_per_sample],
        "loss_masks": [torch.ones(p.numel()) for p in log_probs_per_sample],
        "ref_log_probs": ref_log_probs_per_sample,
        "log_probs": [p.detach() for p in log_probs_per_sample],
    }


def _args(**over):
    base = dict(
        use_kl_loss=True,
        kl_loss_coef=0.02,
        kl_loss_type="k1",
        use_unbiased_kl=False,
        use_rollout_logprobs=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_tapered_kl_math_and_layout(monkeypatch):
    monkeypatch.setenv("NLA_KL_TAPER_HALF_LIFE", "10")
    monkeypatch.setenv("NLA_KL_TAPER_FLOOR", "0")

    lp = [torch.zeros(4), torch.zeros(25)]
    ref = [torch.full((4,), -1.0), torch.full((25,), -1.0)]  # kl_t = +1 everywhere

    seen = {}

    def fake_policy_loss(args, parallel_state, batch, logits, sum_of_sample_mean):
        seen["use_kl_loss_inside"] = args.use_kl_loss
        return torch.tensor(7.0), {"pg_loss": torch.tensor(7.0)}

    _install_loss_stubs(monkeypatch, fake_policy_loss, lp)

    args = _args()
    loss, reported = nla_policy_loss_tapered_kl(
        args,
        SimpleNamespace(cp_size=1),
        _make_batch(lp, ref),
        logits=torch.zeros(1, 32, 8),
        sum_of_sample_mean=lambda x: x.sum(),
    )

    # The wrapped call must see the flat-KL term disabled, then restored.
    assert seen["use_kl_loss_inside"] is False
    assert args.use_kl_loss is True

    # kl_t == 1, so the weighted sum is exactly the sum of taper weights,
    # laid out per-sample: positions restart at 0 for the second sample.
    expected_w = torch.cat([taper_weights(4, 10.0, 0.0), taper_weights(25, 10.0, 0.0)])
    assert reported["kl_loss"] == pytest.approx(expected_w.sum().item())
    assert reported["kl_flat"] == pytest.approx(29.0)
    assert loss.item() == pytest.approx(7.0 + 0.02 * expected_w.sum().item())


def test_position_restart_beats_flat_continuation(monkeypatch):
    """Sample boundaries must reset t to 0 — cat'ing one arange would not."""
    monkeypatch.setenv("NLA_KL_TAPER_HALF_LIFE", "5")
    monkeypatch.setenv("NLA_KL_TAPER_FLOOR", "0")

    lp = [torch.zeros(10), torch.zeros(10)]
    ref = [torch.full((10,), -1.0)] * 2

    _install_loss_stubs(
        monkeypatch, lambda *a, **k: (torch.tensor(0.0), {}), lp
    )
    _, reported = nla_policy_loss_tapered_kl(
        _args(),
        SimpleNamespace(cp_size=1),
        _make_batch(lp, ref),
        logits=torch.zeros(1, 32, 8),
        sum_of_sample_mean=lambda x: x.sum(),
    )
    per_sample = taper_weights(10, 5.0, 0.0).sum().item()
    assert reported["kl_loss"] == pytest.approx(2 * per_sample)


def test_toggle_restored_on_exception(monkeypatch):
    monkeypatch.setenv("NLA_KL_TAPER_HALF_LIFE", "10")

    def boom(*a, **k):
        raise RuntimeError("inner loss failed")

    _install_loss_stubs(monkeypatch, boom, [torch.zeros(3)])
    args = _args()
    with pytest.raises(RuntimeError, match="inner loss failed"):
        nla_policy_loss_tapered_kl(
            args,
            SimpleNamespace(cp_size=1),
            _make_batch([torch.zeros(3)], [torch.zeros(3)]),
            logits=torch.zeros(1, 8, 8),
            sum_of_sample_mean=lambda x: x.sum(),
        )
    assert args.use_kl_loss is True


def test_requires_use_kl_loss(monkeypatch):
    monkeypatch.setenv("NLA_KL_TAPER_HALF_LIFE", "10")
    _install_loss_stubs(monkeypatch, lambda *a, **k: (torch.tensor(0.0), {}), [torch.zeros(3)])
    with pytest.raises(AssertionError, match="use-kl-loss"):
        nla_policy_loss_tapered_kl(
            _args(use_kl_loss=False),
            SimpleNamespace(cp_size=1),
            _make_batch([torch.zeros(3)], [torch.zeros(3)]),
            logits=torch.zeros(1, 8, 8),
            sum_of_sample_mean=lambda x: x.sum(),
        )


def test_requires_cp1(monkeypatch):
    monkeypatch.setenv("NLA_KL_TAPER_HALF_LIFE", "10")
    _install_loss_stubs(monkeypatch, lambda *a, **k: (torch.tensor(0.0), {}), [torch.zeros(3)])
    with pytest.raises(AssertionError, match="cp"):
        nla_policy_loss_tapered_kl(
            _args(),
            SimpleNamespace(cp_size=2),
            _make_batch([torch.zeros(3)], [torch.zeros(3)]),
            logits=torch.zeros(1, 8, 8),
            sum_of_sample_mean=lambda x: x.sum(),
        )
