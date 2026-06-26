"""Tests for the non-finite-gradient optimizer-step guard (nla.grad_guard).

Pins the gap that killed the canonical critic-2 RL run: a FINITE loss with a
NaN gradient (bf16 backbone-backward overflow on a short truncated prefix) slips
past the in-loss finiteness backstop, and miles then steps the optimizer
unconditionally → NaN weights → dead reward model. The guard must skip the step
(and zero grads) on any non-finite gradient, while letting clean steps through.
"""

import types

import torch

from nla.grad_guard import grads_all_finite, install_grad_finiteness_guard


def _param(grad):
    """A minimal parameter-like object carrying a .grad."""
    return types.SimpleNamespace(grad=grad)


class _FakeDTensor:
    """Stand-in for an FSDP2 sharded grad: finiteness is checked on to_local()."""
    def __init__(self, local):
        self._local = local

    def to_local(self):
        return self._local


class _FakeOptimizer:
    def __init__(self, params):
        self._params = params
        self.step_calls = 0
        self.zero_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1

    def zero_grad(self, set_to_none=True):
        self.zero_calls += 1
        for p in self._params:
            p.grad = None


# --------------------------------------------------------------------------- #
# grads_all_finite
# --------------------------------------------------------------------------- #

def test_finite_grads_are_finite():
    params = [_param(torch.randn(4)), _param(torch.randn(2, 3))]
    assert grads_all_finite(params) is True


def test_none_grads_ignored():
    params = [_param(None), _param(torch.randn(3)), _param(None)]
    assert grads_all_finite(params) is True


def test_nan_grad_detected():
    g = torch.randn(4); g[2] = float("nan")
    assert grads_all_finite([_param(torch.randn(3)), _param(g)]) is False


def test_inf_grad_detected():
    g = torch.randn(4); g[0] = float("inf")
    assert grads_all_finite([_param(g)]) is False


def test_dtensor_local_shard_checked():
    # finite local shard → fine; non-finite local shard → caught
    assert grads_all_finite([_param(_FakeDTensor(torch.randn(5)))]) is True
    bad = torch.randn(5); bad[1] = float("nan")
    assert grads_all_finite([_param(_FakeDTensor(bad))]) is False


# --------------------------------------------------------------------------- #
# install_grad_finiteness_guard
# --------------------------------------------------------------------------- #

def test_finite_step_is_taken():
    params = [_param(torch.randn(4))]
    opt = _FakeOptimizer(params)
    state = install_grad_finiteness_guard(opt, lambda: params)
    opt.step()
    assert opt.step_calls == 1
    assert opt.zero_calls == 0
    assert state["skipped"] == 0


def test_nonfinite_step_is_skipped_and_zeroed():
    g = torch.randn(4); g[0] = float("nan")
    params = [_param(g)]
    opt = _FakeOptimizer(params)
    skips = []
    state = install_grad_finiteness_guard(opt, lambda: params, on_skip=skips.append)
    opt.step()
    assert opt.step_calls == 0          # real step NOT taken
    assert opt.zero_calls == 1          # grads zeroed
    assert state["skipped"] == 1
    assert skips == [1]                 # callback fired with running count
    assert params[0].grad is None       # poison cleared


def test_intermittent_skips_let_clean_steps_through():
    # finite -> nan -> finite : two real steps, one skip (the realistic case
    # where only some microbatch-windows overflow, so the critic keeps learning)
    holder = _param(torch.randn(3))
    params = [holder]
    opt = _FakeOptimizer(params)
    state = install_grad_finiteness_guard(opt, lambda: params)

    holder.grad = torch.randn(3); opt.step()              # finite
    holder.grad = torch.tensor([1.0, float("inf"), 0.0]); opt.step()  # bad
    holder.grad = torch.randn(3); opt.step()              # finite

    assert opt.step_calls == 2
    assert state["skipped"] == 1


def test_install_is_idempotent():
    params = [_param(torch.randn(2))]
    opt = _FakeOptimizer(params)
    s1 = install_grad_finiteness_guard(opt, lambda: params)
    guarded = opt.step
    s2 = install_grad_finiteness_guard(opt, lambda: params)
    assert s1 is s2                     # same state object
    assert opt.step is guarded          # not double-wrapped
