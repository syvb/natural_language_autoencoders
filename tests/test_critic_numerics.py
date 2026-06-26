"""Numerical-safety regression tests for the online critic.

These pin the two fixes for the batch-size-sensitive RL NaN: the critic was
co-trained on truncated prefixes, occasionally producing tiny-norm predictions
whose gradient through `normalize_activation` blew up (≈1e12 at the zero vector)
and, in bf16 over a full batch, poisoned the critic weights to NaN — after which
every reward collapsed to the failure penalty.

  1. normalize_activation: eps INSIDE the sqrt → finite, bounded backward at
     small/zero norm (NOT .clamp_min() on the .norm() output, which leaves the
     0/0 gradient intact).
  2. nla_critic_loss: a non-finite microbatch is dropped (zero-grad, finite)
     instead of stepping the optimizer with NaN.
"""

import types

import torch
import torch.nn.functional as F

from nla.loss import nla_critic_loss
from nla.schema import MM_ACTIVATION_KEY, normalize_activation


# --------------------------------------------------------------------------- #
# normalize_activation gradient safety
# --------------------------------------------------------------------------- #

def _critic_style_grad(v, scale):
    """Gradient magnitude through the direction-only critic loss form."""
    pred = v.clone().requires_grad_(True)
    gold = torch.randn_like(v)
    F.mse_loss(
        normalize_activation(pred, scale),
        normalize_activation(gold, scale),
        reduction="none",
    ).mean().backward()
    return pred.grad


def test_zero_vector_gradient_is_finite():
    scale = 8.0 ** 0.5
    g = _critic_style_grad(torch.zeros(4, 8), scale)
    assert torch.isfinite(g).all()


def test_tiny_norm_gradient_is_bounded():
    # eps=1e-12 inside the sqrt caps the denominator at 1e-6, so the gradient
    # cannot exceed ~scale/1e-6. The old clamp_min(1e-12) allowed ~scale/1e-12
    # (≈1e12), which overflowed bf16 accumulation over a batch.
    scale = 8.0 ** 0.5
    for mag in (1e-6, 1e-9, 0.0):
        v = torch.randn(4, 8) * mag if mag else torch.zeros(4, 8)
        g = _critic_style_grad(v, scale)
        assert torch.isfinite(g).all()
        assert g.abs().max() < 1e9, (mag, g.abs().max().item())


def test_normal_norm_gradient_unchanged():
    # eps must be negligible for real activations: forward value within 1e-4.
    scale = 60.0
    v = torch.randn(4, 32) * 5.0
    out = normalize_activation(v, scale)
    assert torch.allclose(out.norm(dim=-1), torch.full((4,), scale), atol=1e-3)


def test_normalize_passthrough_when_scale_none():
    v = torch.randn(3, 5)
    assert torch.equal(normalize_activation(v, None), v)


# --------------------------------------------------------------------------- #
# nla_critic_loss finiteness backstop
# --------------------------------------------------------------------------- #

def _batch(d=8):
    toks = [torch.tensor([1, 2, 3]), torch.tensor([4, 5]), torch.tensor([6, 7, 8, 9])]
    T = sum(t.shape[0] for t in toks)  # last-token indices: 2, 4, 8
    args = types.SimpleNamespace(nla_mse_scale=float(d ** 0.5), nla_baseline_rawvar=0.9)
    batch = {"unconcat_tokens": toks, MM_ACTIVATION_KEY: torch.randn(len(toks), d)}
    return args, batch, T, len(toks)


def test_finite_batch_passes_through():
    d = 8
    args, batch, T, B = _batch(d)
    values = torch.randn(1, T, d, requires_grad=True)
    loss, log = nla_critic_loss(args, None, batch, values, None)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(values.grad).all()
    assert float(log["loss_nonfinite"]) == 0.0
    # diagnostics present (×B convention, like fve_nrm)
    for k in ("pred_norm_min", "pred_norm_max", "values_absmax", "loss_nonfinite"):
        assert k in log


def test_nonfinite_microbatch_is_dropped_not_propagated():
    d = 8
    args, batch, T, B = _batch(d)
    values = torch.randn(1, T, d, requires_grad=True)
    # inject inf at a LAST-token position (index 2) so it enters `pred`
    mask = torch.zeros(1, T, d)
    mask[0, 2, 0] = 1.0
    poison = values + mask * float("inf")
    loss, log = nla_critic_loss(args, None, batch, poison, None)
    loss.backward()
    assert torch.isfinite(loss)                       # guard produced finite loss
    assert float(log["loss_nonfinite"]) == B          # flagged (×B)
    assert torch.isfinite(values.grad).all()
    assert values.grad.abs().max() == 0.0             # step skipped, weights safe


def test_backstop_finite_under_many_nonfinite_and_bf16():
    # The realistic divergence: NOT one stray inf but the whole tensor corrupt.
    # nan_to_num(inf)->dtype_max summed over many tokens overflows, and 0*inf=NaN
    # — so the surrogate must zero BEFORE summing. Exercise both dtypes; bf16 is
    # the production dtype and overflows fastest.
    d = 8
    for dtype in (torch.float32, torch.bfloat16):
        args, batch, T, B = _batch(d)
        v = torch.randn(1, T, d, dtype=dtype)
        v[0, :, 0] = float("inf")     # every token non-finite
        v[0, 4, 1] = float("nan")
        values = v.clone().requires_grad_(True)
        loss, log = nla_critic_loss(args, None, batch, values, None)
        loss.backward()
        assert torch.isfinite(loss), dtype             # surrogate stays finite
        assert float(log["loss_nonfinite"]) == B, dtype
        assert torch.isfinite(values.grad).all(), dtype
        assert float(values.grad.abs().max()) == 0.0, dtype
