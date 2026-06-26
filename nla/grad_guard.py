"""Skip-the-optimizer-step-on-non-finite-gradient guard.

Why this exists (the gap the in-loss backstop can't cover):

miles' shared FSDP train loop (backends/fsdp_utils/actor.py) does, every step:

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    grad_norm = grad_norm.full_tensor().item()
    self.optimizer.step()          # UNCONDITIONAL — no finiteness check

`clip_grad_norm_` gives no protection: a single NaN/Inf gradient makes the total
norm NaN, the clip coefficient NaN, and `grads.mul_(coef)` then homogenizes
EVERY shard's grads to NaN — after which the unconditional `optimizer.step()`
writes NaN into the weights. For the online critic (which IS the reward model)
that is terminal: once the weights are NaN every reward collapses to the failure
penalty. Observed exactly this on the canonical critic-sharded-across-2-GPUs run:
step 0 had a FINITE loss (0.48) and normal-norm predictions but a NaN grad_norm
(a bf16 backbone-backward overflow on Qwen massive activations when scoring a
short truncated prefix), and the unconditional step poisoned the critic.

`nla.loss.nla_critic_loss`'s finiteness backstop only catches a non-finite
*loss*; it cannot catch a NaN created *during* `.backward()` with a finite loss.
This guard sits at the optimizer-step boundary instead and skips the step
(zeroing grads) when any gradient is non-finite, so a transient bad
microbatch costs one skipped update rather than a dead model — and the critic
keeps training on the clean steps (no freeze required).

Global consistency across ranks is automatic: because `clip_grad_norm_` has
already propagated the NaN coefficient to every shard, a non-finite step looks
non-finite on ALL ranks, so every rank skips together (no extra collective, no
risk of one rank stepping while another doesn't).
"""

import torch


def grads_all_finite(params) -> bool:
    """True iff every gradient is finite. None grads are ignored.

    For FSDP2 sharded params the grad is a DTensor; we check the LOCAL shard
    (`to_local()`), which is correct here — a NaN anywhere makes the clipped
    grads NaN on every shard, so the per-rank local check agrees globally.
    """
    for p in params:
        g = getattr(p, "grad", None)
        if g is None:
            continue
        g = g.to_local() if hasattr(g, "to_local") else g
        if not torch.isfinite(g).all():
            return False
    return True


def install_grad_finiteness_guard(optimizer, params_fn, on_skip=None):
    """Wrap ``optimizer.step`` so it no-ops (and zeroes grads) on non-finite grads.

    Args:
        optimizer: the optimizer whose ``.step`` to guard.
        params_fn: zero-arg callable returning the parameters to check
            (e.g. ``model.parameters``) — called fresh each step.
        on_skip: optional callback ``on_skip(total_skipped:int)`` invoked when a
            step is skipped.

    Returns a small mutable state dict ``{"skipped": int}`` the caller can read.
    Idempotent: re-installing on an already-guarded optimizer returns the
    existing state without double-wrapping.
    """
    if getattr(optimizer, "_nla_grad_guarded", False):
        return optimizer._nla_grad_guard_state

    state = {"skipped": 0}
    real_step = optimizer.step

    def guarded_step(*args, **kwargs):
        if grads_all_finite(params_fn()):
            return real_step(*args, **kwargs)
        # Non-finite grad: drop this update entirely. zero_grad so the poison
        # doesn't linger into the next accumulation window.
        optimizer.zero_grad(set_to_none=True)
        state["skipped"] += 1
        if on_skip is not None:
            on_skip(state["skipped"])
        return None

    optimizer.step = guarded_step
    optimizer._nla_grad_guarded = True
    optimizer._nla_grad_guard_state = state
    return state
