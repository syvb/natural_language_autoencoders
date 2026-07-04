"""Position-tapered KL penalty for the matryoshka policy loss.

The truncation reward pushes the most informative content to the front of the
explanation, so later response tokens contribute little marginal FVE. Under a
flat per-token KL coefficient those late tokens are dominated by the KL term
— the reward barely reaches them but the regularizer does. This module keeps
miles' policy loss identical except for the KL term, which becomes

    loss += kl_loss_coef * reduce( w(t) * kl_t )
    w(t)  = max(0.5 ** (t / NLA_KL_TAPER_HALF_LIFE), NLA_KL_TAPER_FLOOR)

with t the 0-based position of the token within its response. --kl-loss-coef
is therefore the coefficient at the FIRST response token (e.g. 0.02), decaying
with a half-life measured in tokens.

Enabled by NLAFSDPActor.init (actor role only) when NLA_KL_TAPER_HALF_LIFE is
set to a positive number: it swaps --loss-type to custom_loss pointing here,
the same mechanism the critic role uses for nla_critic_loss. miles itself is
untouched.

Env knobs:
    NLA_KL_TAPER_HALF_LIFE  tokens for the weight to halve (0/unset = off)
    NLA_KL_TAPER_FLOOR      lower bound on w(t), relative to the first-token
                            coefficient (default 0.0)

Reported metrics (all land in wandb as train/<name>):
    kl_loss      WEIGHTED mean KL — the term actually in the loss (times
                 kl_loss_coef); not comparable to flat-KL runs
    kl_flat      unweighted mean KL — what a flat-KL run would have reported
    kl_coef_eff  the KL coefficient effectively applied per token this step:
                 kl_loss_coef x batch-mean taper weight. Varies per batch with
                 the truncation-drawn response lengths; equals kl_loss_coef in
                 the no-taper limit. THIS is "the current KL penalty" to watch.
    kl_penalty   kl_loss_coef x kl_loss — the actual scalar added to the loss

Cost note: the wrapped call runs with use_kl_loss off, and the KL term is
rebuilt here from a second get_log_probs_and_entropy pass — one extra
response-token softmax over the vocab per micro-batch, negligible next to
the backbone forward/backward.
"""
import os

import torch


def taper_half_life() -> float:
    return float(os.environ.get("NLA_KL_TAPER_HALF_LIFE", "0") or 0)


def taper_floor() -> float:
    return float(os.environ.get("NLA_KL_TAPER_FLOOR", "0") or 0)


def taper_weights(
    n: int, half_life: float, floor: float, device=None
) -> torch.Tensor:
    """w(t) = max(0.5 ** (t / half_life), floor) for t in [0, n)."""
    assert half_life > 0, f"half_life must be positive, got {half_life}"
    t = torch.arange(n, device=device, dtype=torch.float32)
    return torch.clamp(torch.pow(0.5, t / half_life), min=floor)


def nla_policy_loss_tapered_kl(args, parallel_state, batch, logits, sum_of_sample_mean):
    """Drop-in for miles' policy_loss_function with a position-tapered KL term."""
    from miles.backends.training_utils.loss import (
        get_log_probs_and_entropy,
        policy_loss_function,
    )
    from miles.utils.ppo_utils import compute_approx_kl

    half_life = taper_half_life()
    assert half_life > 0, "nla_policy_loss_tapered_kl dispatched without NLA_KL_TAPER_HALF_LIFE"
    assert args.use_kl_loss, "KL taper requires --use-kl-loss (it replaces that term)"
    assert parallel_state.cp_size == 1, (
        "KL taper indexes token positions within each response; cp>1 shards "
        "the token dim and breaks that (NLA requires cp_size=1 anyway)."
    )

    # Everything except the KL term must match upstream exactly: run the stock
    # loss with the flat-KL term disabled, then add the tapered term.
    args.use_kl_loss = False
    try:
        loss, reported = policy_loss_function(
            args, parallel_state, batch, logits, sum_of_sample_mean
        )
    finally:
        args.use_kl_loss = True

    lpe = get_log_probs_and_entropy(
        logits,
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    per_sample = lpe["log_probs"]  # list of [R_i] tensors, response tokens only
    log_probs = torch.cat(per_sample, dim=0)
    ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)

    importance_ratio = None
    if getattr(args, "use_unbiased_kl", False):
        old = (
            batch["rollout_log_probs"]
            if getattr(args, "use_rollout_logprobs", False)
            else batch["log_probs"]
        )
        importance_ratio = torch.exp(log_probs - torch.cat(old, dim=0))

    kl = compute_approx_kl(
        log_probs,
        ref_log_probs,
        kl_loss_type=args.kl_loss_type,
        importance_ratio=importance_ratio,
    )
    floor = taper_floor()
    weights = torch.cat(
        [taper_weights(p.numel(), half_life, floor, device=kl.device) for p in per_sample]
    )
    kl_loss = sum_of_sample_mean(kl * weights)
    loss = loss + args.kl_loss_coef * kl_loss

    reported["kl_loss"] = kl_loss.clone().detach()
    reported["kl_flat"] = sum_of_sample_mean(kl).clone().detach()
    reported["kl_coef_eff"] = (
        args.kl_loss_coef * sum_of_sample_mean(weights).clone().detach()
    )
    reported["kl_penalty"] = args.kl_loss_coef * reported["kl_loss"]
    return loss, reported
