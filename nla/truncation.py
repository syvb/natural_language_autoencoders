"""Random-length truncation of AV rollouts — "information-upfront" RL.

During RL we cap each AV (actor) generation at a random number of explanation
*content* tokens, drawn uniformly from [min_tokens, max_tokens]. Because the
reward is then computed on a random-length prefix of the explanation, over
training the model is pushed to make EVERY prefix maximally informative — i.e.
to put the most important information first.

Three decisions are baked in (chosen with the user):

  1. SHARED PER GROUP. All ``n_samples_per_prompt`` samples of a prompt get the
     SAME random length (it is keyed on ``group_index``, which is identical
     across a group — see ``NLADataSource.get_samples``). GRPO normalises
     advantage WITHIN each group, so if length varied per-sample the reward gap
     would be dominated by length (a longer prefix almost always reconstructs
     better) rather than by content quality — and the model cannot control the
     random length, so that signal would be noise. A shared length makes the
     within-group comparison isolate content quality at a fixed length.

  2. CONTENT TOKENS ONLY. The random length counts the explanation CONTENT; the
     fixed ``"<explanation>\\n"`` opening tag is always kept on top (we add its
     token length as an offset to ``max_new_tokens``). This avoids spending
     tiny-length samples entirely on boilerplate.

  3. CAP, DON'T POST-TRUNCATE. We do not generate the full response and then
     slice it. We set ``max_new_tokens = content_len + opening_offset`` so the
     model simply stops early. That makes the actor's trained tokens, loss mask
     and rollout logprobs naturally length-limited with ZERO post-hoc token
     surgery — distributionally identical to generate-then-truncate for the kept
     prefix (each token depends only on its prefix, not on the cap). The closing
     ``</explanation>`` tag is therefore never generated when the model is cut
     off mid-content. On the rare sample that finishes WITHIN its budget the tag
     is present, but the reward/critic extractor (``extract_explanation_open``)
     takes only the inter-tag content, so the closing tag never reaches the
     critic regardless.

Removing the old token-limit penalty is part of the same change: with random
truncation a generation that hits the cap is the EXPECTED outcome, not a
failure, so ``nla_generate`` no longer promotes ``TRUNCATED -> FAILED`` and
``nla.reward`` no longer skips truncated samples (both gated on ``enabled``).

Configuration (precedence: CLI arg > env var > default):

  --nla-trunc-max-tokens / NLA_TRUNC_MAX_TOKENS   (>0 enables; default 0 = off)
  --nla-trunc-min-tokens / NLA_TRUNC_MIN_TOKENS   (default 1)
  --nla-trunc-seed       / NLA_TRUNC_SEED         (default: args.rollout_seed)

Truncation is OFF unless ``max_tokens > 0`` so ordinary RL runs are unchanged.
"""

import os
import random
from dataclasses import dataclass

from nla.schema import EXPLANATION_OPEN

# What wrap_explanation() prepends to the content in the SFT target
# ("<explanation>\n{text}\n</explanation>"). The well-SFT'd actor opens with
# these tokens, so their count is the content-vs-total offset.
OPENING_PREFIX = EXPLANATION_OPEN + "\n"

# Min content tokens. NOT 1: at very short lengths the online critic (AR) is
# trained to regress a full-text gold activation from a near-empty prefix — an
# impossible target that makes its weights diverge to NaN within a few steps
# (the critic is also the reward model, so the NaN surfaces as NaN rewards). A
# floor of 16 keeps the critic's targets learnable and groups non-degenerate.
DEFAULT_MIN_TOKENS = 16
DEFAULT_MAX_TOKENS = 0  # 0 => truncation disabled


def _arg_or_env_int(args, attr: str, env: str, default: int) -> int:
    """CLI arg (if set, non-None) wins; else env var; else default.

    The argparse defaults are None (not the numeric default) precisely so an
    unset flag falls through to the env var. An empty env string is treated as
    unset too.
    """
    val = getattr(args, attr, None)
    if val is not None:
        return int(val)
    env_val = os.environ.get(env)
    if env_val is not None and env_val != "":
        return int(env_val)
    return default


@dataclass(frozen=True)
class TruncationConfig:
    enabled: bool
    min_tokens: int
    max_tokens: int
    seed: int

    def length_for_group(self, group_index: int) -> int:
        """The shared content-token budget for every sample in this group."""
        return sample_truncation_length(
            self.seed, group_index, self.min_tokens, self.max_tokens
        )


def resolve_truncation_config(args) -> TruncationConfig:
    """Build the TruncationConfig from args (with env fallback).

    Disabled unless ``max_tokens > 0`` — keeps non-truncation RL runs byte-for-
    byte unchanged. When enabled, asserts ``1 <= min <= max`` loudly so a
    mis-set budget fails at startup rather than silently degenerating.
    """
    max_tokens = _arg_or_env_int(args, "nla_trunc_max_tokens", "NLA_TRUNC_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    min_tokens = _arg_or_env_int(args, "nla_trunc_min_tokens", "NLA_TRUNC_MIN_TOKENS", DEFAULT_MIN_TOKENS)
    seed = _arg_or_env_int(args, "nla_trunc_seed", "NLA_TRUNC_SEED", int(getattr(args, "rollout_seed", 0) or 0))
    enabled = max_tokens > 0
    if enabled:
        assert 1 <= min_tokens <= max_tokens, (
            f"NLA truncation: need 1 <= min_tokens ({min_tokens}) <= "
            f"max_tokens ({max_tokens}). Set --nla-trunc-min-tokens / "
            f"--nla-trunc-max-tokens (or NLA_TRUNC_MIN/MAX_TOKENS) accordingly."
        )
    return TruncationConfig(
        enabled=enabled, min_tokens=min_tokens, max_tokens=max_tokens, seed=int(seed)
    )


def sample_truncation_length(seed: int, group_index: int, lo: int, hi: int) -> int:
    """Uniform random int in [lo, hi] (inclusive), deterministic in
    (seed, group_index).

    SHARED PER GROUP: the n_samples_per_prompt members of one prompt share
    ``group_index`` and therefore get the SAME length. ``group_index`` keeps
    increasing across the whole run (it is never reset — see NLADataSource), so
    different groups, and the same prompt drawn on a later step, get different
    lengths; over training the budget cycles through the full [lo, hi] range.

    Seeded by a STRING, not Python ``hash()``: ``random.Random``'s string
    seeding routes through SHA-512, so the draw is stable across processes and
    independent of ``PYTHONHASHSEED`` (every rollout worker must agree on the
    same group's length).
    """
    assert lo >= 1 and hi >= lo, f"bad truncation range [{lo}, {hi}]"
    rng = random.Random(f"nla-trunc:{seed}:{group_index}")
    return rng.randint(lo, hi)


def opening_tag_token_len(tokenizer) -> int:
    """Token count of the fixed ``"<explanation>\\n"`` opening the actor emits.

    Added to the content budget so ``max_new_tokens`` leaves room for the tag
    and the random length is spent on content. Approximate only if the model
    deviates from the SFT opening (rare for a well-trained actor); off-by-a-token
    here just shifts the effective content length by ~1.
    """
    return len(tokenizer.encode(OPENING_PREFIX, add_special_tokens=False))


def max_new_tokens_for_content(content_len: int, opening_offset: int, hard_cap: int | None) -> int:
    """``content_len + opening_offset``, clamped to ``[1, hard_cap]``.

    ``hard_cap`` is the rollout's existing ``max_new_tokens`` (from
    ``--rollout-max-response-len``); we never generate more than it allowed.
    """
    n = content_len + opening_offset
    if hard_cap is not None:
        n = min(n, hard_cap)
    return max(1, n)
