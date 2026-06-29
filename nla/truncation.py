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

TWO MODES (``--nla-trunc-mode`` / ``NLA_TRUNC_MODE``):

  - ``tokens`` (default, v1): the budget is a number of CONTENT TOKENS, drawn
    uniformly from [min_tokens, max_tokens], applied by lowering
    ``max_new_tokens`` (cap, don't post-truncate). Unchanged from v1.

  - ``items`` (v2): the budget is a number of newline-separated LIST ITEMS,
    drawn from a TAPERING distribution over [1, max_items] (more mass on short
    prefixes, where the front-loading signal is strongest), optionally annealed
    by a CURRICULUM (start long → shrink, so the online critic sees easy long
    prefixes early). A token cap can't express "K items", so item mode generates
    to the hard cap and POST-TRUNCATES the rollout at the K-th newline (see
    ``item_truncation_cut``) — the actor's tokens/loss-mask/logprobs are sliced
    to the kept prefix in nla_generate. The per-group-shared and content-only
    properties (1, 2 above) are preserved; only the "cap, don't post-truncate"
    mechanism (3) is replaced, because item boundaries are not known until the
    text exists.

Configuration (precedence: CLI arg > env var > default):

  --nla-trunc-mode       / NLA_TRUNC_MODE         (tokens | items; default tokens)
  --nla-trunc-max-tokens / NLA_TRUNC_MAX_TOKENS   (tokens mode; >0 enables; default 0 = off)
  --nla-trunc-min-tokens / NLA_TRUNC_MIN_TOKENS   (tokens mode; default 16)
  --nla-trunc-max-items  / NLA_TRUNC_MAX_ITEMS    (items mode; >0 enables; default 0 = off)
  --nla-trunc-taper      / NLA_TRUNC_TAPER        (items mode; >1 favors short; default 2.0)
  --nla-trunc-curriculum-groups / NLA_TRUNC_CURRICULUM_GROUPS
                                                  (items mode; anneal long→short over this many
                                                   groups; 0 = off, full taper from the start)
  --nla-trunc-seed       / NLA_TRUNC_SEED         (default: args.rollout_seed)

Truncation is OFF unless the active mode's enabling budget (``max_tokens`` for
tokens, ``max_items`` for items) is > 0, so ordinary RL runs are unchanged.
"""

import os
import random
import re
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

# Item mode (v2) defaults.
DEFAULT_MODE = "tokens"        # "tokens" (v1) | "items" (v2)
DEFAULT_MAX_ITEMS = 0          # 0 => item-mode truncation disabled
DEFAULT_TAPER_POWER = 2.0      # >1 biases the draw toward FEWER items (short prefixes)
DEFAULT_CURRICULUM_GROUPS = 0  # 0 => no curriculum (full taper from step 0)


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
    # v2 item-mode fields — defaulted so v1 callers (and tests) that build the
    # config with only the four token-mode kwargs keep working unchanged.
    mode: str = DEFAULT_MODE
    max_items: int = DEFAULT_MAX_ITEMS
    taper_power: float = DEFAULT_TAPER_POWER
    curriculum_groups: int = DEFAULT_CURRICULUM_GROUPS

    def length_for_group(self, group_index: int) -> int:
        """The shared content-token budget for every sample in this group (tokens mode)."""
        return sample_truncation_length(
            self.seed, group_index, self.min_tokens, self.max_tokens
        )

    def items_for_group(self, group_index: int) -> int:
        """The shared content-ITEM budget for every sample in this group (items mode).

        Shared per group for the same GRPO-cleanliness reason as the token budget
        (see module docstring decision 1): a within-group reward gap should reflect
        content quality at a fixed prefix length, not a length the model can't control.
        """
        return sample_item_count(
            self.seed, group_index, self.max_items, self.taper_power, self.curriculum_groups
        )


def _arg_or_env_str(args, attr: str, env: str, default: str) -> str:
    val = getattr(args, attr, None)
    if val is not None and val != "":
        return str(val)
    env_val = os.environ.get(env)
    if env_val is not None and env_val != "":
        return env_val
    return default


def _arg_or_env_float(args, attr: str, env: str, default: float) -> float:
    val = getattr(args, attr, None)
    if val is not None:
        return float(val)
    env_val = os.environ.get(env)
    if env_val is not None and env_val != "":
        return float(env_val)
    return default


def resolve_truncation_config(args) -> TruncationConfig:
    """Build the TruncationConfig from args (with env fallback).

    Disabled unless the active mode's enabling budget is > 0 (``max_tokens`` for
    tokens mode, ``max_items`` for items mode) — keeps non-truncation RL runs
    byte-for-byte unchanged. When enabled, asserts the budget is sane loudly so a
    mis-set config fails at startup rather than silently degenerating.
    """
    mode = _arg_or_env_str(args, "nla_trunc_mode", "NLA_TRUNC_MODE", DEFAULT_MODE)
    assert mode in ("tokens", "items"), (
        f"NLA truncation mode must be 'tokens' or 'items', got {mode!r}"
    )
    max_tokens = _arg_or_env_int(args, "nla_trunc_max_tokens", "NLA_TRUNC_MAX_TOKENS", DEFAULT_MAX_TOKENS)
    min_tokens = _arg_or_env_int(args, "nla_trunc_min_tokens", "NLA_TRUNC_MIN_TOKENS", DEFAULT_MIN_TOKENS)
    max_items = _arg_or_env_int(args, "nla_trunc_max_items", "NLA_TRUNC_MAX_ITEMS", DEFAULT_MAX_ITEMS)
    taper_power = _arg_or_env_float(args, "nla_trunc_taper", "NLA_TRUNC_TAPER", DEFAULT_TAPER_POWER)
    curriculum_groups = _arg_or_env_int(
        args, "nla_trunc_curriculum_groups", "NLA_TRUNC_CURRICULUM_GROUPS", DEFAULT_CURRICULUM_GROUPS
    )
    seed = _arg_or_env_int(args, "nla_trunc_seed", "NLA_TRUNC_SEED", int(getattr(args, "rollout_seed", 0) or 0))

    if mode == "tokens":
        enabled = max_tokens > 0
        if enabled:
            assert 1 <= min_tokens <= max_tokens, (
                f"NLA truncation: need 1 <= min_tokens ({min_tokens}) <= "
                f"max_tokens ({max_tokens}). Set --nla-trunc-min-tokens / "
                f"--nla-trunc-max-tokens (or NLA_TRUNC_MIN/MAX_TOKENS) accordingly."
            )
    else:  # items
        enabled = max_items > 0
        if enabled:
            assert max_items >= 1, f"NLA truncation: need max_items >= 1, got {max_items}"
            assert taper_power > 0, f"NLA truncation: need taper_power > 0, got {taper_power}"
            assert curriculum_groups >= 0, (
                f"NLA truncation: need curriculum_groups >= 0, got {curriculum_groups}"
            )
    return TruncationConfig(
        enabled=enabled, min_tokens=min_tokens, max_tokens=max_tokens, seed=int(seed),
        mode=mode, max_items=max_items, taper_power=taper_power,
        curriculum_groups=curriculum_groups,
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


def sample_item_count(
    seed: int, group_index: int, max_items: int, taper_power: float, curriculum_groups: int
) -> int:
    """Number of newline-separated list items to keep (items mode), in [1, max_items].

    TAPER: a uniform draw u∈[0,1) is reshaped by ``u ** taper_power``. With
    taper_power > 1 the mass concentrates near 0 → toward FEWER items, i.e. short
    prefixes, where the front-loading pressure is strongest. taper_power == 1 is
    uniform; < 1 would favor longer prefixes.

    CURRICULUM: a floor on the item count that starts at ``max_items`` and decays
    linearly to 1 over the first ``curriculum_groups`` groups. Early in training
    the budget is forced long (the online critic — also the reward model — sees
    easy, information-rich prefixes), then the floor drops and the taper opens up
    the short end. ``curriculum_groups == 0`` disables it (full taper from group 0).
    ``group_index`` increases monotonically across the run, so it doubles as a
    training-progress clock without threading the step count through the rollout.

    Deterministic in (seed, group_index) and SHARED across a group's members
    (same group_index → same count) for clean GRPO within-group comparison.
    String-seeded random.Random (SHA-512) → stable across processes / PYTHONHASHSEED.
    """
    assert max_items >= 1, f"max_items must be >= 1, got {max_items}"
    if curriculum_groups > 0:
        progress = min(1.0, group_index / curriculum_groups)
    else:
        progress = 1.0
    floor_k = round(max_items - progress * (max_items - 1))  # max_items (progress 0) → 1 (progress 1)
    floor_k = max(1, min(max_items, floor_k))
    rng = random.Random(f"nla-trunc-items:{seed}:{group_index}")
    biased = rng.random() ** taper_power           # ∈ [0, 1), skewed low for taper_power > 1
    span = max_items - floor_k + 1
    k = floor_k + int(biased * span)
    return max(floor_k, min(max_items, k))


def split_into_items(text: str) -> list[str]:
    """Split an explanation into list items on runs of newlines; drop blanks.

    The single source of truth for "what is a list item" — used by the item-count
    truncation (RL + AR warm-start) and by the reward's per-item length penalty,
    so they always agree on item boundaries.
    """
    return [it.strip() for it in re.split(r"\n+", text) if it.strip()]


def _kth_item_char_end(text: str, k: int) -> int | None:
    """Character index in ``text`` where the k-th list item ends (start of the
    k-th newline run), or None if there are fewer than k items (nothing to cut).
    """
    runs = list(re.finditer(r"\n+", text))
    if len(runs) < k:
        return None  # <= k items already — keep everything
    return runs[k - 1].start()


def item_truncation_cut(decode_fn, response_token_ids: list[int], k: int) -> int:
    """How many of ``response_token_ids`` to KEEP so the decoded text holds k items.

    ``decode_fn(list[int]) -> str`` decodes a token-id prefix (e.g.
    ``tokenizer.decode``). We find the character offset where the k-th item ends
    (``_kth_item_char_end``), then binary-search the smallest token-prefix length
    whose decoding reaches that offset. Binary search is valid because
    len(decode(ids[:t])) is monotonic non-decreasing in t. Returns
    ``len(response_token_ids)`` (no truncation) when there are <= k items.

    Token-space (not char-space) truncation is what the rollout needs: the actor's
    tokens, rollout logprobs and loss mask are per-token and must be sliced at the
    same index to stay aligned for GRPO.
    """
    if k < 1:
        k = 1
    n = len(response_token_ids)
    if n == 0:
        return 0
    full = decode_fn(response_token_ids)
    char_end = _kth_item_char_end(full, k)
    if char_end is None or char_end <= 0:
        return n
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if len(decode_fn(response_token_ids[:mid])) >= char_end:
            hi = mid
        else:
            lo = mid + 1
    return max(1, lo)


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
