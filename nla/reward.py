"""Reward = -MSE(critic_fwd(explanation), gold_activation) on L2-normalized
vectors (so MSE = 2(1-cos)). Set NLA_LOG_MSE_REWARD=1 to use -log(MSE) instead;
GRPO-normalisation makes them near-equivalent in practice.

Called from miles.rollout.rm_hub via --custom-rm-path nla.reward.nla_rm.
sglang_rollout.py:255 fires this per-sample as each generation completes.

The forward runs on the CRITIC TRAINER (GPUs 2,3, FSDP-sharded, idle during
generation) via Ray remote — live weights, no duplicate model, no checkpoint
staleness. RolloutManager.set_critic_handles (train.py:26) stashed the Ray
handles on args before any rollout fires.

Async accumulator: collect samples until --nla-reward-batch-size is hit (or a
50ms timeout for the tail), then dispatch one batched critic_fwd to both critic
ranks. Event loop stays free during the forward (asyncio.to_thread around ray.get)
so later groups' SGLang callbacks fire → generation pipelines with reward compute.
"""

import asyncio
import functools
import math
import os
import re

import ray
import torch

from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

from nla.config import load_nla_config
from nla.schema import extract_explanation_open, normalize_activation
from nla.truncation import TruncationConfig, resolve_truncation_config, split_into_items


_MSE_EPS = 1e-8
_USE_LOG_MSE_REWARD = bool(int(os.environ.get("NLA_LOG_MSE_REWARD", "0")))
# Per-item length penalty (v2, item 4a): discourages the actor from cramming all
# of the reconstruction-relevant content into one giant first item (which would
# game item-based truncation — one item reconstructs gold, the rest are filler).
# reward = -MSE  -  NLA_ITEM_LEN_PENALTY * Σ_items max(0, item_tokens - target).
# Off by default (coefficient 0). target = NLA_ITEM_LEN_TARGET tokens/item.
_ITEM_LEN_PENALTY = float(os.environ.get("NLA_ITEM_LEN_PENALTY", "0"))
_ITEM_LEN_TARGET = int(os.environ.get("NLA_ITEM_LEN_TARGET", "25"))
# Quote-mark penalty (anti-verbatim-quoting experiment): discourages the actor
# from quoting input text verbatim (especially echoing the last context token)
# by penalizing EVERY quotation-mark character in the explanation:
# reward = -MSE - NLA_QUOTE_PENALTY * (# quote chars). Deliberately naive and
# purely lexical — the point is to see whether removing the *marks* removes
# the *behavior* (re-routes content into paraphrase) or just its punctuation.
# Off by default (coefficient 0).
_QUOTE_PENALTY = float(os.environ.get("NLA_QUOTE_PENALTY", "0"))
# "Any kind of quotation mark" — includes the typewriter apostrophe (so
# contractions/possessives are penalized too; intentional per experiment
# design), backtick, all the curly/angle/CJK variants, fullwidth AND halfwidth
# forms, ornamental (❛❜❝❞) and double-low-reversed-9 (⹂) marks. Keep in sync
# with QUOTES in experiments/kitft-quote-penalty-rl/quote_stats_wandb.py.
_QUOTE_CHARS = frozenset("\"'`‘’‚‛“”„‟«»‹›「」『』〝〞〟＂＇｀｢｣❛❜❝❞⹂")
# Verbatim-repeat penalty (anti-echo experiment, generalizes the quote-mark
# penalty): the AV never SEES the input text — it only gets the activation —
# so any long verbatim match between its explanation and the context is
# reconstruction-by-echo. Penalize "copy coverage":
#   reward = -MSE - NLA_REPEAT_PENALTY * (# explanation chars lying inside a
#            common substring of length >= NLA_REPEAT_MIN_CHARS with the
#            sample's context, both lowercased + whitespace-collapsed).
# The context is the sample's detokenized_text_truncated (text up to the
# extraction position), carried by the RL parquet (--keep-debug-metadata,
# default on) into Sample.metadata via NLADataSource. The min-length threshold
# keeps incidental short matches ("of the") free; coverage (union of matching
# windows) counts every copied span, not just the longest. Off by default.
_REPEAT_PENALTY = float(os.environ.get("NLA_REPEAT_PENALTY", "0"))
_REPEAT_MIN_CHARS = int(os.environ.get("NLA_REPEAT_MIN_CHARS", "12"))

_WS_RE = re.compile(r"\s+")


def _norm_overlap(s: str) -> str:
    return _WS_RE.sub(" ", s.lower()).strip()


@functools.lru_cache(maxsize=1024)
def _ctx_shingles(ctx_raw: str) -> frozenset:
    """All NLA_REPEAT_MIN_CHARS-length substrings of the normalized context.
    Cached on the RAW string: all samples of a group share one context, so
    each unique context is shingled once per drain, not 8 times."""
    c = _norm_overlap(ctx_raw)
    L = _REPEAT_MIN_CHARS
    return frozenset(c[i:i + L] for i in range(max(0, len(c) - L + 1)))


def _copied_chars(expl: str, ctx_raw: str) -> int:
    """Chars of the normalized explanation covered by >=1 length-L window that
    appears verbatim in the normalized context (union of matching windows)."""
    L = _REPEAT_MIN_CHARS
    e = _norm_overlap(expl)
    if len(e) < L or not ctx_raw:
        return 0
    sh = _ctx_shingles(ctx_raw)
    covered = 0
    covered_until = 0
    for i in range(len(e) - L + 1):
        if e[i:i + L] in sh:
            covered += i + L - max(covered_until, i)
            covered_until = i + L
    return covered


def _repeat_penalty(expl: str, ctx_raw) -> float:
    if _REPEAT_PENALTY <= 0:
        return 0.0
    if not ctx_raw:
        raise RuntimeError(
            "NLA_REPEAT_PENALTY is set but this sample has no "
            "detokenized_text_truncated in metadata — the RL parquet was built "
            "without --keep-debug-metadata. Rebuild it (the flag defaults on) "
            "or unset the penalty."
        )
    return -_REPEAT_PENALTY * _copied_chars(expl, ctx_raw)


# Under -mse_nrm, 0.0 is the BEST reward (perfect reconstruction) and -2.0 is
# orthogonal. Under -log(MSE), 0.0 corresponds to mse=1 (mid-range). Use the
# orthogonal-equivalent value so a failed extraction is never advantaged.
FAILED_EXTRACTION_REWARD = -math.log(2.0) if _USE_LOG_MSE_REWARD else -2.0
# Flush timeout: originally 50ms for single-sample async_rm path where
# samples arrive fast (per-sample, not per-group) and 50ms catches tail
# stragglers. With --group-rm routing through the accumulator, groups arrive
# staggered over the ~60s generation window — first group lands alone, 50ms
# fires before more arrive, batch-size=256 never kicks in. 5s lets ~10-30
# groups coalesce; adds ≤5s latency in a 100s+ rollout. Override with
# NLA_REWARD_FLUSH_SECS if the stagger pattern differs.
_TAIL_FLUSH_SECONDS = float(os.environ.get("NLA_REWARD_FLUSH_SECS", "5.0"))

_TOKENIZER = None
_CFG = None
# Random-length truncation config — when enabled, TRUNCATED samples are scored
# (not penalised). Set in _lazy_init, read in _prep_batch. See nla.truncation.
_TRUNC: TruncationConfig | None = None

_pending: list[tuple[Sample, asyncio.Future]] = []
_drain_task: asyncio.Task | None = None


def _lazy_init(args):
    global _TOKENIZER, _CFG, _TRUNC
    if _TOKENIZER is not None:
        return
    _TRUNC = resolve_truncation_config(args)
    # Tokenizer and sidecar from the critic's HF dir. FSDP: args.critic_load IS
    # the HF dir. Megatron: critic_load is torch_dist (no tokenizer, no sidecar),
    # so --nla-critic-sidecar-source must point at the FSDP-generated HF dir.
    # Same arg the trainer-side critic uses for its sidecar — single source of truth.
    sidecar_dir = args.nla_critic_sidecar_source or args.critic_load
    _TOKENIZER = load_tokenizer(sidecar_dir, trust_remote_code=True)
    # Megatron critic_fwd passes attention_mask=None (causal-only). With left-pad
    # the last real token attends left to padding → corrupted. Right-pad puts padding
    # after the last real token where causal never reaches. FSDP doesn't care (passes
    # the mask through), so this is a no-op there. Defense-in-depth for older critic
    # checkpoints saved before prepare_critic_checkpoint forced right-pad.
    _TOKENIZER.padding_side = "right"
    _CFG = load_nla_config(sidecar_dir, _TOKENIZER)
    assert _CFG.critic_prompt_template is not None, (
        f"critic sidecar at {sidecar_dir!r} has no critic_prompt_template"
    )


def _item_length_penalty(items: list[str]) -> float:
    """Reward shaping: -coef * Σ_items max(0, item_tokens - target). 0 when off
    or when every item is within target. Penalizes long individual items so the
    actor spreads content across items instead of one giant one (see _ITEM_LEN_*).

    ITEMS MODE ONLY: the penalty exists to stop the actor gaming item-count
    truncation by cramming everything into one item. Token-mode truncation
    can't be gamed that way, so a lingering NLA_ITEM_LEN_PENALTY export from a
    v2 shell must not shape v3 rewards."""
    if _ITEM_LEN_PENALTY <= 0 or not items:
        return 0.0
    if _TRUNC is None or _TRUNC.mode != "items":
        return 0.0
    excess = 0
    for it in items:
        n = len(_TOKENIZER(it, add_special_tokens=False)["input_ids"])
        excess += max(0, n - _ITEM_LEN_TARGET)
    return -_ITEM_LEN_PENALTY * excess


def _quote_penalty(expl: str) -> float:
    """Reward shaping: -coef * (# quotation-mark chars in the explanation).
    0 when off. Counted on the raw explanation STRING (not tokens) so every
    mark costs the same regardless of how the tokenizer merges it."""
    if _QUOTE_PENALTY <= 0:
        return 0.0
    n = sum(1 for ch in expl if ch in _QUOTE_CHARS)
    return -_QUOTE_PENALTY * n


# Copied-bits penalty (anti-verbatim-repetition, the principled successor to the
# quote-mark penalty): charge the actor per BIT of input text reproduced
# verbatim, beyond what chance phrase collision explains:
#   reward -= NLA_OVERLAP_PENALTY * Σ_spans max(0, bits(span) - B0)
# where spans are maximal runs of >= NLA_OVERLAP_MIN_SPAN consecutive words
# (lowercase \w+, punctuation-blind) shared between the explanation and the
# sample's input context (metadata["detokenized_text_truncated"], carried by
# building the RL parquet with --keep-debug-metadata), and
# bits(span) = Σ -log2 p(word) under corpus unigram counts (NLA_OVERLAP_UNIGRAMS,
# JSON built by the experiment's parquet-build script). B0 defaults to 15 bits
# ~= log2(context_words × generation_words), the birthday bound at which a
# span's occurrence in both texts stops being explainable by chance — so common
# phrases are free and charged bits are log-likelihood evidence of copying.
# Single-word matches are always free (min span 2): naming input content is the
# AV's job; the one-word final-token echo is legitimate activation content.
# Off by default (coefficient 0).
_OVERLAP_PENALTY = float(os.environ.get("NLA_OVERLAP_PENALTY", "0"))
_OVERLAP_DEDUCTIBLE = float(os.environ.get("NLA_OVERLAP_DEDUCTIBLE", "15"))
_OVERLAP_MIN_SPAN = int(os.environ.get("NLA_OVERLAP_MIN_SPAN", "2"))
_OVERLAP_UNIGRAMS_PATH = os.environ.get("NLA_OVERLAP_UNIGRAMS")
_WORD_RE = re.compile(r"\w+")
_UNIGRAMS: dict | None = None


def _load_unigrams() -> dict:
    global _UNIGRAMS
    if _UNIGRAMS is None:
        import json
        assert _OVERLAP_UNIGRAMS_PATH, (
            "NLA_OVERLAP_PENALTY > 0 requires NLA_OVERLAP_UNIGRAMS (JSON with "
            "__total__/__vocab__/counts, built by the experiment's parquet-build script)"
        )
        with open(_OVERLAP_UNIGRAMS_PATH) as f:
            d = json.load(f)
        _UNIGRAMS = {
            "counts": d["counts"],
            # add-1 smoothing over the FULL vocab (incl. pruned singletons) so
            # unseen words price at the ceiling, not at count=1.
            "denom": d["__total__"] + d["__vocab__"],
        }
    return _UNIGRAMS


def _surprisal(word: str) -> float:
    u = _load_unigrams()
    return -math.log2((u["counts"].get(word, 0) + 1) / u["denom"])


def _copied_bits(expl: str, ctx: str) -> float:
    """Charged bits: Σ_spans max(0, span_bits - B0) over maximal shared word
    runs of length >= _OVERLAP_MIN_SPAN. Greedy maximal-fragment decomposition
    (Grusky et al. 2018)."""
    gen = _WORD_RE.findall(expl.lower())
    src = _WORD_RE.findall(ctx.lower())
    if not gen or not src:
        return 0.0
    pos: dict[str, list[int]] = {}
    for j, w in enumerate(src):
        pos.setdefault(w, []).append(j)
    charged, i = 0.0, 0
    while i < len(gen):
        best = 0
        for j in pos.get(gen[i], ()):
            k = 0
            while i + k < len(gen) and j + k < len(src) and gen[i + k] == src[j + k]:
                k += 1
            best = max(best, k)
        if best >= max(_OVERLAP_MIN_SPAN, 1):
            span_bits = sum(_surprisal(w) for w in gen[i:i + best])
            charged += max(0.0, span_bits - _OVERLAP_DEDUCTIBLE)
        i += best if best else 1
    return charged


def _overlap_penalty(expl: str, ctx: str | None) -> float:
    """Reward shaping: -coef * charged copied bits. 0 when off. Fails loud if
    enabled but the sample carries no context text (RL parquet built without
    --keep-debug-metadata)."""
    if _OVERLAP_PENALTY <= 0:
        return 0.0
    assert ctx, (
        "NLA_OVERLAP_PENALTY > 0 but sample metadata has no "
        "detokenized_text_truncated — rebuild the RL parquet with "
        "--keep-debug-metadata"
    )
    return -_OVERLAP_PENALTY * _copied_bits(expl, ctx)


# Full-batch quote metrics, appended per reward drain as JSON lines. Unlike the
# NLA_ROLLOUT_TEXT_DUMP (first 20 samples, overwritten), this covers EVERY
# scored sample — the source of truth for quote-usage curves. A sidecar
# (quote_stats_wandb.py) or post-hoc analysis reads it. Off unless set.
_QUOTE_STATS_JSONL = os.environ.get("NLA_QUOTE_STATS_JSONL")


def _dump_quote_stats(explanations: list[str], contexts: list | None = None) -> None:
    if not _QUOTE_STATS_JSONL or not explanations:
        return
    import json
    counts = [sum(1 for ch in e if ch in _QUOTE_CHARS) for e in explanations]
    rec = {
        "n": len(counts),
        "quote_chars_mean": sum(counts) / len(counts),
        "quote_chars_max": max(counts),
        "frac_zero": sum(1 for c in counts if c == 0) / len(counts),
    }
    if _REPEAT_PENALTY > 0 and contexts:
        cov = [_copied_chars(e, c) if c else 0
               for e, c in zip(explanations, contexts, strict=True)]
        rec.update({
            "repeat_covered_mean": sum(cov) / len(cov),
            "repeat_covered_max": max(cov),
            "repeat_frac_zero": sum(1 for c in cov if c == 0) / len(cov),
        })
    if _OVERLAP_PENALTY > 0 and contexts:
        bits = [_copied_bits(e, c) if c else 0.0
                for e, c in zip(explanations, contexts, strict=True)]
        rec.update({
            "overlap_bits_mean": sum(bits) / len(bits),
            "overlap_bits_max": max(bits),
            "overlap_frac_zero": sum(1 for b in bits if b < 0.5) / len(bits),
        })
    with open(_QUOTE_STATS_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _prep_batch(samples: list[Sample]):
    """Extract explanations, tokenize, stack golds. Returns (payload, orig_idx,
    penalties) for the subset with valid extractions; FAILED ones get the fixed
    penalty. ``penalties`` is the per-item length penalty aligned with orig_idx
    (added to the -MSE reward in _drain)."""
    dump_path = os.environ.get("NLA_ROLLOUT_TEXT_DUMP")
    if dump_path:
        with open(dump_path, "w") as f:
            for i, s in enumerate(samples[:20]):
                f.write(f"=== sample {i} (status={s.status.name}) ===\n{s.response}\n\n")
    # When random-length truncation is on, hitting the (random) cap is the
    # EXPECTED outcome, so TRUNCATED samples must be scored, not skipped. When
    # it is off, keep the legacy behaviour: only COMPLETED samples go through the
    # critic (nla_generate promotes TRUNCATED→FAILED to avoid length drift —
    # trunc-with-tag would otherwise score ≈3.63 → adv≈+1.2σ and push length up).
    trunc_on = _TRUNC is not None and _TRUNC.enabled
    scoreable = (
        (Sample.Status.COMPLETED, Sample.Status.TRUNCATED) if trunc_on
        else (Sample.Status.COMPLETED,)
    )
    prompts, golds, orig_idx, penalties, expls, ctxs = [], [], [], [], [], []
    for i, s in enumerate(samples):
        if s.status not in scoreable:
            continue
        # extract_explanation_open tolerates the missing </explanation> left by
        # mid-content truncation, and (v2 untagged) returns the whole response
        # when there is no <explanation> tag.
        expl = extract_explanation_open(s.response)
        if expl is not None:
            ctx = s.metadata.get("detokenized_text_truncated")
            prompts.append(_CFG.critic_prompt_template.format(explanation=expl))
            golds.append(s.metadata["activation_vector"])
            orig_idx.append(i)
            expls.append(expl)
            ctxs.append(ctx)
            penalties.append(
                _item_length_penalty(split_into_items(expl))
                + _quote_penalty(expl)
                + _repeat_penalty(expl, ctx)
                + _overlap_penalty(expl, ctx)
            )
    if not prompts:
        return None, [], []
    _dump_quote_stats(expls, ctxs)
    # add_special_tokens=True matches stage0 extractor (extractors.py:131).
    # Gemma needs BOS here; Qwen has bos_token=None (no-op). See sft_critic.py.
    tok = _TOKENIZER(prompts, add_special_tokens=True, padding=True, return_tensors="pt")
    gold = torch.tensor(golds, dtype=torch.float32)  # [B, d]
    return (tok["input_ids"], tok["attention_mask"], gold), orig_idx, penalties


def _mse_to_reward(pred: torch.Tensor, gold: torch.Tensor, scale: float) -> list[float]:
    pn = normalize_activation(pred, scale)
    gn = normalize_activation(gold, scale)
    mse = ((pn - gn) ** 2).mean(dim=1)  # [B]
    if _USE_LOG_MSE_REWARD:
        out = [-math.log(max(m, _MSE_EPS)) for m in mse.tolist()]
    else:
        out = (-mse).tolist()
    # NaN guard: a non-finite critic prediction (inf/NaN — e.g. a diverged online
    # critic, or a bad forward) yields a non-finite MSE → NaN reward, which poisons
    # GRPO (one NaN advantage corrupts the whole step). Map non-finite rewards to
    # the failed-extraction penalty so they can't propagate. math.isfinite catches
    # both NaN and ±inf; note max(NaN, eps) == NaN, so the log path needs this too.
    # This is a SAFETY NET, not a fix: if the critic *weights* go NaN every reward
    # becomes the penalty — keep the critic stable (min content length, see
    # nla.truncation) so this never fires in normal operation.
    return [r if math.isfinite(r) else FAILED_EXTRACTION_REWARD for r in out]


async def _drain(args):
    global _drain_task
    _drain_task = None
    batch, _pending[:] = _pending[:], []
    if not batch:
        return
    # Once `batch` is detached from _pending, any failure below would orphan the
    # awaiting futures (nla_rm callers hang forever on critic OOM / NCCL timeout).
    # Propagate the exception to every unresolved future, then re-raise so the
    # rollout worker itself dies loudly instead of silently stalling.
    try:
        samples = [s for s, _ in batch]
        rewards = [FAILED_EXTRACTION_REWARD] * len(samples)

        payload, orig_idx, penalties = _prep_batch(samples)
        if payload is not None:
            ids, mask, gold = payload
            # All critic ranks must participate in FSDP's per-layer all-gather.
            # Dispatch to every handle; results are identical, take rank 0's.
            # to_thread: ray.get blocks but releases GIL → event loop proceeds.
            handles = args._nla_critic_handles
            refs = [h.critic_fwd.remote(ids, mask) for h in handles]
            pred = await asyncio.to_thread(lambda: ray.get(refs)[0])  # [B, d] CPU
            base = _mse_to_reward(pred, gold, _CFG.mse_scale)
            for j, r, pen in zip(orig_idx, base, penalties, strict=True):
                # Don't shape a failed-extraction penalty (already the floor).
                rewards[j] = r if r == FAILED_EXTRACTION_REWARD else r + pen

        for (_, fut), r in zip(batch, rewards, strict=True):
            fut.set_result(r)
    except BaseException as exc:
        for _, fut in batch:
            if not fut.done():
                fut.set_exception(exc)
        raise


async def _flush_after_timeout(args):
    await asyncio.sleep(_TAIL_FLUSH_SECONDS)
    if _drain_task is not None:  # still us, nobody drained in the window
        await _drain(args)


async def nla_rm(args, sample_or_samples, **_kwargs):
    global _drain_task
    _lazy_init(args)

    # batched_async_rm path (--group-rm): group of 8 arrives as a list.
    #
    # OLD: dispatch critic_fwd immediately per group → 64 serial ~2s critic_fwd
    # calls (FSDP collective, one-at-a-time across 6 ranks) = ~128s. This was
    # the ACTUAL rollout bottleneck at 27b — not SGLang, not event-loop blocking.
    # Observed 2.44s between group completions = exactly critic_fwd latency.
    # Generation is parallel on different GPUs but reward-via-critic serializes.
    #
    # NEW: route group members through the accumulator. With _TAIL_FLUSH=0.05s,
    # multiple concurrent groups (all finishing their gather at similar times)
    # coalesce into one big critic_fwd batch. 64 groups × 8 = 512 samples could
    # be 1-4 critic_fwd calls instead of 64. Total reward: ~10-20s vs ~128s.
    if isinstance(sample_or_samples, list):
        futs = []
        for s in sample_or_samples:
            fut = asyncio.get_running_loop().create_future()
            _pending.append((s, fut))
            futs.append(fut)
        if len(_pending) >= args.nla_reward_batch_size:
            if _drain_task is not None:
                _drain_task.cancel()
            await _drain(args)
        elif _drain_task is None:
            _drain_task = asyncio.create_task(_flush_after_timeout(args))
        return await asyncio.gather(*futs)

    # per-sample path (default): accumulate across concurrent coroutines.
    fut = asyncio.get_running_loop().create_future()
    _pending.append((sample_or_samples, fut))
    if len(_pending) >= args.nla_reward_batch_size:
        if _drain_task is not None:
            _drain_task.cancel()
        await _drain(args)
    elif _drain_task is None:
        _drain_task = asyncio.create_task(_flush_after_timeout(args))
    return await fut
