"""Stage 3: Build final training parquets — {av_sft,ar_sft,rl}.parquet + complete sidecars.

Transforms base-schema rows into the stage-specific training format:

  AV-SFT (actor SFT):
    prompt    list[dict]  [{"role":"user","content": actor_template with <INJECT> literal}]
    response  str         "<explanation>\n{api_explanation}\n</explanation>"
    activation_vector

  AR-SFT (critic SFT):
    prompt    str         critic_template filled with api_explanation. Ends with
                          a known suffix (e.g. `</text> <summary>`).
    activation_vector
    EVERY ROW tokenized at build time — assert it ends with the expected
    suffix IDs. Training extracts at the last-token position (no scanning,
    just tokens[-1] — GPU-friendly).

  RL:
    Same as AV-SFT minus response.

Token IDs are NOT per-row columns — they're dataset constants, shipped once
in the sidecar, loaded once by training (v4 contract).

The parquet `prompt` column contains the literal `<INJECT>` placeholder —
training-side NLADataSource swaps it for ㊗ at load time. We compute and
write the REAL ㊗ token ID + neighbor IDs to the sidecar so training can
verify against its live tokenizer.

Activation vectors are passed through RAW (norm="none"). Normalization is a
training-time decision — NLADataSource or the training loop can normalize
as needed. Data-gen's job is just "get vectors out of the model".
"""

import argparse
import random
from dataclasses import replace
from typing import Any  # noqa: F401 — tokenizer: Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from nla.datagen._common import add_storage_args, load_tokenizer, make_storage
from nla.datagen.injection_tokens import build_token_meta
from nla.datagen.sidecar import read_sidecar, write_sidecar
from nla.schema import wrap_explanation
from nla.truncation import sample_item_count, sample_truncation_length, split_into_items

_INJECT_PLACEHOLDER = "<INJECT>"

# Matches the paper appendix prompt template, but 2-3 snippets
# (not 4-5) since that's what our stage2 API prompt produces. The
# {injection_char} slot is what becomes <INJECT> in the parquet (then the real
# injection char at training time). Neighbor-ID computation in build_token_meta
# looks at the bytes immediately around {injection_char}, so keep the
# <concept>...</concept> wrapping tight.
_DEFAULT_ACTOR_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

# v2 (--explanation-format list): NO <explanation> wrapper and NO "paragraph"
# framing — the actor's raw output IS a newline-separated list, one short item
# per line, salience-ordered. Removing the closing tag (+ never-train-EOS in the
# actor SFT) is what lets the list generalize to arbitrary length. The
# <concept>{injection_char}</concept> wrapping is IDENTICAL to the tagged template
# so the injection token + neighbor IDs are unchanged.
_ACTOR_TEMPLATE_LIST = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce a list of short text descriptions of that vector, one per line, ordered from the most to the least salient aspect.

Here is the vector:

<concept>{injection_char}</concept>

Please provide the list."""
# v3 (--explanation-format bullets): same untagged / never-stop design and the
# same OUTPUT TEXT as list — the difference is the prompt actually SAYS bullet
# points. v2 shipped with a prompt that still described "2-3 text snippets" in
# <explanation> tags (05_build_rl_parquet built the RL parquet without the
# format flag), so the actor's instructions and its trained output format
# disagreed. The targets deliberately have NO literal "- " markers: each would
# cost ~1.2 tokens of the truncation budget per item with zero reconstruction
# information, and would break per-token comparability with v1/v2. The
# <concept>{injection_char}</concept> wrapping is again IDENTICAL so the
# injection token + neighbor IDs are unchanged.
_ACTOR_TEMPLATE_BULLETS = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then describe that vector as a list of bullet points, one per line, ordered from the most to the least salient aspect.

Here is the vector:

<concept>{injection_char}</concept>

Please provide the bullet points."""

_ACTOR_TEMPLATES = {
    "tagged": _DEFAULT_ACTOR_TEMPLATE,
    "list": _ACTOR_TEMPLATE_LIST,
    "bullets": _ACTOR_TEMPLATE_BULLETS,
}

# Critic template ends with a fixed suffix (no marker char). Training extracts
# at the last-token position of the tokenized prompt. The suffix token IDs are
# recorded in the sidecar so training can verify the tail matches.
_DEFAULT_CRITIC_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"
_CHUNK_SIZE = 4096

# Provenance: always carried (cheap — a few bytes/row, always useful)
_PROVENANCE_COLS = ["n_raw_tokens", "activation_layer", "doc_id"]
# Heavy debug: gated on --keep-debug-metadata (detokenized text can dominate file size)
_HEAVY_DEBUG_COLS = ["detokenized_text_truncated"]

# Explicit schemas: activation_vector passes through as arrow arrays (never
# hits Python), and struct/list types aren't inferable from Python dicts anyway.
_PROMPT_STRUCT = pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))
_PROVENANCE_FIELDS = [
    ("n_raw_tokens", pa.int64()),
    ("activation_layer", pa.int64()),
    ("doc_id", pa.string()),
]
_HEAVY_DEBUG_FIELDS = [
    ("detokenized_text_truncated", pa.string()),
]


def _schema_for(stage: str, keep_heavy_debug: bool, d_model: int) -> pa.Schema:
    # FixedSizeList for activation_vector — all vectors are exactly d_model
    # wide. Variable-length ListArray silently corrupts under take() when the
    # values buffer exceeds 4 GiB (hit on 100k RL: 500k × 3584 × 4 = 6.7 GiB).
    av = pa.list_(pa.float32(), d_model)
    match stage:
        case "av_sft":
            core = [
                ("prompt", _PROMPT_STRUCT),
                ("response", pa.string()),
                ("activation_vector", av),
            ]
        case "rl":
            core = [
                ("prompt", _PROMPT_STRUCT),
                ("activation_vector", av),
            ]
        case "ar_sft":
            core = [
                ("prompt", pa.string()),
                ("activation_vector", av),
            ]
        case _:
            raise AssertionError(f"unreachable: stage={stage!r}")
    fields = core + _PROVENANCE_FIELDS
    if keep_heavy_debug:
        fields += _HEAVY_DEBUG_FIELDS
    return pa.schema(fields)


def _format_items(expl: str, fmt: str) -> str:
    """Normalize a raw api_explanation into the actor-output text for ``fmt``.

    tagged (v1): the raw explanation unchanged (the <explanation> wrapper is the
    AV response's job, not the text's — the critic sees it unwrapped).
    list (v2) and bullets (v3): one item per line (collapse \\n\\n paragraphs →
    \\n). bullets differs from list only in the PROMPT wording — the output text
    is identical, with no literal "- " markers (see _ACTOR_TEMPLATE_BULLETS).
    Normalizing here (instead of at stage2) lets any format build from existing
    base parquets without re-running the API. Used for BOTH the AV-SFT response
    and the AR-SFT critic input so the critic is trained on exactly the text
    the actor is trained to emit.
    """
    if fmt == "tagged":
        return expl
    return "\n".join(split_into_items(expl))


def _build_av_sft_cols(
    batch: pa.RecordBatch, actor_prompt_content: str, fmt: str
) -> dict[str, pa.Array]:
    n = len(batch)
    api_expl = batch.column("api_explanation").to_pylist()
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    # tagged (v1): response = "<explanation>\n{expl}\n</explanation>".
    # list/bullets (v2/v3): response = the normalized item text, no wrapper.
    # extract_explanation_open treats a no-tag response as the whole payload.
    responses = [
        wrap_explanation(e) if fmt == "tagged" else _format_items(e, fmt)
        for e in api_expl
    ]
    return {
        "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
        "response": pa.array(responses, type=pa.string()),
    }


def _build_rl_cols(batch: pa.RecordBatch, actor_prompt_content: str) -> dict[str, pa.Array]:
    n = len(batch)
    prompt_msg = [{"role": "user", "content": actor_prompt_content}]
    return {
        "prompt": pa.array([prompt_msg] * n, type=_PROMPT_STRUCT),
    }


def _maybe_truncate_explanation(
    expl: str, row_key: int, max_items: int, taper: float, seed: int
) -> str:
    """v2 item 2 (AR warm-start truncation): keep the first K newline items of the
    explanation, K drawn from the SAME tapering distribution RL uses (no curriculum
    — warm-start has no training-progress clock). Pre-calibrates the online critic
    on the short prefixes it will face at RL step 0, instead of meeting them OOD
    (the step-0 cliff that drove it to NaN). Gold activation is untouched, so this
    is exactly the RL setup: regress the full gold from a truncated prefix.
    max_items <= 0 → no truncation (return as-is)."""
    if max_items <= 0:
        return expl
    items = split_into_items(expl)
    k = sample_item_count(seed, row_key, max_items, taper, curriculum_groups=0)
    if k >= len(items):
        return expl
    return "\n".join(items[:k])


def _maybe_truncate_to_tokens(
    expl: str, row_key: int, min_tokens: int, max_tokens: int, seed: int, tokenizer: Any,
    keep_full_frac: float = 0.0,
) -> str:
    """v3 AR warm-start truncation: keep the first K TOKENS of the (already
    formatted) explanation, K ~ U[min_tokens, max_tokens] — the SAME uniform draw
    RL's tokens-mode truncation uses. Pre-calibrates the online critic on the
    1-token-and-up prefixes it will face at RL step 0 (with min_tokens=1 the
    shortest targets are near-impossible; the grad-finiteness guard is the
    backstop, this warm-start exposure is the first line of defense).
    Stripped like extract_explanation_open strips the RL rollout prefix, so the
    warm-start critic input matches the RL-time critic input byte-for-byte.

    keep_full_frac: fraction of rows (deterministic per-row draw) left UNTRUNCATED.
    With every row truncated to K <= max_tokens, text longer than max_tokens would
    be out-of-distribution for the critic at every stage (RL prefixes are also
    capped), making full-length round-trip evals read artificially low. A small
    untruncated slice keeps full-length text in-distribution — matching v2's item
    truncation, which left many rows whole. max_tokens <= 0 → no truncation."""
    if max_tokens <= 0:
        return expl
    if keep_full_frac > 0.0:
        if random.Random(f"nla-trunc-keepfull:{seed}:{row_key}").random() < keep_full_frac:
            return expl
    ids = tokenizer(expl, add_special_tokens=False)["input_ids"]
    k = sample_truncation_length(seed, row_key, min_tokens, max_tokens)
    if k >= len(ids):
        return expl
    out = tokenizer.decode(ids[:k]).strip()
    assert out, (
        f"token truncation to k={k} produced an empty critic input (row {row_key}). "
        f"Explanation starts with whitespace-only tokens — upstream cleaning should "
        f"have removed those."
    )
    return out


def _build_ar_sft_cols(
    batch: pa.RecordBatch, critic_template: str, suffix_ids: list[int], tokenizer: Any,
    row_offset: int = 0, trunc_max_items: int = 0, trunc_taper: float = 2.0, trunc_seed: int = 0,
    trunc_min_tokens: int = 1, trunc_max_tokens: int = 0, trunc_keep_full_frac: float = 0.0,
    fmt: str = "tagged",
) -> dict[str, pa.Array]:
    api_expl = batch.column("api_explanation").to_pylist()
    prompts: list[str] = []
    n_suf = len(suffix_ids)
    for j, expl in enumerate(api_expl):
        expl = _format_items(expl, fmt)  # match the AV's trained output format
        expl = _maybe_truncate_explanation(
            expl, row_offset + j, trunc_max_items, trunc_taper, trunc_seed
        )
        expl = _maybe_truncate_to_tokens(
            expl, row_offset + j, trunc_min_tokens, trunc_max_tokens, trunc_seed, tokenizer,
            keep_full_frac=trunc_keep_full_frac,
        )
        prompt = critic_template.format(explanation=expl)
        # Verify the tokenized prompt ENDS with the expected suffix IDs.
        # Training extracts at tokens[-1], so this check guarantees that's the
        # right spot. If the explanation's final bytes merge with the suffix
        # start (BPE edge case), the tail won't match and we fail loud here,
        # not silently at training.
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        assert len(ids) >= n_suf and ids[-n_suf:] == suffix_ids, (
            f"critic prompt does not end with expected suffix IDs {suffix_ids}. "
            f"Got tail: {ids[-n_suf:] if len(ids) >= n_suf else ids}. "
            f"Prompt: {prompt[:200]!r}... "
            f"This means the explanation's final characters merged with the "
            f"template suffix at the BPE boundary. Either the explanation ends "
            f"with an unusual sequence, or the critic template needs a delimiter "
            f"before the suffix."
        )
        prompts.append(prompt)
    return {
        "prompt": pa.array(prompts, type=pa.string()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="stage2 output (for av_sft/ar_sft) or stage1 output (for rl)")
    p.add_argument("--stage", required=True, choices=["av_sft", "ar_sft", "rl"])
    p.add_argument("--output", required=True)
    p.add_argument("--actor-template", default=None,
                   help="actor prompt template (must contain {injection_char}). "
                        "Default depends on --explanation-format.")
    p.add_argument("--critic-template", default=_DEFAULT_CRITIC_TEMPLATE)
    p.add_argument("--explanation-format", choices=["tagged", "list", "bullets"], default="tagged",
                   help="tagged (v1): actor wraps its output in <explanation> tags and the AV-SFT "
                        "response is wrapped. list (v2): no wrapper — the actor emits a raw "
                        "newline-separated list; extract_explanation_open reads the whole output. "
                        "bullets (v3): like list, but the prompt says bullet points and each "
                        "item is prefixed '- '.")
    p.add_argument("--ar-truncate-max-items", type=int, default=0,
                   help="(ar_sft) >0: truncate each critic-input explanation to the first K newline "
                        "items, K ~ taper over [1, this]. Pre-calibrates the online critic on the "
                        "short prefixes it meets at RL step 0 (v2 item 2). 0 = no truncation.")
    p.add_argument("--ar-truncate-taper", type=float, default=2.0,
                   help="(ar_sft) taper power for the per-row item-count draw (>1 favors short prefixes).")
    p.add_argument("--ar-truncate-max-tokens", type=int, default=0,
                   help="(ar_sft) >0: truncate each critic-input explanation to its first K TOKENS, "
                        "K ~ U[--ar-truncate-min-tokens, this] — matches RL's tokens-mode uniform "
                        "truncation (v3). Mutually exclusive with --ar-truncate-max-items. "
                        "0 = no truncation.")
    p.add_argument("--ar-truncate-min-tokens", type=int, default=1,
                   help="(ar_sft) lower bound of the uniform token-count draw (with "
                        "--ar-truncate-max-tokens).")
    p.add_argument("--ar-truncate-keep-full-frac", type=float, default=0.0,
                   help="(ar_sft, with --ar-truncate-max-tokens) fraction of rows left "
                        "untruncated so full-length text stays in-distribution for the "
                        "critic (RL prefixes are capped at max_tokens, so without this "
                        "the critic never sees text longer than max_tokens anywhere).")
    p.add_argument("--ar-truncate-seed", type=int, default=0,
                   help="(ar_sft) seed for the per-row truncation draw.")
    p.add_argument("--keep-debug-metadata", action=argparse.BooleanOptionalAction, default=True,
                   help="carry detokenized_text_truncated through "
                        "(heavy; off for prod). Provenance (n_raw_tokens/doc_id/activation_layer) "
                        "is always carried.")
    add_storage_args(p)
    args = p.parse_args()

    storage = make_storage(args)

    # Resolve the actor template from the format unless explicitly overridden.
    if args.actor_template is None:
        args.actor_template = _ACTOR_TEMPLATES[args.explanation_format]
    assert not (args.ar_truncate_max_items > 0 and args.ar_truncate_max_tokens > 0), (
        "--ar-truncate-max-items (v2, item-based) and --ar-truncate-max-tokens "
        "(v3, token-based) are mutually exclusive — pick the one matching the "
        "RL truncation mode."
    )

    assert "{injection_char}" in args.actor_template, (
        f"--actor-template must contain '{{injection_char}}' placeholder. Got: {args.actor_template!r}"
    )
    if args.stage == "ar_sft":
        assert "{explanation}" in args.critic_template, (
            f"--critic-template must contain '{{explanation}}' placeholder. "
            f"Got: {args.critic_template!r}"
        )

    in_meta = read_sidecar(storage, args.input)
    assert in_meta.stage == "base", (
        f"expected stage=base input (from stage1 or stage2), got stage={in_meta.stage!r}. "
        f"Re-feeding stage3 output into stage3 would produce double-wrapped garbage."
    )
    assert in_meta.extraction.norm == "none", (
        f"expected raw vectors (norm='none') from upstream, got norm={in_meta.extraction.norm!r}. "
        f"Data-gen never normalizes — if this sidecar claims otherwise, something "
        f"upstream transformed the data or the sidecar is stale. Training-side code "
        f"is the only place that should normalize."
    )

    tokenizer = load_tokenizer(in_meta.extraction.base_model)

    # ar_sft needs critic suffix IDs; av_sft/rl don't.
    critic_template_for_meta = args.critic_template if args.stage == "ar_sft" else None
    token_meta = build_token_meta(
        tokenizer, args.actor_template, critic_template=critic_template_for_meta
    )
    inj_id = token_meta.injection_token_id
    suffix_ids = token_meta.critic_suffix_ids  # None for av_sft/rl

    # Actor prompt is constant across all rows (template is fixed). Build once.
    actor_prompt_content = args.actor_template.format(injection_char=_INJECT_PLACEHOLDER)

    in_pf = pq.ParquetFile(storage.open_read(args.input))
    in_col_names = in_pf.schema_arrow.names
    if args.stage in ("av_sft", "ar_sft"):
        assert "api_explanation" in in_col_names, (
            f"stage={args.stage} requires api_explanation column — run stage2 first. "
            f"Available columns: {in_col_names}"
        )
    assert in_pf.metadata.num_rows > 0, (
        f"input parquet is empty — nothing to build. Check upstream split fractions."
    )

    out_schema = _schema_for(args.stage, args.keep_debug_metadata, in_meta.extraction.d_model)
    carry_cols = _PROVENANCE_COLS + (_HEAVY_DEBUG_COLS if args.keep_debug_metadata else [])
    storage.ensure_parent(args.output)
    row_count = 0

    # activation_vector + carry_cols are never transformed, just copied — so
    # pass them through as arrow arrays (batch.column(name)) instead of
    # round-tripping through to_pylist → from_pylist. At 4096 rows × 3584
    # floats that was 14.7M Python objects per batch, ~3.6B across the run.
    # Only api_explanation (small string col) gets materialized.
    passthrough_cols = ["activation_vector", *carry_cols]
    with pq.ParquetWriter(storage.open_write(args.output), out_schema) as writer:
        for batch in tqdm(in_pf.iter_batches(batch_size=_CHUNK_SIZE), desc="chunks",
                          total=(in_pf.metadata.num_rows + _CHUNK_SIZE - 1) // _CHUNK_SIZE):
            match args.stage:
                case "av_sft":
                    built = _build_av_sft_cols(batch, actor_prompt_content, args.explanation_format)
                case "ar_sft":
                    assert suffix_ids is not None
                    built = _build_ar_sft_cols(
                        batch, args.critic_template, suffix_ids, tokenizer,
                        row_offset=row_count,
                        trunc_max_items=args.ar_truncate_max_items,
                        trunc_taper=args.ar_truncate_taper,
                        trunc_seed=args.ar_truncate_seed,
                        trunc_min_tokens=args.ar_truncate_min_tokens,
                        trunc_max_tokens=args.ar_truncate_max_tokens,
                        trunc_keep_full_frac=args.ar_truncate_keep_full_frac,
                        fmt=args.explanation_format,
                    )
                case "rl":
                    built = _build_rl_cols(batch, actor_prompt_content)
                case _:
                    raise AssertionError(f"unreachable: stage={args.stage!r}")

            for col in passthrough_cols:
                built[col] = batch.column(col)

            writer.write_table(pa.table(built, schema=out_schema))
            row_count += len(batch)

    # Derive dataset_id from the input's — preserves stage0's corpus/slice hash
    # so two runs from different corpora on the same model/layer don't collide.
    out_meta = replace(
        in_meta,
        dataset_id=f"{args.stage}_{in_meta.dataset_id.removeprefix('base_')}",
        stage=args.stage,
        row_count=row_count,
        keep_debug_metadata=args.keep_debug_metadata,
        tokens=token_meta,
        prompt_templates={"actor": args.actor_template, "critic": args.critic_template},
        parent_datasets=[in_meta.dataset_id],
        created_by="nla.datagen.stage3_build",
        created_at="",
        git_commit="",
    )
    write_sidecar(storage, args.output, out_meta)
    print(f"wrote {row_count} rows ({args.stage}) → {args.output}")
    print(f"injection: {token_meta.injection_char!r} id={inj_id} "
          f"neighbors=({token_meta.injection_left_neighbor_id}, {token_meta.injection_right_neighbor_id})")
    if args.stage == "ar_sft":
        print(f"critic_suffix_ids: {suffix_ids} (extraction at tokens[-1])")


if __name__ == "__main__":
    main()
