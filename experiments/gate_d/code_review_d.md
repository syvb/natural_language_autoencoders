# Gate D pipeline — adversarial code review (pre-D1-batch / pre-D2)

Reviewed: `labeler_d.py`, `build_bundles_d.py`, `rebuild_evidence.py`,
`extract_d.py`, `attn_rows.py`, `gen_d.py`, `ground_score.py`, `ridge_d.py`,
and the gate_c reuse surface (`train_critic.py`, `predict_critic.py`,
`evaluate.py`). Empirical checks were run locally where they settle a
question; results are inlined.

Verified facts used below:

- `meta_d.json`: 15,000 positions (11,682 train / 3,318 holdout, 22.1%).
- `bundles_d.json`: 15,000 bundles, 1,500 quiet; 0 non-quiet bundles with
  empty `heads`; 0 degenerate armT labels.
- `gend_out/attn_read.jsonl`: 30,000 rows, **0 duplicate (i,s) keys**,
  148 rows with `explanation: null`. `cpre.jsonl`: 15,000 rows, 0 dups,
  5 nulls. The `cat`-merge did not introduce duplicates.
- Token lengths through the AR critic template (`'Summary of the following
  text: <text>{explanation}</text> <summary>'`, ~12 tokens overhead):
  armT max 99, armZ reads max 163, armC reads max 142, pilot-v4 armH labels
  max 153. All comfortably under `--max-len 320`.
- Pilot v4 used `--sample-every 150 --limit 100` (ids 0,150,…,14850).
- `ridge_d_result.json`: holdout R² = 0.401 → CAUTION branch (residualized
  target + the +0.05 bar), correctly saved `resid_attn_holdout.npy` +
  `resid_attn_rows.npy`.

---

## BLOCKER

### B1. The planned regen-on-violation pass does not exist, and the pieces you would improvise it from will destroy data

- `labeler_d.py` (whole file): there is **no mechanical checker** (banned
  words, future-leak, bad opener) anywhere in the codebase, **no way to
  submit an arbitrary subset of ids** (`labeler_d.py:100` — the only
  selectors are `--sample-every` stride and `--limit` prefix, which cannot
  express "these 900 failing positions"), and **fetch clobbers its output**
  (`labeler_d.py:142` — `Path(args.out).write_text(json.dumps(results))`
  writes only the current batch's results; no merge with an existing file).
- Failure scenario: main 15k batch fetched to `labels_h.json`; regen batch
  for ~1k failing ids fetched with the same `--out` → the file now contains
  only the ~1k regen labels. (Recoverable from the API for 29 days, but the
  workflow as coded produces a wrong primary artifact.) Alternatively, the
  operator filters by hand-editing `bundles_d.json`, at which point the
  `custom_id = f"pos-{i}"` keys still map correctly — but nothing merges
  the two result files.
- Fix (small): add `--ids-file` (JSON list of bundle keys) to the id
  selection at `labeler_d.py:100`; make fetch **merge**:
  `results = json.loads(out.read_text()) if out.exists() else {}` before
  the loop (overwriting per-key is then the desired regen semantics); and
  write the checker (banned-word regex from the PROMPT's own list +
  opener check `^The (write|attention)` + future-leak check = label content
  words ∩ words(doc tokens after pos, minus words at/before pos) ≠ ∅,
  using `doc_tokens.jsonl`) that emits the ids-file. Do this **before**
  submitting the 15k batch so the loop is one pass, not improvised.

---

## MAJOR

### M1. The D0.3 GO verdict was computed against the future-leaking evidence; re-scored on fixed evidence the decisive margin shrinks ~40%

- `ground_score.py:48` reads `attn_evidence.jsonl` — the *original* file
  whose source contexts `ctx = decode(ids[j-3:j+4])` (`extract_d.py:111`)
  spill up to 3 tokens **past the labeled position**. The src-specific
  clause (`ground_score.py:72-73`) subtracts only `tail_w` (which ends at
  pos), so future tokens survive into `src_set`. AV reads of `attn_out`
  plausibly verbalize the continuation (the activation encodes next-token
  information), and only the *real* pairing shares the actual continuation
  — a leak channel that inflates exactly the clause the GO gate hinges on.
- Empirical re-score (same word machinery, 5 perms): src margin
  **+0.0025 on attn_evidence.jsonl → +0.0015 on attn_evidence2.jsonl**.
  Still positive, so GO probably survives, but the gate must be re-run
  properly (`attn_evidence2` has no `lens_top` field — `ground_score.py:54`
  will KeyError; make lens optional) before arm Z is treated as alive.
- Also worth saying out loud: with n≈15k the CI is so tight that a src
  margin of +0.0015 (reads share ~0.2% of their content words with
  attended sources beyond chance) is "significant" but minuscule. The
  pre-registered gate has no effect-size floor — that's a pre-registration
  design point, not a code bug, but the writeup should report the absolute
  number, and the soft-fail branch (drop arm Z) deserves consideration.
- Fix: re-run `ground_score.py` pointed at `attn_evidence2.jsonl` (lens
  union becomes src∪tail only), regenerate `ground_attn_read.json` /
  `ground_cpre.json`, and record both versions.

### M2. The arm-building step (matched n=8000, 5% quiet stratum, drop failures) is not implemented; `armT_train.json` as-is is not the pre-registered arm

- `build_bundles_d.py:73-76` writes **all** 11,682 train rows (≈10% quiet)
  into `armT_train.json`. PLAN.md D1 pre-registers: quiet positions below
  the 10th percentile are dropped except a 5% stratum, and D2 trains on
  **matched n = 8,000 per arm**. There is no `build_arms_d.py` (PLAN.md's
  infra map names one). If `train_critic.py` is pointed at the current
  files: arms have unmatched n (armH loses failed/regen-dropped labels,
  armZ loses 148 nulls ×2 samples, armT loses nothing), the quiet fraction
  is 10% not 5%, and the direction-only loss (`train_critic.py:148-149`
  normalizes targets to unit length) puts full weight on quiet positions
  whose write directions are essentially noise.
- Fix: write `build_arms_d.py` that (a) drops `explanation is None` /
  `label is None` rows, never falls back to `raw`; (b) applies the quiet
  rule (keep a 5% quiet stratum, same sampled stratum across arms); (c)
  subsamples each arm to a **shared** 8,000-position training set (same
  tidx across arms where possible — a per-arm random 8k confounds the
  contrast with position-set differences); (d) emits per-arm eval files
  covering all 3,318 holdout positions (1 text per position — see M3).

### M3. armZ/armC pair-file pitfalls: 2 samples/pos in eval breaks `evaluate.py` silently; null explanations must be dropped, not raw-substituted

- `gen_d.py:142` produces 2 samples per position for `attn_read`. If the
  armZ **eval** pair file carries both, `predict_critic.py` happily emits
  6,636 rows with duplicated `tidx`, and `evaluate.py:131-135` builds
  `ia = {tidx: k}` dicts that keep only the **last** occurrence — one
  sample per position is silently discarded with an arbitrary choice, and
  `n_shared` understates what was computed. Training pairs may keep both
  samples (two texts → same target is fine); eval pairs must be exactly
  one per tidx (fix to s=0).
- 148 attn_read rows / 5 cpre rows have `explanation: null`
  (`gen_d.py:113-115` stores `raw` separately). `ground_score.py:61` falls
  back to `raw` for scoring — acceptable there, but pair files must not:
  `raw[:400]` contains unparsed generation text. Drop those positions from
  the affected arm (and from the matched train set, per M2c).

### M4. `evaluate.py` (gate_c) — exact change list for gate D

Run as-is it crashes loudly on missing files, so nothing silent — but two
items would bite after a naive rename. Required changes:

1. `evaluate.py:53-57`: `meta2.json → meta_d.json`, `acts18.npy → v_pre.npy`,
   `diff_targets.npy → attn_out.npy`.
2. `evaluate.py:61-92`: **delete the ridge recompute and load
   `resid_attn_holdout.npy` + `resid_attn_rows.npy`** saved by `ridge_d.py`
   (identical machinery, SEED=0, λ=1e4, holdout R²=0.401 already
   pre-registered as the CAUTION-branch nuisance). Recomputing invites
   drift if the λ grid or seed is ever touched; the pre-registered residual
   already exists on disk. Keep `Dh = attn_out[ho]` for the descriptive
   `cos_d` and the partial-alignment robustness check.
3. `evaluate.py:27` `ARMS` and `evaluate.py:124,126` baseline/contrast
   names: baseline `arm0p_8k → armC_8k`; contrasts
   `["armA_8k","armB_8k"] → ["armT_8k","armH_8k","armZ_8k"]` (Holm at
   `evaluate.py:148-153` is already generic over `len(order)`, so the 3-way
   correction comes for free — but the docstring/pre-registration says
   "up to 3 contrasts"; make sure dead arms are simply absent, not zero
   rows).
4. `evaluate.py:170-181` (partial alignment beyond baseline): `Dh` is now
   the attention write; fine, just rename mentally — no code change beyond
   item 1.
5. Guard against M3: assert `len(np.unique(tidx)) == len(tidx)` per arm
   right after `evaluate.py:103`.
6. Verdict thresholds (`evaluate.py:155-167`) match PLAN.md D2 unchanged
   (CAUTION keeps the +0.05 bar) — no change needed.

---

## MINOR

### m1. `labeler_d.py:139` — fetch crashes on an empty/odd content block

`res.result.message.content[0].text` raises IndexError if a succeeded
result has an empty content array (e.g. a refusal-shaped reply) — one weird
row out of 15k aborts the whole fetch loop after the API call but before
`write_text`, so nothing is saved (re-fetchable, but annoying). Wrap:
`text = next((b.text for b in res.result.message.content if b.type == "text"), "")`.

### m2. `labeler_d.py:127-128` — batch id only printed, no manifest

If the terminal output is lost, the id is recoverable via
`client.messages.batches.list()`, but nothing records *which ids* were
submitted. Append `{batch_id, n, ids_hash, args}` to a `batches_d.jsonl`.
Then fetch can verify coverage (every expected `pos-{i}` came back) instead
of silently producing a partial label file.

### m3. `labeler_d.py:100` — `--sample-every` footgun on the full submit

Answering the specific question: the full submit cannot *duplicate* pilot
positions within a batch (dict keys are unique), and it *re-labels* the 100
pilot positions (≈$0.10 — fine; do not merge pilot files into the full
results, the prompt evolved v1→v4). The real hazard is shell-history reuse
of `--sample-every 150 --limit 100` on the production submit, which would
silently submit 100 requests instead of 15,000 — and with m2 unfixed,
nothing downstream notices until pair-building. Make submit print
`len(reqs)` vs `len(inputs)` and require `--yes` when they differ.

### m4. `train_critic.py:67-68` — right-truncation would silently cut the `<summary>` suffix anchor

`truncation=True, max_length=args.max_len` truncates the **end** of the
prompt; the critic extraction is suffix-anchored at `tokens[-1]`
(repo invariant), so an over-long label would silently anchor on label
text instead of `<summary>`. Empirically safe today (max 175 tokens incl.
template vs 320), but add
`assert ids[-1] == <last template token>` (or check the rendered prompt
endswith the suffix before truncation) — one line buys away a silent
catastrophic failure mode if a future arm has longer labels.

### m5. `train_critic.py:131-134` — `final_eval_cos` is the first 512 eval rows only

The in-training eval caps at `min(len(eval_set), 512)` over the **prefix**
of the eval file, which is meta-ordered → early docs only (doc-clustered
bias). Fine as a training curve; do not quote `RESULT.json:final_eval_cos`
as an arm's result — the verdict comes from `predict_critic.py` (all rows)
+ `evaluate.py`.

### m6. `ground_score.py:83-110` — null treated as a fixed scalar in the bootstrap CI

The doc-clustered bootstrap resamples only the real scores; the permutation
null (20 perms) enters as a constant, so the CI ignores null-estimation
variance. With 20 perms × ~15k positions the null SE is tiny, so this is
materially fine — note it in the writeup. Also: permutations can assign
evidence from the *same document* (shared vocabulary inflates the null →
conservative for the gate), and `if q == i: continue` makes per-perm n
fluctuate by a few. None of these flip a verdict at this n.

### m7. `build_bundles_d.py:27-28` — silent fallback to the leaky evidence file

If `attn_evidence2.jsonl` is absent (fresh checkout, partial sync), the
script silently rebuilds bundles from the future-leaking
`attn_evidence.jsonl` and prints only the filename. Given the entire D1
redesign exists to remove that leak, make the fallback a hard error (or at
minimum `sys.exit` unless `--allow-v1`).

### m8. `labeler_d.py:104-115` — direct mode writes results only at the end

A crash at request 95/100 loses all paid work (it *reads* an existing file
for resume, but never checkpoints). Write every ~10 labels. Low stakes
(direct mode is pilot-only) but free to fix.

### m9. `build_bundles_d.py:33-34` — quiet threshold computed over train+holdout pooled

The 10th-percentile threshold uses all 15k norms, so holdout norms
influence which *training* labels get the quiet register. Statistically
negligible (a rank threshold), but trivially avoidable: compute the
percentile on train rows only.

---

## NIT

### n1. `rebuild_evidence.py:53-61` — `TOPK_SRC + 2` overscan can under-fill sources

If >2 of the top-7 weights are `j==0` or dust (<0.02), fewer than 5 sources
survive. The BOS sink usually costs exactly one slot, so in practice you
get 5; scanning `np.argsort(-w)` lazily until 5 survivors would be exact.
On the brief's off-by-one questions: the window **cannot** include tokens
after the position (`min(j+4, pos+1)` — slice end exclusive at pos+1), and
it **does** include the position token itself when `j ≥ pos-3`, which is
correct ("at or before the position", and the prompt's carry-over clause
expects it).

### n2. `labeler_d.py:141` — error rows live in the same dict as labels

`{"label": None, "raw": "ERROR:errored"}` vs `"ERROR:expired"` — expired
results should be resubmitted (batch hit 24h), errored ones inspected.
Preserve the distinction; the regen checker (B1) should treat both as
regen candidates.

### n3. `extract_d.py:80-81` — sequential RNG instead of per-doc keyed RNG

`rng.choice` then `rng.random()` couples sampled positions and the split to
stream order. Single-process and deterministic here, so no harm — but it
violates the repo's per-doc-keyed-RNG invariant in spirit; worth a comment
so nobody shards this script later and silently changes the corpus.

### n4. `build_bundles_d.py:58-69` — degenerate armT path unguarded

If all of a non-quiet bundle's sources were filtered, the label would end
"retrieves prior context: ." — currently 0 occurrences (verified), but the
`parts`-empty case costs one `if` to route to the quiet template.

### n5. `gen_d.py:70-75` — resume glob `f"{job}*.jsonl"` is prefix-greedy

`attn_read*` would also match e.g. a stale `attn_read_old.jsonl` copied
into `gend_out/`, polluting the done-set. Harmless today (verified: only
`attn_read.jsonl`/`cpre.jsonl` exist, 0 duplicate keys), but keep backups
outside `gend_out/`.

---

## Explicitly checked, no finding

- **extract_d.py layer indexing / GQA** (`extract_d.py:84-97`):
  `hidden_states[20]` is the input to decoder layer 20 (hidden_states[0] =
  embeddings) → correct `v_pre`; `attentions[20]` is that layer's pattern;
  `o_proj` pre/post hooks capture the 28-query-head concat and the write;
  the per-query-head decomposition through `W_O` column slices is the right
  GQA treatment (patterns and writes both per *query* head), and the
  `cos > 0.999` reconstruction assert ran on every doc. Qwen2 `o_proj` has
  no bias, so the head-sum identity holds. No extraction-level poisoning
  found beyond the (already fixed) ctx future-leak.
- **attn_rows.py**: deterministic re-derivation from `doc_tokens.jsonl`,
  zero-padded past doc length, head ordering consistent with
  `head_norms.npy`; `rebuild_evidence.py`'s `total_norm = ‖attn_out‖`
  matches extract_d semantics.
- **Batch sizing**: 15,000 requests ≈ 50 MB body — well under the 100k
  requests / 256 MB batch limits; `custom_id = "pos-{i}"` round-trips
  losslessly through `removeprefix` to the string bundle keys.
- **MSE_SCALE / normalization for attn writes**: direction-only loss is
  consistent with the cosine-based pre-registered metric and with the
  causal judge's rescale-to-true-norm; norm information (quiet vs strong)
  is intentionally out of scope of the critic. No diff-specific semantics
  hardcoded in `train_critic.py`/`predict_critic.py` (`--targets` is
  generic; `predict_critic` needs `nla_meta.yaml` copied into the ckpt,
  already documented in its docstring).
- **Token budget vs `--max-len 320`**: all four arms measured; max 163
  label tokens + ~12 template tokens. Headroom ~2×.
- **ridge_d.py**: faithful copy of the evaluate.py machinery; CAUTION
  branch handled per plan; residuals saved for D2.
