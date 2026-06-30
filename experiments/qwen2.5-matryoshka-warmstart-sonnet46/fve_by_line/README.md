# FVE-by-line: how much of the activation is recovered from each prefix of the AV's list

For a few random held-out activations, this shows **each line of the v2 NLA's verbalization** next
to the **round-trip FVE you get if you truncate the explanation to that line** — i.e. how the
reconstruction quality builds up line by line.

## Method

For a held-out activation `v` (`av_eval.parquet`, raw):
- **AV** (`syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v2-rl` iter_200) verbalizes `v` → a
  newline list of snippets (greedy, 384-tok cap).
- For each `k = 1..N`, the **AR critic** (the co-trained v2 critic,
  `nla-qwen2.5-7b-L20-v2-rl-checkpoints` iter_200 `critic/hf`, with the fixed finite `value_head`)
  reconstructs `v̂_k` from the **first k lines only**.
- **Per-example FVE** (both vectors L2-normalized to `mse_scale=√d=59.87`, the training convention):

      FVE_k = 1 − ‖v̂_k − v‖² / ‖v − μ‖²

  where `μ` is the population mean of normalized held-out golds (over all 5,007 rows). FVE=1 perfect,
  FVE=0 no better than predicting the mean, FVE<0 worse than the mean.

`fve_by_line.py` produces `results/fve_by_line.json`; `plot_fve_by_line.py` renders one diagram per
example (lines + cumulative-FVE bars) plus `fve_by_line_overview.png`.

## What the diagrams show

**Information is front-loaded.** FVE jumps most on the first line and saturates within the first
~5–10 lines; the long tail of the list barely moves it, and sometimes *lowers* it (later lines add
drift). This is the behavior the truncation-RL objective was designed to induce — put the
reconstruction-relevant content first.

| example (doc) | N lines | FVE @1 line | FVE max (@line) | FVE full |
|---|--:|--:|--:|--:|
| Ultra-FineWeb:en:71869 | 44 | 0.73 | 0.80 (@19) | 0.75 |
| Ultra-FineWeb:en:31969 | 30 | 0.60 | 0.78 (@12) | 0.72 |
| Ultra-FineWeb:en:91336 | 25 | 0.36 | 0.60 (@11) | 0.58 |
| Ultra-FineWeb:en:62481 | 48 | 0.55 | 0.75 (@16) | 0.58 |
| Ultra-FineWeb:en:25283 | 53 | 0.65 | 0.83 (@10) | 0.70 |
| Ultra-FineWeb:en:33925 | 46 | 0.64 | 0.71 (@4)  | **0.51** |

The last example is the clearest illustration of over-extension: FVE peaks at line 4 (0.71) and
*decays* to 0.51 by the full list — the model said the useful part early and the rest hurt.

## Reproduce

```bash
# GPU box (≥40 GB holds AV+AR), repo on PYTHONPATH (driver_fve_by_line.sh does setup):
AV_DIR=/workspace/av_ckpt AR_DIR=/workspace/ar_ckpt EVAL=/workspace/av_eval.parquet \
  python3 fve_by_line.py 6 0      # -> results/fve_by_line.json
python3 plot_fve_by_line.py       # -> results/fve_by_line_ex*.png + overview
```
