# AV/AR round-trip FVE vs explanation-truncation length

How well does the AR (critic/reconstructor) recover the gold layer-20 activation
from only the **first L tokens (or lines)** of the AV's explanation? Sweeping L
shows how the truncation-RL'd model **front-loads information**, and a published
`kitft` AV/AR pair serves as a control.

FVE here = `1 − dirMSE / var`, both pred & gold L2-normalized to `mse_scale`,
raw-mean baseline (identical to `../eval_round_trip_fve.py`'s `stats()` and the
`fve_nrm` training metric). The value head is fixed across all truncation
lengths, so the **shape** of each curve is driven by the backbone reading more/
less text; the head only sets the overall level.

## Results (N=100 held-out distinct docs, greedy, 2026-06-27)

See `results/`. Headline (token sweep — the apples-to-apples unit):

| | tokens to FVE≥0 | to FVE≥0.5 | to FVE≥0.6 | full-length FVE |
|---|---|---|---|---|
| **RLed truncation** (kl0.01/iter_200) | 4 | 17 | 28 | 0.705 |
| **published kitft** (control) | 52 | 84 | 87 | 0.727 |

The truncation-RL model front-loads ~**5× faster** (FVE≥0.5 at 17 vs 84 tokens)
with **no full-length quality loss** (both plateau ~0.70–0.73). The control's
first ~50 tokens are *worse than predicting the mean* (FVE<0) — its key content
lands late. Log-log panels show error decaying ~`L^-0.44` until it hits the
irreducible full-explanation floor (~0.295 = 1−0.705).

> ⚠️ **Line panels are per-model only, not cross-model comparable.** The control
> writes a few long lines; the RLed model writes many short bullets, so one
> "line" = different token counts. Blank lines are coalesced (`\n+` → one break)
> so they don't inflate the line count. Use the **token** panels to compare models.

## ⚠️ Caveat: the RLed AR value head is corrupt — we swap in the SFT head

All four pushed truncation-RL AR critics (`syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard`,
`kl0`/`kl0.01` × `iter_100`/`iter_200`) have a **corrupt `ar/value_head.safetensors`**:
~12% of weights destroyed (NaN + finite values up to 3e38), exactly half the
output rows affected — an FSDP critic-shard save bug. It never touched training
(the live in-memory head was clean → AV trained, `fve_nrm` hit 0.68); only the
export is bad, and the DCP critic checkpoints were deleted, so it's unrecoverable.

**Workaround (what `setup_box.sh` does):** pair the RLed AR **backbone** (clean)
with the **clean SFT-warmstart value head** from
`syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46`. Validated: full-length FVE =
0.705 ≈ training `fve_nrm` 0.68, and the good half of the corrupt RLed head has
median |w| 0.0014 = the SFT head's — the head barely changes over 200 RL steps.
The control (`kitft`) uses its own clean native head, so it's a true baseline and
double-checks the workaround introduces no artifacts.

## Reproduce from scratch

A 48 GB GPU is plenty (7B bf16 inference ≈15 GB; AV and AR loaded one at a time).
Whole thing is ~15 min of compute + downloads, ~$0.10–0.15.

```bash
# 1. provision an RTX A6000 / A40 (48 GB), North America, reliable, >24h, <$1/hr
OFF=$(vastai search offers 'gpu_name=RTX_A6000 num_gpus=1 reliability>0.98 rentable=true' \
  -o dph_total --raw | python3 -c "import sys,json;print(json.load(sys.stdin)[0]['id'])")
vastai create instance $OFF --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 200 --ssh --direct --label nla-av-eval --onstart-cmd 'touch ~/.no_auto_tmux; sleep infinity'
# wait for actual_status=running (if it boots 'stopped': vastai start instance <id>); get IP/PORT

# 2. push token + repo to the box
ssh -p $PORT root@$IP "umask 077; printf '%s' \"$(cat ~/.hf_token)\" > /root/.hf_token"
rsync -az -e "ssh -p $PORT" --exclude '.git' --exclude '.venv*' --exclude '*.parquet' \
  /home/debian/nla-doll/ root@$IP:/workspace/nla/

# 3. set up (deps + download all models/data + SFT-head swap) and run both sweeps
B=/workspace/nla/experiments/qwen2.5-matryoshka-warmstart-sonnet46/fve_truncation_sweep
ssh -p $PORT root@$IP "bash $B/setup_box.sh"
ssh -p $PORT root@$IP "bash $B/run_sweeps.sh 100"          # ~12 min; nohup it for safety

# 4. pull CSVs into results/ and plot locally (CPU, matplotlib)
L=/home/debian/nla-doll/experiments/qwen2.5-matryoshka-warmstart-sonnet46/fve_truncation_sweep
scp -P $PORT root@$IP:/workspace/sweep/token_fve.csv       $L/results/token_fve.csv
scp -P $PORT root@$IP:/workspace/sweep/lines_fve.csv       $L/results/lines_fve.csv
scp -P $PORT root@$IP:/workspace/sweep_kitft/token_fve.csv $L/results/token_fve_kitft.csv
scp -P $PORT root@$IP:/workspace/sweep_kitft/lines_fve.csv $L/results/lines_fve_kitft.csv
python $L/plot.py

# 5. (optional) 100-sample markdown dump of AV generations
ssh -p $PORT root@$IP "cd /workspace && PYTHONPATH=/workspace/nla \
  python $B/gen_md_samples.py 100 /workspace/samples.md"
scp -P $PORT root@$IP:/workspace/samples.md $L/../av_kl0.01_iter200_100samples.md

# 6. TEAR DOWN (confirm the label is yours first!)
vastai show instances-v1
echo y | vastai destroy instance <ID>
```

`sweep_fve.py` is env-parameterized: `AV_DIR`, `AR_DIR`, `EVAL`, `OUTDIR`, `TAG`,
plus `argv[1]` = N samples. It auto-frees the AV after generation so the critic
gets full GPU headroom. See `../../../ENV.md` §3 (vast) and §11 (transformers-only
AV inference) for the gory details, and memory `ar-value-head-corrupt-export` for
the value-head bug.
