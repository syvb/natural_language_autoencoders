# v3 RL training signal per token position: reconstruction vs KL

How much does each of the two RL loss components contribute at each token
position of a v3-matryoshka rollout?

In the v3 run (`../run_rl_v3.sh` + miles at the pinned commit) the two signals
enter differently:

- **Reconstruction** is sequence-level: `reward = −mse_nrm(critic(prefix), gold)`
  on a rollout capped at `L ~ U[1,120]` content tokens (tokens-mode truncation,
  offset 0 — the trained sequence *is* the prefix; see `nla/truncation.py`
  "cap, don't post-truncate"). GRPO broadcasts the group-normalized advantage
  over the tokens, so the *per-position* reconstruction structure is the
  marginal reward **Δr(t) = r(t) − r(t−1)** of adding token t.
- **KL** is genuinely per-token: `kl_loss_coef · k1_t` with
  `k1_t = log π(x_t) − log π_ref(x_t)` at the sampled token (miles
  `--use-kl-loss`, default `k1` estimator; `configs/rl.sh`), ref = the v3 SFT
  warm-start, **β = 0.03**.

Both terms exist only for `t ≤ L`, so they share the exposure factor
`P(L ≥ t) = (121−t)/120` — the *raw* Δr(t) and β·k1(t) curves are directly
comparable in reward-units-per-token, and the exposure-weighted versions give
the expected contribution as trained.

## Method (`sweep_reward_kl.py`, on-box)

For N=200 held-out distinct-doc prompts (`av_eval_v3.parquet`, same selection
as `../fve_truncation_sweep`):

1. one full-length rollout per prompt from the RLed AV at **T=1**
   (top_p=1, top_k=0 — the rollout distribution; capping at L is
   distributionally identical to taking the first L tokens);
2. teacher-force the same sequence through the policy **and** the warm-start
   reference (both with activation injection; sidecars asserted equal) →
   per-position `k1` and exact full-vocab KL;
3. critic forward on every prefix `t = 0..len` mirroring `nla/reward.py`
   exactly (`extract_explanation_open`, critic template, `−mse_nrm`,
   failed-extraction −2.0) → `r(t)` per sample.

Checkpoints: policy+critic `syvb/nla-qwen2.5-7b-L20-v3-rl` `iter_0000200/{av,ar}`
(native value head, post-`c560ac4` export guard, finiteness asserted),
ref `syvb/nla-qwen2.5-7b-L20-av-matryoshka-sonnet46-v3`.

Caveats baked into the numbers:

- `r(0)` is the failed-extraction floor (−2.0), so **Δr(1) ≈ r(1)+2 is a
  floor artifact**, not reconstruction — marginal plots start at t=2.
- At position t the reward mean is over samples with `content_len ≥ t`; the
  KL mean also includes the EOS token of samples that end exactly there
  (EOS is a real trained token at that position). `n_alive` is in the CSV.
- The critic is the *final* co-trained critic scoring the *final* policy —
  a post-hoc decomposition of the converged signal, not a replay of training
  (both moved during the run).

## Run

```bash
# box: 1× H100/H200 80GB, stock pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel image
rsync repo → /workspace/nla ; put HF token at /root/.hf_token
bash setup_box.sh
cd /workspace/nla && PYTHONPATH=. python experiments/.../reward_kl_decomposition/sweep_reward_kl.py 200
# pull /workspace/out/rkl/* → results/ ; locally:
python plot_reward_kl.py
```

## Results

(filled in after the run — see `results/`)
