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

## Results (N=200 held-out docs, T=1, seed 0, 2026-07-07)

Figures: `results/fig_reward_kl_pertoken.png`, `results/fig_reward_kl_cumulative.png`.

| t | r(t) | Δr(t) | k1(t) | β·k1(t) |
|---|---|---|---|---|
| 1 | −0.939 | (+1.06 vs floor) | 2.87 | 0.086 |
| 2 | −0.717 | +0.222 | 7.37 | 0.221 |
| 5 | −0.508 | +0.062 | 5.25 | 0.158 |
| 10 | −0.398 | +0.018 | 2.42 | 0.073 |
| 20 | −0.297 | +0.006 | 2.85 | 0.086 |
| 50 | −0.241 | +0.001 | 2.16 | 0.065 |
| 120 | −0.225 | −0.001 | 0.85 | 0.026 |
| 159 | −0.251 | −0.004 | 0.39 | 0.012 |

**Reconstruction is extremely front-loaded; the KL penalty is broad and
dominates everywhere past the first ~2 tokens.**

- **Marginal reconstruction decays geometrically**: token 1 alone reaches
  r(1) = −0.94 (cos ≈ 0.53 from a single token), the next ~10 tokens add
  ~+0.5, and by t ≈ 20 each additional token is worth < 0.01 reward. Past
  t ≈ 110 marginals go slightly *negative* (the over-extension dip).
- **Per-token KL to the warm-start is large at every position** — mean 2–7
  nats over t ≤ 20 (median 2.6–2.9, so this is broad divergence, not a heavy
  tail), decaying to ~0.9 by t = 120. RL restructured the *beginning* of the
  explanation the most — exactly where the FVE front-loading gains live.
  Beyond the trained horizon (t > 120, exposure 0) KL keeps falling to ~0.4:
  positions that training never rewarded or penalized diverged least. The
  EOS token, when emitted, carries k1 ≈ 1.9.
- **Crossover at t ≈ 2**: β·k1(t) exceeds Δr(t) from the second token on.
  Summed over the trained range (t = 2..120, exposure-weighted):
  reconstruction gained **0.67** reward units vs **4.96** paid in KL penalty
  (raw, unweighted: 0.71 vs 7.93) — at convergence the KL term is ~7–11×
  the marginal reconstruction value the tokens buy. The reward the policy
  actually banks is overwhelmingly earned by tokens 1–10; everything after
  is nearly pure KL cost at the margin.
- Sanity: exact full-vocab KL ≈ k1 at every position (e.g. 2.97 vs 2.87 at
  t=1), so the k1 estimate is faithful. Mean full-length reward −0.246
  matches the run's converged raw_reward (−0.261). 57/200 sampled responses
  contain stray CJK chars (~11 each, embedded in fluent English) — the known
  v3 leak amplified by T=1 sampling vs 8/150 greedy; reward level confirms
  injection is healthy.

**Interpretation caveat:** this compares raw reward units. In the actual
gradient, the reconstruction side is GRPO-group-normalized (advantage =
(r−mean)/std within a group at fixed L) while the KL term is not, so the
7–11× ratio describes the *reward-vs-penalty budget* of the converged
policy, not the literal gradient ratio during training. It is also an
end-of-training snapshot: KL was accumulated over 200 steps, and both the
policy and the co-trained critic moved.
