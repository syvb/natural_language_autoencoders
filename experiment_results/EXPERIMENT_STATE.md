# NLA length-penalty experiment — live state

## Goal
Sweep length penalties during NLA RL, measure length(tokens)↔reconstruction(FVE) tradeoff +
generate 1000 held-out samples (explanation + FVE) per penalty. Code change: `nla/reward.py`
`NLA_LENGTH_PENALTY` env var (reward -= λ·response_length tokens).

## Box (4× H100, vast.ai)
- instance 41382679, ssh `ssh8.vast.ai:22678`, key `~/.ssh/id_ed25519`, env conda `miles` at /opt/conda.
- ~$9.92/hr. KEEP ALIVE until user says done (user iterating on penalties).
- Released NLA: kitft/nla-qwen2.5-7b-L20-{av,ar} at /workspace/models/{av,ar}; base Qwen2.5-7B at /workspace/models/base.
- RL data /workspace/rldata/rl.parquet (pile-10k docs 0-1200); held-out /workspace/heldout/heldout.parquet (docs 5000-5180, 1402 rows).

## Stack fixes (deployment patches to /workspace/miles — Ray/CUDA compat)
- ray pinned 2.48.0. torch 2.9.1+cu129, flash-attn 2.8.3 (prebuilt wheel), sglang @24c9100.
- rl_run.sh sets RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0 + RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 + CVD=0,1,2,3.
- miles/ray/rollout.py: engine num_gpus 0.2→1; CVD=range(base+span) in env_vars; dropped /usr/local/cuda/compat from LD_LIBRARY_PATH (was CUDA error 803). sglang_engine.py has a [NLA-DIAG] print.
- Scripts in vast_ops/: rl_run.sh, rl_kl_sweep.sh, eval_heldout.sh, extract_heldout.py, gen_*_data.sh, plot_progress.py.
- GPU layout: ACTOR_GPUS=2 CRITIC_GPUS=1 ROLLOUT_GPUS=1.

## Checkpoint gotchas
- ACTOR saves DCP-only (--load <save dir>, has latest_checkpointed_iteration.txt); CRITIC saves HF (+value_head.safetensors).
- eval_heldout: --hf-checkpoint released-av --load <actor save dir> --finetune --critic-load <critic hf>; needs --critic-save (assert) → throwaway, cleaned.
- DON'T name shell vars AR/AS/CC/etc — conda activate exports them. Use ACTOR_DCP/CRITIC_HF.
- Each actor DCP ~84GB → delete after eval (disk 300GB). Persist metrics+samples to local, not checkpoints.

## Results (SCOPE REDUCED by user: base + 0.006 + 0.002 only, then DESTROY box)
Held-out (1024 samples each, n_samples=2 over heldout.parquet, rollout-seed 9999):
- BASE NLA (released, no penalty): len 144, FVE 0.681, cos 0.885  [done, local]
- KL-on λ=0.006: len 34, FVE 0.478, cos 0.811  [done, local: experiment_results/heldout/pen0.006_kl.samples.jsonl]
- KL-on λ=0.002: TRAINING (run_kl002.out). After train: eval+extract+pull.
Deprecated KL-off (no held-out): λ=0 len 139 FVE 0.73; λ=0.01 len 30 FVE 0.54.
Local: experiment_results/heldout/{base_nla,pen0.006_kl}.samples.jsonl; pen{0,0.01,0.006}_seed1234/ metrics;
comparison_base_vs_penalty.md (base vs 0.006). Base eval = `ACTOR_DCP=BASE CRITIC_HF=/workspace/models/ar`.

## FINAL STEPS (after 0.002 train done)
1. 0.002 held-out eval: `ACTOR_DCP=/workspace/results/pen0.002_kl_seed1234/actor CRITIC_HF=.../critic/iter_0000035/hf TAG=pen0.002_kl_eval bash /workspace/eval_heldout.sh`; then `python /workspace/extract_heldout.py pen0.002_kl_eval`; pull samples.jsonl to experiment_results/heldout/pen0.002_kl.samples.jsonl.
2. Persist 0.002 per_step + train.log locally.
3. Regenerate `python vast_ops/generate_comparison_md.py pen0.002_kl pen0.006_kl`; rebuild tradeoff chart (base+0.002+0.006).
4. DESTROY box: `vastai destroy instance 41382679` (echo y |). ~$40 spent.
