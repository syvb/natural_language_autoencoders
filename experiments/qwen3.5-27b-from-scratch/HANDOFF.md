# HANDOFF: from-scratch NLA training run on Qwen3.5-27B

You are picking this up with no prior context. This file tells you what the
job is, what is already proven, what is not, the rules you must not break,
and the order of operations. Command-level procedure lives in
[`RUNBOOK.md`](./RUNBOOK.md) in this directory — this file is the context and
the orders; the runbook is the how.

## The mission

Train a **Natural Language Autoencoder** pair from scratch on
`Qwen/Qwen3.5-27B`: an **AV** (actor) that verbalizes a layer-42
residual-stream activation injected into its context as a list of bullet
points, and an **AR** (critic, a 43-layer truncation of the same model +
linear value head) that reconstructs the activation from that text. Pipeline:
activation extraction → dataset build → AV/AR SFT warm-start (2 epochs,
gated) → matryoshka RL (GRPO, uniform token-truncation reward U[1,120],
**position-tapered KL**: coefficient 0.02 at the first response token,
halving every 40 tokens — this run's one recipe change vs the validated 7B
v3 run) → eval. Gold explanations are reused from
`ceselder/nla-matryoshka-warmstart-sonnet46` (text-only, tokenizer-portable);
zero API spend. Success = HF checkpoints per save + round-trip FVE numbers
(full-length and 10/20-token prefixes) showing the matryoshka property vs the
warm-start baseline.

## Where everything is

- **Repo/branch**: clone `github.com/syvb/natural_language_autoencoders`,
  branch **`qwen3.5-27b-from-scratch`**. You do NOT have write access to that
  fork (nor to upstream `kitft/...`) — **create your own fork and push your
  work there**, keeping this branch as the base.
- **This directory** (`experiments/qwen3.5-27b-from-scratch/`): every script
  you need. `config.env` = production profile (single source of truth;
  scripts load it with setdefault semantics — your exported env wins).
  `RUNBOOK.md` = step-by-step. `driver_realsmoke.sh` + `smoke_27b.env` = the
  pre-run smoke. `check_health.py` = log-parsing gate (exit 1 on
  FAIL). `10_bringup_check.py` = injection gate.
- **Credentials expected on the training box**: HuggingFace write token and
  WandB token
- **Upstream framework**: `miles` (cloned at
  `nla/miles_patches/UPSTREAM_PIN`, patched by `setup_box.sh`). **Never edit
  `miles/` in this repo** — extend via `nla.train_actor.NLAFSDPActor` and the
  `--*-path` function-pointer args; box-side hot-patches belong in
  `setup_box.sh`.

## Validation status — read this before trusting anything

| surface | status |
|---|---|
| Full pipeline mechanics (SFT from scratch, RL w/ taper, save@N, export) | ✅ validated end-to-end at Qwen2.5-7B scale on the pinned stack |
| Qwen3.5-27B bring-up (loads on tf 5.x, L42 extraction, injection alters generation, 0% CJK) | ✅ validated, 1 GPU |
| Stack pin `lmsysorg/sglang:v0.5.10.post1` (torch 2.9.1 / sglang 0.5.10 / tf 5.3.0) | ✅ validated; **do not take newer images** — sglang ≥0.5.11 ships torch 2.11 which breaks miles multi-rank FSDP |
| Code review | ✅ 4-agent review, all findings fixed (see git log 030e4e3/77a1740) |
| **The 27B SFT/RL path itself** | ❌ **NEVER EXECUTED. Your first action is the real-model smoke below.** |
| B200 (sm_100) kernels for this stack | ❌ untested; first contact happens in your smoke |

**Step 0, before the full run:** rent one **8×B200** (the
production GPU — do NOT silently substitute another GPU type; if B200 is
unavailable, stop and tell the human the options), image
`lmsysorg/sglang:v0.5.10.post1`, disk ≥1 TB. Ship this repo + tokens, then:

```bash
nohup bash experiments/qwen3.5-27b-from-scratch/driver_realsmoke.sh > driver.log 2>&1 &
```

It runs the ACTUAL 27B pipeline at reduced scale (6k rows, 20 SFT steps, 10
RL steps at the production 4/2/2 topology, save@8, real export path), is
artifact-gated (re-running resumes after a fix), and ends with
`SMOKE3_DONE`/`SMOKE3_FAIL` sentinels. Est. 2.5–3.5 h. Expect a one-time
flash-attn **sm100 source build** (~20 min; the setup script caps and forces
it correctly). The RL stage is where any Blackwell surprise will surface —
that is the point. Only proceed to the full run when it's green.

## Hard rules (violating these has burned this project before)

1. **wandb is mandatory for every training run**: `--use-wandb
   --wandb-project nla-qwen3.5-27b --wandb-team octahedral-systems`, key via
   env from the key file, never in argv. (The launchers do this; don't
   disable it.)
2. **AV generation is always T=1 sampling** (`do_sample=True,
   temperature=1.0, top_p=1.0, top_k=0`) — never greedy, and don't label the
   sampling in user-facing copy.
3. **Smoke = production config.** Same model, same image, same GPU type;
   reduce only rows/steps. If you can't match, say so — don't substitute.
4. **Destroy/stop GPU boxes when done or blocked.** Verify zero running
   instances before ending a session.
5. **Data invariants** (CLAUDE.md): datagen never normalizes (raw vectors,
   sidecar carries `injection_scale`/`mse_scale`); document-level splits;
   sidecar `nla_meta.yaml` is the contract — never hardcode token IDs or
   templates; `cp_size==1` only.
6. **Never name a checkpoint directory `av`** (shadows the PyAV package →
   bizarre transformers import errors). Use `av_ckpt`-style names.
7. **Never run `python -m nla...` with cwd = the repo's parent** (namespace
   shadowing; the launchers cd defensively — keep that).
8. Commit and push finished work to your own fork proactively.

## Constants already determined (do not re-derive blindly)

- Extraction layer **42** (2/3 of 64), `d_model` **5120**, critic **43
  layers** (~17B). `mse_scale` auto = √5120.
- **`INJECTION_SCALE=115`** — measured (mean L2 95.6 at L42, 300-row
  bring-up). The full extraction re-measures; if `norm_stats.json` disagrees
  materially, update `config.env` and say so.
- Injection marker auto-selects to **`㈜` (id 158983)** on this tokenizer;
  the cache + sidecar handle it. The CJK smell test still applies: a broken
  injection path makes generations free-associate Chinese.
- **`LOSS_MASK_TYPE=qwen3`** (already in `config.env`). The plain `qwen`
  mask type hard-crashes on Qwen3.5's chat template.
- **`enable_thinking=False` at every `apply_chat_template` call site**
  (already wired: RL rollouts, SFT tokenization, neighbor computation,
  bring-up). Qwen3.5's template otherwise opens a `<think>` block in every
  generation prompt. Don't add new template call sites without it.

## The full run (after the smoke is green — same box works)

Follow RUNBOOK.md §Execution exactly (it exports `WORK`, tees every log, and
names each gate). Summary of the shape and the decision rules:

1. **Full extraction** (~450k rows, hours on 1 GPU) → confirm
   `INJECTION_SCALE` → `10_bringup_check.py` gate.
2. **Dataset build + critic init.**
3. **SFT**: default **2 epochs per role** (~220k rows each; AV needs
   `INJECTION_SCALE`, AR trains on token-truncated inputs). Gates: AR
   `fve_nrm` climbing well above 0 by end of epoch 1; stop at 1 epoch if
   round-trip FVE ≥ ~95% of the critic-gold ceiling, extend to 3 if still
   climbing at 2. `check_health.py <log> --sft` must PASS.
4. **Convert + upload** warm-start checkpoints (`04_convert_upload.py`,
   repos under `HF_REPO_PREFIX` in `config.env` — adjust the namespace to
   one you can write to).
5. **RL**: 20-step smoke first (gate: `check_health.py --rl --taper` PASS,
   `kl_flat` metric present = taper dispatched), then the full run:
   **budget `NUM_ROLLOUT=300`, save@50, push every save**
   (`push_ckpt_to_hf.sh`). **Stop rule**: reward slope ≈ 0 over ~30 steps
   AND 10-token-prefix FVE flat across two consecutive saves — expect to
   stop at 150–250. If reward is still climbing at 300, extend by resume in
   +100-step increments. `run_rl.sh` auto-resumes from the save root; verify
   the "RESUMING from ... @ iter N" line — a "FRESH RL start" after a crash
   means the tracker is missing: stop and investigate.
6. **Eval**: round-trip FVE on the held-out `av_eval` split, full-length AND
   10/20-token prefixes, vs the warm-start baseline; ~100 AV samples at T=1,
   read them, report CJK rate. The matryoshka signature = short-prefix FVE
   far above the warm-start's.

Expected metric shapes and failure-mode playbook (grad-guard skips, NCCL
desync symptoms, value-head corruption tripwires, thinking-leak): RUNBOOK.md
§Expected-metric-shapes and §Failure-modes. Two cheat notes: `loss_nonfinite`
must be 0 every step for both roles (the guard skips, never poisons —
occasional skips OK, >40% = stop and investigate), and `raw_reward` pinned
at −2.0 means the reward path is dead, not "bad policy".
