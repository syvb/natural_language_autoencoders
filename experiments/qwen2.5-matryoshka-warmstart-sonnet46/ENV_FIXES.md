# Environment fixes to run NLA SFT on the stock PyTorch image

Target box: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` (conda at `/opt/conda`, torch
2.5.1+cu124, python 3.11), Hopper (H100/H200). Miles pinned at `radixark/miles@051cd15`.
SFT (`--debug-train-only`) is pure FSDP training on pre-generated parquets — it never
starts an SGLang engine — so we stub SGLang instead of installing the full inference stack.
These are the fixes that were needed; `setup_box.sh` applies them programmatically.

## 1. Deps (into the base conda env, no build_conda.sh)
`transformers==4.57.1` (pin: 5.x changes `apply_chat_template` return type; Qwen2.5 needs
≤4.57.x), `flash-attn` (prebuilt wheel), `ray[default] ring_flash_attn sglang-router
omegaconf tensorboard blobfile pybase64 pylatexenc accelerate wandb`, then
`pip install -e miles` and `pip install -e nla`. Apply the NLA miles patches:
`cd miles && git apply $NLA_REPO/nla/miles_patches/*.patch`.

## 2. SGLang SFT stub completion (`nla/_sglang_sft_stubs.py`)
The shipped stub was incomplete for this miles pin. Four edits:

- **`_add_sglang_arguments` must RETURN the parser** (miles does `parser =
  add_sglang_arguments(parser)`) and must register the `--sglang-*` args miles reads in
  shared (non-engine) code paths. Enumerate them with
  `grep -rhoE "args\.sglang[a-z_]+" miles/` — for this pin: `sglang_router_ip/port`,
  `sglang_router_request_timeout_secs`, `sglang_{tp,dp,pp,ep}_size`,
  `sglang_{data,pipeline,expert}_parallel_size`, `sglang_enable_{dp_attention,metrics}`,
  `sglang_server_concurrency`, `sglang_dest_name`, `sglang_moe_a`,
  `sglang_moe_runner_backend`, `sglang_speculative_algorithm`. Defaults: sizes=1,
  ints/None as appropriate, bools store_true.
- **`RouterArgs.add_cli_args` must RETURN the parser** too (harmless; miles ignores it).
- **Do NOT stub `sglang_router`** — the real `sglang-router` pip package IS installed (a
  miles dep) and registers the router args properly. Remove the `sglang_router` stub block
  from the shipped file.
- **Add the FSDP-backend weight-sync stubs** (imported at module load by
  `miles.backends.fsdp_utils.update_weight_utils` / `actor`, unused in SFT). Register the
  full dotted module names in `sys.modules`:
  `sglang.srt.patch_torch` + `sglang.srt.utils.patch_torch` (`.monkey_patch_torch_reductions`),
  `sglang.srt.utils` (`.MultiprocessingSerializer`),
  `sglang.srt.weight_sync.tensor_bucket` + `sglang.srt.model_executor.model_runner`
  (`.FlattenedTensorBucket`), `sglang.srt.batch_invariant_ops`
  (`.enable_batch_invariant_mode`).
- **The stub must NOT import torch.** It loads in *every* interpreter via the `.pth` below,
  including Ray bootstrap workers — eagerly importing torch there aborts them
  (`Fatal Python error: Aborted` during worker connect). (This bit us once; keep the FSDP2
  compat as a miles edit, see §3.)

## 3. torch 2.5.1 FSDP2 import (miles edit)
miles imports the FSDP2 API from the public namespace (promoted in torch 2.6):
`from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard`.
On 2.5.1 these live under `torch.distributed._composable.fsdp`. Patch miles:
```
sed -i 's|from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard|from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard|g' \
  miles/backends/fsdp_utils/actor.py
```
(`FullyShardedDataParallel` already exists in the public namespace on 2.5.1 — leave it.)

## 4. Load the stub in every process (Ray workers included)
Ray worker subprocesses import miles directly and never run `train_sft.py`, so the stub
must auto-load at interpreter startup. A one-line `.pth` (named to sort AFTER the editable
`__editable__.nla*.pth` so `nla` is importable):
```
SP=$(python -c 'import site;print(site.getsitepackages()[0])')
echo 'import nla._sglang_sft_stubs' > "$SP/zzz_nla_sglang_stub.pth"
```

## 5. Skip the router at runtime
`RolloutManager.__init__` calls `_start_router()` unconditionally, but it early-returns if
`--sglang-router-ip` is set. Pass `--sglang-router-ip 127.0.0.1 --sglang-router-port 39999`
on the SFT command line (already in `run_av_sft.sh` / `run_ar_sft.sh`) so no router
process is launched. `init_http_client` no-ops when `rollout_num_gpus == 0` (SFT).

## 6. Eval-only gotcha: `av` dir name vs PyAV
transformers (vision utils) pulls in `torchvision`, which does `import av` (PyAV). If the
eval runs with a checkpoint directory named `av/` on the cwd/PYTHONPATH, `import av`
resolves to that dir → `AttributeError: module 'av' has no attribute 'logging'` →
surfaces confusingly as `Could not find Qwen2ForCausalLM`. Fixes: run from a clean cwd,
don't name a dir `av`, or `pip uninstall -y torchvision` (not needed for text-only eval).
The eval scripts here load AV with `.to("cuda")` (not `device_map=`).
