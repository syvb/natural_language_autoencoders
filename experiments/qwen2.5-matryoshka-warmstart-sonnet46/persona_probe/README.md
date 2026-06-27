# Persona-probe: does a system-prompt persona surface in the AV's reading of the residual stream?

Uses the RLed AV (kl0.01/iter_0000200) as a readout: build a Qwen2.5-7B-Instruct
**chat** context whose **system prompt** asserts a persona of increasing strength
(0=neutral … 5=extreme), run the model, extract the **layer-20 activation** (exact
training convention: truncate to `layers[:21]`, hook `layers[20]`, last real token,
raw bf16), inject it into the AV, and read the bulleted explanation. Question: does
the trait appear in the explanation, and does it rise toward the top of the list as
the prompt strengthens? Traits: **sycophancy, liking the color yellow, sadness,
aggressively liking tofu**.

Run at three extraction distances from the persona text:
- `persona_probe.py`   — **far**: persona + neutral user + a full neutral *assistant continuation*, extract at the continuation's end (24 continuations × 4 traits × 6 strengths).
- `persona_sweep_near.py` — **near**: persona + a *short* neutral fragment, extract at the fragment's end (24 fragments/cell). The main quantitative sweep.
- `persona_diag.py`    — **immediate**: extract at the end of the persona text itself (`sysend`), and after one neutral user turn (`userend`). N=1/cell, yes/no diagnostic.

## Findings

**1. The persona signal is highly local — it washes out with distance.**
- **Far** (after a full neutral continuation): the trait appears **0%** of the time for *all four* traits at *every* strength. The explanation just describes the neutral continuation; the strength-5 explanation is near-identical to the strength-0 control.
- **Near** (short fragment): only **sadness** appears reliably (peaks ~42% at strength 4); sycophancy barely (≤4%); yellow/tofu never. See `results/persona_near.png`.
- **Immediate** (`sysend`): sycophancy appears at strength ≥2 (rank ~1–4, rising with strength); sadness pins to **rank 1** from strength 1; yellow/tofu still never appear.

**2. Stronger prompt → trait appears more and ranks higher — for the traits that surface at all.** In the near sweep, sadness appearance climbs 0→0.08→0.25→0.42 (L1–L4) and its mean rank improves toward the top; at `sysend`, sycophancy climbs from absent (L0–1) to rank 1–2 (L4–5). So the answer to "does a stronger sycophancy prompt push the sycophancy line higher in the list?" is **yes, but only when the activation is read at/near the persona text** — it's gone after any real intervening content.

**3. Affective/behavioral personas surface; concrete object/color preferences do not.**
sadness > sycophancy ≫ yellow ≈ tofu (never, at any distance or strength). The AV
verbalizes "melancholic persona" / "personality bot with flattery" from the residual
stream, but does not surface "yellow" or "tofu" even when the persona is entirely
about them and read immediately. (Note these are CHAT-format activations, which are
out-of-distribution for the AV — trained on raw-web-text activations — so absolute
rates understate what a same-distribution probe might find.)

See `results/persona_examples.md` for representative explanations across strengths.

## Reproduce
A 48 GB GPU is plenty. Box already set up by the FVE-sweep `setup_box.sh` needs only
the base model added; or:
```bash
# on a stock pytorch box with the repo at /workspace/nla and /root/.hf_token:
pip install -e /workspace/nla transformers==4.57.1 pyarrow pyyaml safetensors \
    "huggingface_hub>=0.34,<1.0" hf_transfer accelerate matplotlib
python - <<'PY'
from huggingface_hub import snapshot_download
import shutil
snapshot_download("Qwen/Qwen2.5-7B-Instruct", local_dir="/workspace/models/qwen2.5-7b-instruct", max_workers=16)
snapshot_download("syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard", allow_patterns="kl0.01/iter_0000200/av/*", local_dir="/workspace/dl", max_workers=16)
shutil.move("/workspace/dl/kl0.01/iter_0000200/av", "/workspace/av_ckpt")   # NOT /workspace/av (see gotcha)
PY
cd /workspace
PYTHONPATH=/workspace/nla python /workspace/nla/.../persona_probe/persona_probe.py        # far
PYTHONPATH=/workspace/nla python /workspace/nla/.../persona_probe/persona_sweep_near.py    # near (main)
PYTHONPATH=/workspace/nla python /workspace/nla/.../persona_probe/persona_diag.py          # immediate
# pull results/*.{csv,json}, then locally: python plot.py
```

### ⚠️ Gotcha: don't put the AV checkpoint at `/workspace/av`
A dir literally named `av` on `sys.path` (CWD `/workspace` is on it) shadows the
**PyAV** package that torchvision imports (transitively via transformers) →
`module 'av' has no attribute 'logging'`, surfacing confusingly as
`Could not find Qwen2ForCausalLM`. Put the AV at `/workspace/av_ckpt`. The
`persona_diag.py`/`persona_sweep_near.py` scripts also use `Qwen2ForCausalLM`
directly instead of `AutoModelForCausalLM` to avoid the lazy auto-mapping fragility.

## Data preserved for future analysis
`results/persona_{probe,near}_raw.json` (full per-sample explanations + parsed
trait-rank, far & near), `persona_diag_raw.json` (immediate/userend), the two
`*_agg.csv` summaries, `persona_near.png`, and `persona_examples.md`.
