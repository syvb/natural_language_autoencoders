# Gate C — trained-critic test (in progress)

Can a *trained* reader extract beyond-content layer-diff information from
text labels? Four critic arms (content-control L18+L22, diffav, hybrid,
+full-n slope), all init from the released L20 AR, evaluated held-out with
the causal-KL judge. Design reviewed adversarially before spend (see
session notes); pre-registered decision rules in the PR description.

Data artifacts (activations, generations, arm files) are on the HF Hub:
`syvb/nla-layer-diff-experiments` (public). Scripts here:

| file | role |
|---|---|
| `extract2.py` | 15k positions × layers 18/20/22, token IDs stored |
| `gen2.py` | sharded SGLang generation (fast embed path, resumable jsonl) |
| `build_arms.py` | merge outputs → targets, arm pair files, hybrid inputs |
| `hybrid_v2.py` | hybrid diff-labeler (Anthropic API, direct + batch modes) |
| `train_critic.py` | standalone critic SFT (no Miles), wandb logging |
| `run*.sh` | pod drivers |
