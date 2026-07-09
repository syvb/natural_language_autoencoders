"""v2: train per-trait linear probes on the A/B answer (Qwen2.5-7B-Instruct, layer 20).

For every normalized question we build TWO completions -- the behavior-matching letter and the
non-matching letter -- and read the layer-20 residual activation at the answer-letter token.
Label = 1 for the matching completion, 0 for the non-matching one. The matching/non-matching
pair of one question always share a CV fold (grouped by question) so the near-identical prompts
can't leak across the split.

Per trait we produce two directions in raw layer-20 activation space:
  - probe_unit : L2-regularized logistic-regression weight vector (the *trained* probe), mapped
                 back to raw space (w / sigma) and unit-normalized. <- v2's "different choice".
  - md_unit    : difference of class means (mu_match - mu_nonmatch), unit-normalized. CAA-classic
                 baseline, for comparison.
Both are evaluated by grouped 5-fold CV (accuracy + AUC); the saved direction is refit on all data.

v2 choice: extraction uses the instruct CHAT TEMPLATE (faithful CAA on an instruct model), set
USE_CHAT_TEMPLATE=False for raw-text extraction instead. Nothing is injected into the AV here.

Run later on the GPU box (base model at BASE, repo on PYTHONPATH). Outputs to OUT.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from datasets_v2 import load_all

BASE = "/workspace/models/qwen2.5-7b-instruct"
LAYER = 20
D_MODEL = 3584
OUT = "/workspace/probes_v2_out"
USE_CHAT_TEMPLATE = True     # instruct-format CAA extraction (v2 default); False -> raw text
K_FOLDS = 5
L2_ALPHA = 1e-3              # logistic-regression L2 penalty (standardized features)
LBFGS_ITERS = 100
BATCH = 16
SEED = 0
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ model + extraction
tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
ADD_SPECIAL = not USE_CHAT_TEMPLATE
A_ID = tok("A", add_special_tokens=False)["input_ids"][-1]
B_ID = tok("B", add_special_tokens=False)["input_ids"][-1]
LETTER_ID = {"A": A_ID, "B": B_ID}

model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
model.model.layers = nn.ModuleList(list(layers[: LAYER + 1]))   # truncate -- only need up to L20
cap = {}
layers[LAYER].register_forward_hook(
    lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))


def build_prompt(stem, letter):
    if USE_CHAT_TEMPLATE:
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
        return prefix + "Answer: (" + letter
    return stem + "\n\nAnswer: (" + letter


@torch.no_grad()
def extract(prompts, expect_letters):
    """Return (n, D_MODEL) float32 activations at the answer-letter token (last real token)."""
    enc = [tok(p, add_special_tokens=ADD_SPECIAL)["input_ids"] for p in prompts]
    for e, lt in zip(enc, expect_letters):       # the prompt ends exactly at the letter
        assert e[-1] == LETTER_ID[lt], f"last token {e[-1]} != letter {lt} ({LETTER_ID[lt]})"
    out = np.empty((len(prompts), D_MODEL), np.float32)
    order = np.argsort([len(e) for e in enc], kind="stable")
    for s in range(0, len(order), BATCH):
        bidx = order[s: s + BATCH]
        seqs = [enc[j] for j in bidx]
        m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), PAD, np.int64)
        att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs):
            inp[k, : len(q)] = q
            att[k, : len(q)] = 1
        cap.clear()
        model.model(input_ids=torch.from_numpy(inp).to(dev),
                    attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for k, j in enumerate(bidx):
            out[j] = cap["h"][k, len(seqs[k]) - 1].float().cpu().numpy()
    return out


# ------------------------------------------------------------------ probe + metrics
def fit_logreg(X, y):
    """L2 logistic regression via LBFGS on standardized X. Returns weight (d,), bias."""
    Xt = torch.tensor(X, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    w = torch.zeros(X.shape[1], device=dev, requires_grad=True)
    b = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=LBFGS_ITERS, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(Xt @ w + b, yt) + L2_ALPHA * (w @ w)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().cpu().numpy(), float(b.detach().cpu())


def auc(y, score):
    y = np.asarray(y)
    ranks = np.argsort(np.argsort(score)) + 1
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def grouped_folds(groups, k, seed):
    ug = np.unique(groups)
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    return [set(f.tolist()) for f in np.array_split(ug, k)]


def cv_eval(X, y, g):
    """Grouped k-fold CV accuracy/AUC for both the LR probe and the mean-diff baseline."""
    acc = {"probe": [], "md": []}
    au = {"probe": [], "md": []}
    for test_groups in grouped_folds(g, K_FOLDS, SEED):
        te = np.array([gi in test_groups for gi in g])
        tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        # LR probe
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr])
        z = ((X[te] - mu) / sd) @ w + b
        acc["probe"].append(float(((z > 0).astype(int) == y[te]).mean()))
        au["probe"].append(auc(y[te], z))
        # mean-diff baseline: project, threshold at midpoint of train projections
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0)
        d /= np.linalg.norm(d) + 1e-8
        thr = 0.5 * (X[tr][y[tr] == 1] @ d).mean() + 0.5 * (X[tr][y[tr] == 0] @ d).mean()
        s = X[te] @ d
        acc["md"].append(float(((s > thr).astype(int) == y[te]).mean()))
        au["md"].append(auc(y[te], s))
    return {k: float(np.mean(v)) for k, v in acc.items()}, {k: float(np.mean(v)) for k, v in au.items()}


# ------------------------------------------------------------------ run
items_by_trait = load_all()
saved = {}
metrics = {}
for trait, items in items_by_trait.items():
    prompts, labels, groups, letters = [], [], [], []
    for it in items:
        prompts.append(build_prompt(it.stem, it.matching_letter)); labels.append(1)
        groups.append(it.idx); letters.append(it.matching_letter)
        prompts.append(build_prompt(it.stem, it.nonmatching_letter)); labels.append(0)
        groups.append(it.idx); letters.append(it.nonmatching_letter)
    print(f"[{trait}] {len(items)} questions -> {len(prompts)} completions; extracting...", flush=True)
    X = extract(prompts, letters)
    y = np.array(labels)
    g = np.array(groups)

    acc, au = cv_eval(X, y, g)

    # full-data directions
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w, _ = fit_logreg((X - mu) / sd, y)
    probe_raw = w / sd                                   # back to raw activation space
    probe_unit = probe_raw / (np.linalg.norm(probe_raw) + 1e-8)
    mu_pos, mu_neg = X[y == 1].mean(0), X[y == 0].mean(0)
    md = mu_pos - mu_neg
    md_unit = md / (np.linalg.norm(md) + 1e-8)
    cos = float(probe_unit @ md_unit)

    saved[f"{trait}_probe_unit"] = probe_unit.astype(np.float32)
    saved[f"{trait}_md_unit"] = md_unit.astype(np.float32)
    saved[f"{trait}_mu_pos"] = mu_pos.astype(np.float32)
    saved[f"{trait}_mu_neg"] = mu_neg.astype(np.float32)
    saved[f"{trait}_feat_mu"] = mu.astype(np.float32)
    saved[f"{trait}_feat_sd"] = sd.astype(np.float32)
    metrics[trait] = {
        "n_questions": len(items), "n_completions": len(prompts),
        "probe_cv_acc": acc["probe"], "probe_cv_auc": au["probe"],
        "md_cv_acc": acc["md"], "md_cv_auc": au["md"],
        "cos_probe_md": cos, "md_norm": float(np.linalg.norm(md)),
    }
    np.savez(f"{OUT}/{trait}_acts.npz", X=X.astype(np.float16), y=y, g=g)
    print(f"  probe acc/auc {acc['probe']:.3f}/{au['probe']:.3f} | "
          f"mean-diff acc/auc {acc['md']:.3f}/{au['md']:.3f} | cos(probe,md) {cos:.3f}", flush=True)

np.savez(f"{OUT}/probes_v2.npz",
         meta=json.dumps({"layer": LAYER, "use_chat_template": USE_CHAT_TEMPLATE,
                          "A_ID": int(A_ID), "B_ID": int(B_ID)}),
         **saved)
json.dump(metrics, open(f"{OUT}/probe_metrics.json", "w"), indent=2)
print("\n=== summary ===")
for t, m in metrics.items():
    print(f"{t:12s} probe {m['probe_cv_acc']:.3f}/{m['probe_cv_auc']:.3f}  "
          f"md {m['md_cv_acc']:.3f}/{m['md_cv_auc']:.3f}  cos {m['cos_probe_md']:.3f}")
print("PROBE_TRAIN_DONE")
