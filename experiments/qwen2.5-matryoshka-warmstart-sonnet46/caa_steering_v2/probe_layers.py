"""Train A/B probe + mean-diff directions at several layers (not just 20).

Same A/B-answer extraction as probe_train.py, but captures layers {8,12,16,20} in one forward pass.
Per (trait, layer): grouped 5-fold CV accuracy/AUC for the LR probe and the mean-diff baseline, and
the unit directions (raw activation space). Also records each layer's mean residual L2 norm, so
later steering strength can be set as a fraction of it (fair across layers).

Outputs probes_layers.npz (directions + resid norms) and probe_layers_metrics.json.
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
LAYERS = [8, 12, 16, 20]
D = 3584
OUT = "/workspace/probes_layers_out"
K_FOLDS = 5
L2_ALPHA = 1e-3
LBFGS_ITERS = 100
BATCH = 16
SEED = 0
dev = "cuda"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
PAD = tok.pad_token_id
LETTER_ID = {"A": tok("A", add_special_tokens=False)["input_ids"][-1],
             "B": tok("B", add_special_tokens=False)["input_ids"][-1]}

model = Qwen2ForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda").eval()
layers = model.model.layers
model.model.layers = nn.ModuleList(list(layers[:max(LAYERS) + 1]))   # keep 0..max
caps = {}
for L in LAYERS:
    layers[L].register_forward_hook(lambda m, i, o, L=L: caps.__setitem__(L, o[0] if isinstance(o, tuple) else o))


def build_prompt(stem, letter):
    pre = tok.apply_chat_template([{"role": "user", "content": stem}], add_generation_prompt=True, tokenize=False)
    return pre + "Answer: (" + letter


@torch.no_grad()
def extract(prompts, expect_letters):
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    for e, lt in zip(enc, expect_letters):
        assert e[-1] == LETTER_ID[lt]
    out = {L: np.empty((len(prompts), D), np.float32) for L in LAYERS}
    order = np.argsort([len(e) for e in enc], kind="stable")
    for s in range(0, len(order), BATCH):
        bidx = order[s:s + BATCH]
        seqs = [enc[j] for j in bidx]
        m = max(len(x) for x in seqs)
        inp = np.full((len(seqs), m), PAD, np.int64)
        att = np.zeros((len(seqs), m), np.int64)
        for k, q in enumerate(seqs):
            inp[k, :len(q)] = q
            att[k, :len(q)] = 1
        caps.clear()
        model.model(input_ids=torch.from_numpy(inp).to(dev), attention_mask=torch.from_numpy(att).to(dev), use_cache=False)
        for L in LAYERS:
            for k, j in enumerate(bidx):
                out[L][j] = caps[L][k, len(seqs[k]) - 1].float().cpu().numpy()
    return out


def fit_logreg(X, y):
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


def auc(y, s):
    y = np.asarray(y)
    r = np.argsort(np.argsort(s)) + 1
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")


def folds(g, k, seed):
    ug = np.unique(g)
    np.random.RandomState(seed).shuffle(ug)
    return [set(f.tolist()) for f in np.array_split(ug, k)]


def cv(X, y, g):
    pa, pu, ma, mu = [], [], [], []
    for tg in folds(g, K_FOLDS, SEED):
        te = np.array([gi in tg for gi in g]); tr = ~te
        mu_, sd_ = X[tr].mean(0), X[tr].std(0) + 1e-6
        w, b = fit_logreg((X[tr] - mu_) / sd_, y[tr])
        z = ((X[te] - mu_) / sd_) @ w + b
        pa.append(((z > 0).astype(int) == y[te]).mean()); pu.append(auc(y[te], z))
        d = X[tr][y[tr] == 1].mean(0) - X[tr][y[tr] == 0].mean(0); d /= np.linalg.norm(d) + 1e-8
        thr = 0.5 * (X[tr][y[tr] == 1] @ d).mean() + 0.5 * (X[tr][y[tr] == 0] @ d).mean()
        s = X[te] @ d
        ma.append(((s > thr).astype(int) == y[te]).mean()); mu.append(auc(y[te], s))
    return float(np.mean(pa)), float(np.mean(pu)), float(np.mean(ma)), float(np.mean(mu))


items_by_trait = load_all()
saved = {}
metrics = {}
resid_norm = {L: [] for L in LAYERS}
for trait, items in items_by_trait.items():
    prompts, labels, groups, letters = [], [], [], []
    for it in items:
        prompts += [build_prompt(it.stem, it.matching_letter), build_prompt(it.stem, it.nonmatching_letter)]
        labels += [1, 0]; groups += [it.idx, it.idx]; letters += [it.matching_letter, it.nonmatching_letter]
    print(f"[{trait}] extracting {len(prompts)} completions x{len(LAYERS)} layers...", flush=True)
    acts = extract(prompts, letters)
    y = np.array(labels); g = np.array(groups)
    metrics[trait] = {}
    for L in LAYERS:
        X = acts[L]
        resid_norm[L].append(float(np.linalg.norm(X, axis=1).mean()))
        pa, pu, ma, mu = cv(X, y, g)
        m_, s_ = X.mean(0), X.std(0) + 1e-6
        w, _ = fit_logreg((X - m_) / s_, y)
        praw = w / s_; pu_unit = praw / (np.linalg.norm(praw) + 1e-8)
        md = X[y == 1].mean(0) - X[y == 0].mean(0); md_unit = md / (np.linalg.norm(md) + 1e-8)
        saved[f"{trait}_L{L}_probe_unit"] = pu_unit.astype(np.float32)
        saved[f"{trait}_L{L}_md_unit"] = md_unit.astype(np.float32)
        metrics[trait][L] = {"probe_acc": pa, "probe_auc": pu, "md_acc": ma, "md_auc": mu}
        print(f"  L{L:2d} probe {pa:.3f}/{pu:.3f}  md {ma:.3f}/{mu:.3f}", flush=True)

resid = {L: float(np.mean(resid_norm[L])) for L in LAYERS}
saved["resid_norm"] = np.array([resid[L] for L in LAYERS], np.float32)
saved["layers"] = np.array(LAYERS, np.int64)
np.savez(f"{OUT}/probes_layers.npz", **saved)
json.dump({"metrics": metrics, "resid_norm": resid}, open(f"{OUT}/probe_layers_metrics.json", "w"), indent=2)
print("\nresid norms:", {L: round(resid[L], 1) for L in LAYERS})
print("PROBE_LAYERS_DONE", flush=True)
