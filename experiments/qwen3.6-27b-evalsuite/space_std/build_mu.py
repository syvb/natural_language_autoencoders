"""Rebuild mu.npy — the FVE baseline for the Qwen3.6-27B explorer Space.

FVE_k = 1 − ||n(v̂_k)−n(v)||² / ||n(v)−μ||², where n(·) rescales to
mse_scale=√d_model and μ is the population mean of the normalized held-out
activations. This mirrors suite_fve.fve()'s `mu = gn.mean(0)`, but over the
FULL av_eval set (5003 rows) for a stable, click-independent baseline.

No GPU needed — just reads raw activation vectors from the released eval
parquet. Run whenever the eval set or d_model/mse_scale changes:

    python build_mu.py --out mu.npy
"""
import argparse
import math

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATA_REPO = "ceselder/nla-qwen36-27b-matryoshka-data"
PARQUET = "av_eval.parquet"
D_MODEL = 5120
MSE_SCALE = math.sqrt(D_MODEL)  # ≈ 71.55


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="mu.npy")
    args = ap.parse_args()

    path = hf_hub_download(DATA_REPO, PARQUET, repo_type="dataset")
    pf = pq.ParquetFile(path)
    acc = np.zeros(D_MODEL, np.float64)
    n = 0
    for batch in pf.iter_batches(batch_size=512, columns=["activation_vector"]):
        flat = batch.column("activation_vector").flatten().to_numpy(
            zero_copy_only=False).astype(np.float32)
        v = flat.reshape(len(batch), -1)
        assert v.shape[1] == D_MODEL, v.shape
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1e-12
        acc += (v / norms * MSE_SCALE).sum(0, dtype=np.float64)
        n += len(batch)
    mu = (acc / n).astype(np.float32)
    np.save(args.out, mu)
    print(f"wrote {args.out}: {n} rows, ||mu||={np.linalg.norm(mu):.2f}, "
          f"mse_scale={MSE_SCALE:.2f}")


if __name__ == "__main__":
    main()
