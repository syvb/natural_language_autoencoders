import torch
from safetensors.torch import load_file
v=load_file("/workspace/hf_out/ar/value_head.safetensors")["weight"].float()
flat=v.flatten()
n=flat.numel()
nan=torch.isnan(flat); inf=torch.isinf(flat)
print(f"total={n}  nan={int(nan.sum())} ({nan.float().mean()*100:.3f}%)  inf={int(inf.sum())}")
fin=flat[torch.isfinite(flat)]
af=fin.abs()
for thr in [1,10,1e2,1e3,1e6,1e10,1e30]:
    print(f"  |w|>{thr:>8.0e}: {int((af>thr).sum())}")
print("finite |w| percentiles:", [round(torch.quantile(af, q).item(),4) for q in [0.5,0.9,0.99,0.999,0.9999]])
print("finite mean|w|", af.mean().item(), "median|w|", af.median().item())
# rows with any bad entry
bad = (~torch.isfinite(v)) | (v.abs()>10)
print("rows with >=1 bad (nan/inf/|w|>10):", int(bad.any(1).sum()), "/", v.shape[0])
print("total bad entries (nan/inf/|w|>10):", int(bad.sum()), f"({bad.float().mean()*100:.3f}%)")
