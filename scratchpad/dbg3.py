import torch
from huggingface_hub import hf_hub_download
tok=open("/root/.hf_token").read().strip()
from safetensors.torch import load_file
for sub in ["kl0.01/iter_0000200","kl0.01/iter_0000100","kl0/iter_0000200","kl0/iter_0000100"]:
    try:
        p=hf_hub_download("syvb/nla-qwen2.5-7b-L20-rltrunc-gradguard", f"{sub}/ar/value_head.safetensors", token=tok)
        v=load_file(p)["weight"]
        nan=torch.isnan(v).float().mean().item(); inf=torch.isinf(v).float().mean().item()
        finite=torch.isfinite(v)
        fm=v[finite].abs().max().item() if finite.any() else float('nan')
        print(f"{sub:28s} nan_frac={nan:.4f} inf_frac={inf:.4f} finite_absmax={fm:.3f}")
    except Exception as e:
        print(sub, "ERR", repr(e)[:100])
