import torch
from safetensors.torch import load_file
sd = load_file("/workspace/hf_out/ar/value_head.safetensors")
for k,v in sd.items():
    print(k, tuple(v.shape), v.dtype, "nan?", torch.isnan(v).any().item(), "inf?", torch.isinf(v).any().item(), "absmax", v.float().abs().max().item())
