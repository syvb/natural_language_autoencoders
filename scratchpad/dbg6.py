import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
tok=open("/root/.hf_token").read().strip()
for repo in ["syvb/nla-qwen2.5-7b-L20-ar-matryoshka-sonnet46"]:
    try:
        p=hf_hub_download(repo,"value_head.safetensors",token=tok)
        v=load_file(p)["weight"].float()
        bad=((~torch.isfinite(v))|(v.abs()>10))
        print(f"{repo}: bad={bad.float().mean()*100:.3f}%  median|w|={v[torch.isfinite(v)].abs().median().item():.4f}")
    except Exception as e:
        print(repo,"ERR",repr(e)[:120])
