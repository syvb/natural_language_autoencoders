import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors.torch import load_file
tok=open("/root/.hf_token").read().strip()
def check(repo, fn):
    try:
        p=hf_hub_download(repo, fn, token=tok)
        v=load_file(p)["weight"].float()
        bad=((~torch.isfinite(v))|(v.abs()>10))
        print(f"OK  {repo}/{fn}: bad={bad.float().mean()*100:.3f}%  median|w|={v[torch.isfinite(v)].abs().median().item():.4f}")
    except Exception as e:
        print(f"ERR {repo}/{fn}: {repr(e)[:80]}")
for repo in ["syvb/nla-qwen2.5-7b-L20-ar","kitft/nla-qwen2.5-7b-L20-ar"]:
    try:
        fs=list_repo_files(repo, token=tok)
        vh=[f for f in fs if "value_head" in f]
        print(repo, "value_head files:", vh)
        for f in vh: check(repo, f)
    except Exception as e:
        print("ERR list", repo, repr(e)[:80])
