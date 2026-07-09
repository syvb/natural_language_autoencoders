import torch, numpy as np, pyarrow.parquet as pq, yaml
from nla_inference import NLACritic
critic = NLACritic("/workspace/hf_out/ar", device="cuda")
t = pq.read_table("/workspace/out/av_eval.parquet")
v0 = np.asarray(t.column("activation_vector")[0].as_py(), dtype=np.float32)
g = torch.tensor(v0)
print("GOLD norm", g.norm().item(), "nan?", torch.isnan(g).any().item())
expl = "The text describes a music feature about a singer who composes her own songs."
# original single reconstruct
p = critic.reconstruct(expl)
print("PRED(single) norm", p.norm().item(), "nan?", torch.isnan(p).any().item())
# batched style
ctok = critic.tokenizer; ctok.padding_side="right"
if ctok.pad_token_id is None: ctok.pad_token=ctok.eos_token
prompts=[critic.template.format(explanation=expl), critic.template.format(explanation="short text")]
enc=ctok(prompts, return_tensors="pt", add_special_tokens=True, padding=True)
ids=enc["input_ids"].cuda(); am=enc["attention_mask"].cuda()
with torch.no_grad():
    hs=critic.backbone.model(ids, attention_mask=am, use_cache=False).last_hidden_state
print("hs dtype", hs.dtype, "nan?", torch.isnan(hs).any().item())
last=am.sum(1)-1
h=hs[torch.arange(hs.size(0)), last]
print("h nan?", torch.isnan(h).any().item(), "h norm", h.float().norm(dim=-1).tolist())
pv=critic.value_head(h).float().cpu()
print("PRED(batch) norm", pv.norm(dim=-1).tolist(), "nan?", torch.isnan(pv).any().item())
