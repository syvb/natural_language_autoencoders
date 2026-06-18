import torch, json, glob, sys

def load(tag):
    fs = sorted(glob.glob("/workspace/results/%s/rollout/*.pt" % tag),
                key=lambda p: int(p.split("step_")[1].split(".")[0]))
    d = torch.load(fs[-1], weights_only=False)
    return d["samples"] if isinstance(d, dict) and "samples" in d else d

def field(s, k, default=None):
    return s.get(k, default) if isinstance(s, dict) else getattr(s, k, default)

for tag, label in [("pen0_seed1234", "control lambda=0"), ("pen0.01_seed1234", "lambda=0.01")]:
    ss = load(tag)
    rows = []
    for s in ss:
        r = field(s, "response", "") or ""
        e = r.split("<explanation>")[-1].split("</explanation>")[0].strip() if "<explanation>" in r else r.strip()
        rw = field(s, "reward")
        rows.append({"len": field(s, "response_length"),
                     "reward": round(float(rw), 3) if rw is not None else None,
                     "explanation": e})
    json.dump(rows, open("/workspace/results/%s/final_step_explanations.json" % tag, "w"), indent=1)
    print("=== %s (final training step, %d samples) — 2 examples ===" % (label, len(rows)))
    for s in rows[:2]:
        print("[len=%stok reward=%s] %s" % (s["len"], s["reward"], s["explanation"][:380]))
        print("---")
