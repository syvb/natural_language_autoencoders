#!/bin/bash
# Gate G GRPO (gap 16->24): RL the AV toward concise diff-focused descriptions.
set -uo pipefail
cd /work/gate_g
export WANDB_API_KEY=$(cat ~/.wandb_key)
TGT=/work/gate_c/acts24.npy
LAM=${LAM:-0.05}; STEPS=${STEPS:-250}

[ -f /work/gate_c/acts24.npy ] || python extract_gap8.py > extract.log 2>&1
echo EXTRACT_DONE
python build_puretext.py > build_pt.log 2>&1
if [ ! -f reward_ar2/RESULT.json ]; then
  python train_critic_cond.py --no-cond --pairs arm_ptB_train.json --eval-pairs arm_ptB_eval.json \
    --targets $TGT --save-model --run-name grpo_rewardAR2 --out reward_ar2 > rm.log 2>&1
fi
echo "REWARD_AR2_DONE cos=$(python3 -c "import json;print(round(json.load(open('reward_ar2/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"

python grpo_av.py --reward-dir reward_ar2 --lam $LAM --steps $STEPS --out grpo_lora > grpo.log 2>&1
echo "GRPO_DONE $(grep -hoE "recon [0-9.]+ words [0-9]+" grpo.log | tail -1)"

python sample_av.py --src acts24.npy --split all --adapter grpo_lora --n 1 --temp 0.3 --out grpo_descs.jsonl > sampleg.log 2>&1
echo GRPODESC_DONE
python3 - <<'PY'
import json, glob
from pathlib import Path
W=Path('/work/gate_g'); meta=json.load(open('/work/gate_c/meta2.json'))
def le(d):
    a={}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r=json.loads(l); a[r['i']]=r.get('explanation') or ''
    return a
h16=le('av_h16_out'); g={}
for l in open('grpo_descs.jsonl'):
    r=json.loads(l); g[r['i']]=(r['samples'][0] if r.get('samples') else '')
def has(i): return bool((h16.get(i) or '').strip()) and bool((g.get(i) or '').strip())
for split,tag in (('train','train'),('holdout','eval')):
    rows=[{'text':f"[before] {h16[i]} [after] {g[i]}",'tidx':i} for i,m in enumerate(meta) if m['split']==split and has(i)]
    (W/f'arm_grpo_{tag}.json').write_text(json.dumps(rows)); print('arm_grpo',tag,len(rows))
PY
python train_critic_cond.py --no-cond --pairs arm_grpo_train.json --eval-pairs arm_grpo_eval.json \
  --targets $TGT --run-name grpo_eval --out ckpt_grpo > grpoeval.log 2>&1
echo "GRPO_EVAL_DONE cos=$(python3 -c "import json;print(round(json.load(open('ckpt_grpo/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
python3 - <<'PY'
import json, glob, statistics as st
def le(d):
    a={}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r=json.loads(l); a[r['i']]=r.get('explanation') or ''
    return a
h16=le('av_h16_out'); h24=le('av_h24_out')
g={}
for l in open('grpo_descs.jsonl'):
    r=json.loads(l); g[r['i']]=(r['samples'][0] if r.get('samples') else '')
common=[i for i in h16 if i in h24 and i in g and h16[i] and h24[i] and g[i]][:3000]
def jac(a,b):
    sa,sb=set(a.lower().split()),set(b.lower().split()); return len(sa&sb)/max(len(sa|sb),1)
print('[diff-focus] len frozenAV(h24):', round(st.mean(len(h24[i].split()) for i in common),1),
      '| GRPO-AV(h24):', round(st.mean(len(g[i].split()) for i in common),1))
print('[diff-focus] Jaccard(GRPO,before):', round(st.mean(jac(g[i],h16[i]) for i in common),3),
      '| frozenAV-h24 vs before:', round(st.mean(jac(h24[i],h16[i]) for i in common),3))
PY
echo GRPO_ALL_DONE >> retrain.log
