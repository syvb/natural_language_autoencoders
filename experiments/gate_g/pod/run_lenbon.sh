#!/bin/bash
# Gate G length-penalized BoN (gap 16->24): iterate so the candidate pool
# ratchets shorter. Selection reward = recon(before+cand) - LAM*(words/100).
# Before-conditioned recon + length pressure => concise, diff-focused AV2.
set -uo pipefail
cd /work/gate_g
export WANDB_API_KEY=$(cat ~/.wandb_key)
TGT=/work/gate_c/acts24.npy
LAM=${LAM:-0.06}
ROUNDS=${ROUNDS:-2}

python extract_gap8.py > extract.log 2>&1; echo EXTRACT_DONE
python build_puretext.py > build_pt.log 2>&1
python train_critic_cond.py --no-cond --pairs arm_ptB_train.json --eval-pairs arm_ptB_eval.json \
  --targets $TGT --save-model --run-name lb_rewardAR2 --out reward_ar2 > rm.log 2>&1
echo "REWARD_AR2_DONE cos=$(python3 -c "import json;print(round(json.load(open('reward_ar2/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"

ADAPTER=""
LAST=""
for R in $(seq 0 $((ROUNDS-1))); do
  python sample_av.py --src acts24.npy --split train --n 8 --temp 0.9 $ADAPTER --out cand_r$R.jsonl > sample_r$R.log 2>&1
  python score_select_text.py --target acts24.npy --candidates cand_r$R.jsonl --before av_h16_out \
    --reward-dir reward_ar2 --lam $LAM --out best_r$R.jsonl > score_r$R.log 2>&1
  python sft_av.py --src acts24.npy --best best_r$R.jsonl --out av2_r$R > sft_r$R.log 2>&1
  echo "ROUND_${R}_DONE $(grep -oE "recon\(best\) [0-9.]+|mean selected length [0-9]+ words" score_r$R.log | tr '\n' ' ')"
  ADAPTER="--adapter av2_r$R"; LAST="av2_r$R"
done

python sample_av.py --src acts24.npy --split all --adapter $LAST --n 1 --temp 0.3 --out av2lb_descs.jsonl > sampleav2.log 2>&1
echo AV2DESC_DONE
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
h16=le('av_h16_out'); av2={}
for l in open('av2lb_descs.jsonl'):
    r=json.loads(l); av2[r['i']]=(r['samples'][0] if r.get('samples') else '')
def has(i): return bool((h16.get(i) or '').strip()) and bool((av2.get(i) or '').strip())
for split,tag in (('train','train'),('holdout','eval')):
    rows=[{'text':f"[before] {h16[i]} [after] {av2[i]}",'tidx':i} for i,m in enumerate(meta) if m['split']==split and has(i)]
    (W/f'arm_lb_{tag}.json').write_text(json.dumps(rows)); print('arm_lb',tag,len(rows))
PY
python train_critic_cond.py --no-cond --pairs arm_lb_train.json --eval-pairs arm_lb_eval.json \
  --targets $TGT --run-name lb_eval --out ckpt_lb > lbeval.log 2>&1
echo "LB_EVAL_DONE cos=$(python3 -c "import json;print(round(json.load(open('ckpt_lb/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
python3 - <<'PY'
import json, glob, statistics as st
def le(d):
    a={}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r=json.loads(l); a[r['i']]=r.get('explanation') or ''
    return a
h16=le('av_h16_out'); h24=le('av_h24_out')
av2={}
for l in open('av2lb_descs.jsonl'):
    r=json.loads(l); av2[r['i']]=(r['samples'][0] if r.get('samples') else '')
common=[i for i in h16 if i in h24 and i in av2 and h16[i] and h24[i] and av2[i]][:3000]
def jac(a,b):
    sa,sb=set(a.lower().split()),set(b.lower().split()); return len(sa&sb)/max(len(sa|sb),1)
print('[diff-focus] len frozenAV(h24):', round(st.mean(len(h24[i].split()) for i in common),1),
      '| AV2(h24):', round(st.mean(len(av2[i].split()) for i in common),1))
print('[diff-focus] Jaccard(AV2,before):', round(st.mean(jac(av2[i],h16[i]) for i in common),3),
      '| frozenAV-h24 vs before:', round(st.mean(jac(h24[i],h16[i]) for i in common),3))
PY
echo LENBON_ALL_DONE >> retrain.log
