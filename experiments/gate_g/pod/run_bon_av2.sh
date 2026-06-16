#!/bin/bash
# Gate G BoN-AV2 (gap 16->24): co-trained AV2/AR2, expert-iteration round.
# Reward = two-text AR2 (holds AV(h16) before-text) -> BoN rewards diff content.
# Fresh AR2 measures whether AV2's text beats frozen AV2 (ptB 0.8273).
set -uo pipefail
cd /work/gate_g
export WANDB_API_KEY=$(cat ~/.wandb_key)
TGT=/work/gate_c/acts24.npy

python extract_gap8.py > extract.log 2>&1; echo EXTRACT_DONE
python build_puretext.py > build_pt.log 2>&1

# reward AR2 = ptB (frozen AV2 after-text), saved
python train_critic_cond.py --no-cond --pairs arm_ptB_train.json --eval-pairs arm_ptB_eval.json \
  --targets $TGT --save-model --run-name av2_rewardAR2 --out reward_ar2 > rm.log 2>&1
echo "REWARD_AR2_DONE cos=$(python3 -c "import json;print(round(json.load(open('reward_ar2/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"

# BoN: sample AV2 candidates (frozen AV, inject h24), score with two-text AR2
python sample_av.py --src acts24.npy --split train --n 8 --temp 0.8 --out cand_av2.jsonl > sample.log 2>&1
echo SAMPLE_DONE
python score_select_text.py --target acts24.npy --candidates cand_av2.jsonl --before av_h16_out \
  --reward-dir reward_ar2 --out best_av2.jsonl > score.log 2>&1
echo SCORE_DONE
python sft_av.py --src acts24.npy --best best_av2.jsonl --out av2_lora > sft.log 2>&1
echo SFT_DONE
python sample_av.py --src acts24.npy --split all --adapter av2_lora --n 1 --temp 0.3 --out av2_descs.jsonl > sampleav2.log 2>&1
echo AV2DESC_DONE

# eval: fresh two-text AR2 on AV2's descriptions
python3 - <<'PY'
import json, glob
from pathlib import Path
W=Path('/work/gate_g'); meta=json.load(open('/work/gate_c/meta2.json'))
def loadexp(d):
    av={}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r=json.loads(l); av[r['i']]=r.get('explanation') or ''
    return av
h16=loadexp('av_h16_out'); av2={}
for l in open('av2_descs.jsonl'):
    r=json.loads(l); av2[r['i']]=(r['samples'][0] if r.get('samples') else '')
def has(i): return bool((h16.get(i) or '').strip()) and bool((av2.get(i) or '').strip())
for split,tag in (('train','train'),('holdout','eval')):
    rows=[{'text':f"[before] {h16[i]} [after] {av2[i]}",'tidx':i} for i,m in enumerate(meta) if m['split']==split and has(i)]
    (W/f'arm_av2_{tag}.json').write_text(json.dumps(rows)); print('arm_av2',tag,len(rows))
PY
python train_critic_cond.py --no-cond --pairs arm_av2_train.json --eval-pairs arm_av2_eval.json \
  --targets $TGT --run-name av2_eval --out ckpt_av2 > av2eval.log 2>&1
echo "AV2_EVAL_DONE cos=$(python3 -c "import json;print(round(json.load(open('ckpt_av2/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"

# diff-focus metrics: did AV2's text shift toward the change?
python3 - <<'PY'
import json, glob, statistics as st
def loadexp(d):
    av={}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r=json.loads(l); av[r['i']]=r.get('explanation') or ''
    return av
h16=loadexp('av_h16_out'); h24=loadexp('av_h24_out')
av2={}
for l in open('av2_descs.jsonl'):
    r=json.loads(l); av2[r['i']]=(r['samples'][0] if r.get('samples') else '')
common=[i for i in h16 if i in h24 and i in av2 and h16[i] and h24[i] and av2[i]][:3000]
def jac(a,b):
    sa,sb=set(a.lower().split()),set(b.lower().split()); return len(sa&sb)/max(len(sa|sb),1)
print('[diff-focus] len frozenAV(h24):', round(st.mean(len(h24[i].split()) for i in common),1),
      '| AV2(h24):', round(st.mean(len(av2[i].split()) for i in common),1))
print('[diff-focus] Jaccard(AV2, frozenAV-h24):', round(st.mean(jac(av2[i],h24[i]) for i in common),3))
print('[diff-focus] Jaccard(AV2, AV-h16 before):', round(st.mean(jac(av2[i],h16[i]) for i in common),3),
      '| frozenAV-h24 vs h16:', round(st.mean(jac(h24[i],h16[i]) for i in common),3))
PY
echo BON_AV2_ALL_DONE >> retrain.log
