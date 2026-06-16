#!/bin/bash
# Gate G Stage 1 (BoN-SFT, gap 16->24). Tier 0 throughput guard + tier 1 BoN.
# Reward critic scores candidates; AV is SFT'd on best-of-N; a FRESH eval critic
# measures whether the BoN-AV's text beats the frozen AV's (gap-8 text=0.8223).
set -uo pipefail
cd /work/gate_g
export WANDB_API_KEY=$(cat ~/.wandb_key)
COND=/work/gate_c/acts16.npy; TARGET=/work/gate_c/acts24.npy; SRC=acts24.npy

# --- reward critic (gap-8 text arm, saved as reward model) ---
python train_critic_cond.py --pairs arm_g8text_train.json --eval-pairs arm_g8text_eval.json \
  --cond $COND --targets $TARGET --save-model --run-name g8_text_rm --out ckpt_g8_text_rm > rm_train.log 2>&1
echo RM_DONE

# --- tier 0: batched-sampling throughput guard ---
python sample_av.py --src $SRC --split train --n 8 --temp 0.8 --limit 96 --out cand_smoke.jsonl > sample_smoke.log 2>&1
RATE=$(grep -oE "[0-9.]+ seq/s" sample_smoke.log | tail -1 | grep -oE "[0-9.]+")
echo "TIER0_DONE rate=${RATE} seq/s"
ok=$(python3 -c "print(1 if float('${RATE:-0}')>=15 else 0)")
if [ "$ok" != "1" ]; then echo "SAMPLING_TOO_SLOW (${RATE}/s) — aborting before BoN"; echo STAGE1_ABORT >> retrain.log; exit 0; fi

# --- tier 1: BoN over train ---
python sample_av.py --src $SRC --split train --n 8 --temp 0.8 --out cand_h24_train.jsonl > sample_train.log 2>&1
echo SAMPLE_DONE
python score_select.py --cond acts16.npy --target acts24.npy --candidates cand_h24_train.jsonl \
  --reward-dir ckpt_g8_text_rm --out best_h24.jsonl > score.log 2>&1
echo SCORE_DONE
python sft_av.py --src $SRC --best best_h24.jsonl --out av_bon_lora > sft.log 2>&1
echo SFT_DONE
# BoN-AV's own descriptions for ALL positions (the eval text arm)
python sample_av.py --src $SRC --split all --adapter av_bon_lora --n 1 --temp 0.3 --out av_bon_descs.jsonl > sample_bon.log 2>&1
echo BONDESC_DONE
python3 - <<'PY'
import json
from pathlib import Path
W=Path('/work/gate_g'); meta=json.load(open('/work/gate_c/meta2.json'))
av={}
for l in open('av_bon_descs.jsonl'):
    r=json.loads(l); av[r['i']]=(r['samples'][0] if r.get('samples') else '')
for split,tag in (('train','train'),('holdout','eval')):
    rows=[{'text':av.get(i) or '','tidx':i} for i,m in enumerate(meta)
          if m['split']==split and (av.get(i) or '').strip()]
    (W/f'arm_bon_{tag}.json').write_text(json.dumps(rows)); print('arm_bon',tag,len(rows))
PY
python train_critic_cond.py --pairs arm_bon_train.json --eval-pairs arm_bon_eval.json \
  --cond $COND --targets $TARGET --run-name g8_bon --out ckpt_g8_bon > bon_eval.log 2>&1
echo "BON_EVAL_DONE cos=$(python3 -c "import json;print(round(json.load(open('ckpt_g8_bon/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
echo STAGE1_ALL_DONE >> retrain.log
