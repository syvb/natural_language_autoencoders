#!/bin/bash
# Gate G gap-4 (20->24): NO new gen — reuses av_h24_out (target, from gap-8 run)
# and av_h20_out (cond-control, from Stage 0). Build arms + train 4 critics.
set -uo pipefail
cd /work/gate_g
COND=/work/gate_c/acts20.npy; TARGET=/work/gate_c/acts24.npy
export TGTAV=av_h24_out CONDAV=av_h20_out
python3 - <<'PY'
import json, glob, os
from pathlib import Path
W = Path('/work/gate_g'); meta = json.load(open('/work/gate_c/meta2.json'))
def load(d):
    av = {}
    for fn in glob.glob(f'{d}/*.jsonl'):
        for l in open(fn):
            r = json.loads(l); av[r['i']] = r.get('explanation')
    return av
tgt = load(os.environ['TGTAV']); cond = load(os.environ['CONDAV'])
srcs = {'none': lambda i: '', 'cat': lambda i: meta[i]['context_tail'],
        'text': lambda i: tgt.get(i) or '', 'scond': lambda i: cond.get(i) or ''}
for arm, fn in srcs.items():
    for split, tag in (('train', 'train'), ('holdout', 'eval')):
        rows = [{'text': fn(i), 'tidx': i} for i, m in enumerate(meta)
                if m['split'] == split and not (arm in ('text', 'scond') and not (fn(i) or '').strip())]
        (W / f'arm_g4{arm}_{tag}.json').write_text(json.dumps(rows)); print('arm_g4' + arm, tag, len(rows))
PY
echo BUILD_G4_DONE
for arm in none cat scond text; do
  python train_critic_cond.py --pairs arm_g4${arm}_train.json --eval-pairs arm_g4${arm}_eval.json \
    --cond "$COND" --targets "$TARGET" --run-name g4_${arm} --out ckpt_g4_${arm} > train_g4_${arm}.log 2>&1
  echo "G4_DONE_${arm} cos=$(python3 -c "import json;print(round(json.load(open('ckpt_g4_${arm}/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
done
echo G4_ALL_DONE >> retrain.log
