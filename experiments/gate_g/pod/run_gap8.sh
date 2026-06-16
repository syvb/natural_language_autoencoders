#!/bin/bash
# Gate G gap-8 pipeline (16->24): gen AV(h24)=target and AV(h16)=cond-control,
# build arms, train conditioned critics. Env overridable for other gaps.
set -uo pipefail
cd /work/gate_g
COND=${COND:-/work/gate_c/acts16.npy}        # injected condition state
TARGET=${TARGET:-/work/gate_c/acts24.npy}    # reconstruction target
TGT_SRC=${TGT_SRC:-acts24.npy}; TGTAV=${TGTAV:-av_h24_out}
CND_SRC=${CND_SRC:-acts16.npy}; CONDAV=${CONDAV:-av_h16_out}

python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.85 --context-length 1024 > sglang_g8.log 2>&1 &
for i in $(seq 1 120); do curl -sf localhost:30000/health_generate >/dev/null 2>&1 && { echo sglang_up; break; }; sleep 5; done
curl -sf localhost:30000/health_generate >/dev/null || { echo SERVER_NOT_UP; exit 1; }
SRC_ACTS=$TGT_SRC OUT_DIR=$TGTAV  python gen_av.py > gen_g8_tgt.log 2>&1
echo GEN_TGT_DONE
SRC_ACTS=$CND_SRC OUT_DIR=$CONDAV python gen_av.py > gen_g8_cnd.log 2>&1
echo GEN_CND_DONE
pkill -f 'sglang.launch_serve[r]' 2>/dev/null || true; sleep 8

TGTAV="$TGTAV" CONDAV="$CONDAV" python3 - <<'PY'
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
        (W / f'arm_g8{arm}_{tag}.json').write_text(json.dumps(rows)); print('arm_g8' + arm, tag, len(rows))
PY
echo BUILD_G8_DONE

for arm in none cat scond text; do
  python train_critic_cond.py --pairs arm_g8${arm}_train.json --eval-pairs arm_g8${arm}_eval.json \
    --cond "$COND" --targets "$TARGET" --run-name g8_${arm} --out ckpt_g8_${arm} > train_g8_${arm}.log 2>&1
  echo "G8_DONE_${arm} cos=$(python3 -c "import json;print(round(json.load(open('ckpt_g8_${arm}/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
done
echo G8_ALL_DONE >> retrain.log
