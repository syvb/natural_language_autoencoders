#!/bin/bash
# Gate G s20 control: AV verbalizes h_20 (the BEFORE state) instead of h_22.
# Same description format/quality; if text(h22) > s20(h20), the win is the
# after-state/diff content, not generic description quality.
set -uo pipefail
cd /work/gate_g

python -m sglang.launch_server --model-path models/av --port 30000 \
    --disable-radix-cache --mem-fraction-static 0.85 --context-length 1024 \
    > sglang_s20.log 2>&1 &
for i in $(seq 1 120); do
  curl -sf localhost:30000/health_generate >/dev/null 2>&1 && { echo sglang_up; break; }
  sleep 5
done
curl -sf localhost:30000/health_generate >/dev/null || { echo SERVER_NOT_UP; exit 1; }

SRC_ACTS=acts20.npy OUT_DIR=av_h20_out python gen_av.py > gen_s20.log 2>&1
echo GEN_S20_DONE
pkill -f 'sglang.launch_serve[r]' 2>/dev/null || true
sleep 8

python3 -c "
import json, glob
from pathlib import Path
W = Path('/work/gate_g'); meta = json.load(open('/work/gate_c/meta2.json'))
av = {}
for fn in glob.glob('av_h20_out/*.jsonl'):
    for line in open(fn):
        r = json.loads(line); av[r['i']] = r.get('explanation')
for split, tag in (('train','train'), ('holdout','eval')):
    rows = [{'text': av.get(i) or '', 'tidx': i} for i, m in enumerate(meta)
            if m['split'] == split and (av.get(i) or '').strip()]
    (W / f'arm_s20_{tag}.json').write_text(json.dumps(rows)); print('arm_s20', tag, len(rows))
"
echo BUILD_S20_DONE

python train_critic_cond.py --pairs arm_s20_train.json --eval-pairs arm_s20_eval.json \
   --run-name g2_s20 --out ckpt2_s20 > retrain_s20.log 2>&1
echo "S20_DONE cos=$(python3 -c "import json;print(round(json.load(open('ckpt2_s20/RESULT.json'))['final_eval_cos'],4))" 2>/dev/null)"
echo S20_ALL_DONE >> retrain.log
