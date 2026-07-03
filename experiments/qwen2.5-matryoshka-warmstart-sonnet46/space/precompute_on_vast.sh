#!/bin/bash
# Regenerate precache.json on a rented Vast.ai GPU, fetch it back, destroy the box.
#
# Inference-only job: cheapest ≥24GB-VRAM Ampere+ card works (3090/4090/A6000);
# prefers free-bandwidth hosts since it downloads ~45GB of checkpoints.
# Total cost ≈ $0.5, wall-clock ≈ 30-60 min (mostly downloads).
#
# Usage:
#   bash precompute_on_vast.sh                 # provision → run → fetch → destroy
#   STEPS=dirs bash precompute_on_vast.sh      # only rebuild steering_dirs.npz
#   STEPS=cache bash precompute_on_vast.sh     # only resample precache.json
#   INSTANCE_ID=1234567 bash precompute_on_vast.sh   # reuse an existing box
#   KEEP_BOX=1 bash precompute_on_vast.sh      # don't destroy afterwards
#
# STEPS defaults to "cache,dirs". NOTE: "cache" RESAMPLES every explanation —
# don't include it if published material depends on the current samples.
#
# Needs: ~/.hf_token, vastai CLI (~/.local/bin/vastai) with your SSH key registered.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
VASTAI="${VASTAI:-$HOME/.local/bin/vastai}"
LABEL="nla-precache"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

created_here=0
if [[ -z "${INSTANCE_ID:-}" ]]; then
  echo "=== search offers (≥24GB Ampere+, free bandwidth preferred, US/CA) ==="
  QUERY='gpu_name in [RTX_4090,RTX_3090,RTX_A6000,A40] num_gpus=1 reliability>0.99 disk_space>200 inet_down>500 rentable=true cuda_vers>=12.4 geolocation in [US,CA]'
  OFFER=$("$VASTAI" search offers "$QUERY" --raw | python3 -c "
import sys, json
rows = json.load(sys.stdin)
assert rows, 'no offers matched the query'
rows.sort(key=lambda o: (o.get('inet_down_cost') or 0, o['dph_total']))
o = rows[0]
print(o['id'])
print(f\"picked {o['gpu_name']} {o['dph_total']:.3f}/hr, bw {(o.get('inet_down_cost') or 0):.4f}/GB, {o.get('geolocation')}\", file=sys.stderr)")

  echo "=== create instance (offer $OFFER, disk 200GB) ==="
  INSTANCE_ID=$("$VASTAI" create instance "$OFFER" --image "$IMAGE" --disk 200 \
    --ssh --direct --label "$LABEL" \
    --onstart-cmd 'touch ~/.no_auto_tmux; sleep infinity' --raw \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['new_contract'])")
  created_here=1
fi
echo "instance: $INSTANCE_ID"

echo "=== wait for running ==="
for i in $(seq 1 60); do
  STATUS=$("$VASTAI" show instance "$INSTANCE_ID" --raw \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('actual_status') or 'none')")
  echo "  [$i] $STATUS"
  [[ "$STATUS" == "running" ]] && break
  # image pulled but not auto-started (ENV.md boot gotcha)
  [[ "$STATUS" == "stopped" ]] && "$VASTAI" start instance "$INSTANCE_ID" || true
  sleep 20
done
[[ "$STATUS" == "running" ]] || { echo "box never reached running"; exit 1; }

OUT=$("$VASTAI" show instance "$INSTANCE_ID" --raw)
IP=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['public_ipaddr'])")
PORT=$(echo "$OUT" | python3 -c "import sys,json;print(json.load(sys.stdin)['ports']['22/tcp'][0]['HostPort'])")
echo "ssh root@$IP -p $PORT"

echo "=== wait for sshd ==="
for i in $(seq 1 30); do
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$IP" true 2>/dev/null && break
  sleep 10
done

echo "=== push token + code ==="
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$IP" \
  "umask 077; printf '%s' '$(cat ~/.hf_token)' > /root/.hf_token; mkdir -p /workspace/precache"
rsync -az -e "ssh ${SSH_OPTS[*]} -p $PORT" \
  "$HERE/precompute_cache.py" "$HERE/build_steering_dirs.py" \
  "$HERE/default_texts.json" "$HERE/mu.npy" \
  "$REPO_ROOT/nla_inference.py" "root@$IP:/workspace/precache/"

echo "=== install deps ==="
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$IP" \
  '/opt/conda/bin/pip install -q transformers==4.57.1 "huggingface_hub>=0.34,<1.0" \
     safetensors pyyaml numpy accelerate hf_transfer orjson httpx'

STEPS="${STEPS:-cache,dirs}"
if [[ ",$STEPS," == *",dirs,"* ]]; then
  echo "=== build steering directions ==="
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$IP" \
    'cd /workspace/precache && HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=1 \
     /opt/conda/bin/python build_steering_dirs.py --out /workspace/precache/steering_dirs.npz'
  scp "${SSH_OPTS[@]}" -P "$PORT" "root@$IP:/workspace/precache/steering_dirs.npz" "$HERE/steering_dirs.npz"
fi

if [[ ",$STEPS," == *",cache,"* ]]; then
  echo "=== run precompute (streams; ~20-40 min incl. 45GB of downloads) ==="
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$IP" \
    'cd /workspace/precache && HF_TOKEN=$(cat /root/.hf_token) HF_HUB_ENABLE_HF_TRANSFER=1 \
     /opt/conda/bin/python precompute_cache.py --out /workspace/precache/precache.json'

  echo "=== fetch precache.json ==="
  scp "${SSH_OPTS[@]}" -P "$PORT" "root@$IP:/workspace/precache/precache.json" "$HERE/precache.json"
  python3 -c "
import json
d = json.load(open('$HERE/precache.json'))
n = sum(len(e['results']) for e in d['entries'])
ok = sum(r is not None for e in d['entries'] for r in e['results'])
print(f\"precache.json: {len(d['entries'])} texts, {ok}/{n} positions, gpu={d['meta']['gpu']}\")"
fi

if [[ "$created_here" == 1 && -z "${KEEP_BOX:-}" ]]; then
  echo "=== destroy instance $INSTANCE_ID ==="
  echo y | "$VASTAI" destroy instance "$INSTANCE_ID"
else
  echo "NOT destroying box $INSTANCE_ID (reused or KEEP_BOX set) — destroy it yourself:"
  echo "  echo y | $VASTAI destroy instance $INSTANCE_ID"
fi
echo "done. Redeploy the Space with: bash $HERE/deploy.sh"
