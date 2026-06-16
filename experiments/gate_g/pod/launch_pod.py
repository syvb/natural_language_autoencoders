"""Launch a RunPod GPU for Gate G Stage 0. Prints ssh connection info.

Fast GPU priority (H100 first), falls back to A100 80GB. Only manages pods
THIS script creates — never touches the user's own pod.
"""
import json
import sys
import time
from pathlib import Path

import httpx

KEY = Path("~/.runpod_key").expanduser().read_text().strip()
PUBKEY = Path("~/.ssh/id_ed25519.pub").expanduser().read_text().strip()
HF = Path("~/.hf_token").expanduser().read_text().strip()
WB = Path("~/.wandb_key").expanduser().read_text().strip()
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
BASE = "https://rest.runpod.io/v1"

ENTRY = ('mkdir -p /root/.ssh && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys '
         '&& apt-get update -qq && apt-get install -y -qq openssh-server >/dev/null 2>&1 '
         '&& service ssh start && sleep infinity')

body = {
    "name": "nla-gate-g-puretext",
    "imageName": "lmsysorg/sglang:latest",
    "gpuTypeIds": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 PCIe",
                   "NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "gpuTypePriority": "custom",
    "gpuCount": 1,
    "containerDiskInGb": 80,
    "ports": ["22/tcp"],
    "env": {"PUBLIC_KEY": PUBKEY, "HF_TOKEN": HF, "WANDB_API_KEY": WB},
    "dockerEntrypoint": ["/bin/bash", "-c", ENTRY],
}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "kill":
        pid = sys.argv[2]
        r = httpx.delete(f"{BASE}/pods/{pid}", headers=H, timeout=60)
        print("DELETE", pid, r.status_code, r.text[:200]); sys.exit()

    r = httpx.post(f"{BASE}/pods", headers=H, json=body, timeout=120)
    if r.status_code >= 300:
        print("CREATE FAILED", r.status_code, r.text); sys.exit(1)
    pod = r.json()
    pid = pod["id"]
    print("POD_ID", pid)
    for _ in range(60):
        time.sleep(10)
        g = httpx.get(f"{BASE}/pods/{pid}", headers=H, timeout=60).json()
        ip = g.get("publicIp") or ""
        ports = {p.get("privatePort"): p.get("publicPort")
                 for p in (g.get("portMappings") or []) if isinstance(p, dict)}
        # portMappings may be a dict {"22": port}
        if isinstance(g.get("portMappings"), dict):
            ports = {int(k): v for k, v in g["portMappings"].items()}
        sshp = ports.get(22)
        print(f"  status={g.get('desiredStatus')} ip={ip!r} ssh={sshp}", flush=True)
        if ip and sshp:
            print(f"SSH_READY root@{ip} -p {sshp}")
            Path(__file__).parent.joinpath("pod_info.json").write_text(
                json.dumps({"id": pid, "ip": ip, "ssh_port": sshp}))
            break
    else:
        print("TIMEOUT waiting for ssh")
