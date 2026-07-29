#!/usr/bin/env bash
# Deploy nla-agent.html to https://syvb.ca/nla-agent.html (Neocities).
# Reads the API key from ~/.neocities_key at run time — the key must never
# appear in the repo or in the published file.
set -euo pipefail
cd "$(dirname "$0")"
[ -f nla-agent-cache.json.gz ] || python3 build_cache.py
KEY=$(tr -d '\n' < ~/.neocities_key)
curl -sf -H "Authorization: Bearer $KEY" \
  -F "nla-agent.html=@nla-agent.html" \
  -F "nla-agent-cache.json.gz=@nla-agent-cache.json.gz" \
  https://neocities.org/api/upload
echo
echo "deployed → https://syvb.ca/nla-agent.html"
