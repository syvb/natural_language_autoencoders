# Sourced by every bash script here. Loads NLA_RUN_CONFIG (default:
# config.env next to the sourcing script) with SETDEFAULT semantics —
# values already exported by the caller win. Mirrors _config.py.
_CFG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLA_RUN_CONFIG="${NLA_RUN_CONFIG:-$_CFG_DIR/config.env}"
[ -f "$NLA_RUN_CONFIG" ] || { echo "config file not found: $NLA_RUN_CONFIG" >&2; exit 1; }
while IFS='=' read -r _k _v; do
    case "$_k" in ''|\#*) continue ;; esac
    if [ -z "${!_k+x}" ]; then export "$_k=$_v"; fi
done < "$NLA_RUN_CONFIG"
export NLA_RUN_CONFIG
echo "[config] $NLA_RUN_CONFIG (model=$BASE_MODEL layer=$LAYER_INDEX)" >&2
