#!/bin/bash
set -euo pipefail
if [[ -z "${RUNPOD_POD_ID:-}" ]]; then
  echo 'Refusing to assemble weights outside a RunPod Pod' >&2
  exit 1
fi
python3 /opt/wan-image/image_weights.py assemble
exec /start.sh "$@"
