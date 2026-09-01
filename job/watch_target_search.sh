#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: DATASET=CHSIMS|CMUMOSEI bash job/watch_target_search.sh EXP_NAME [seconds]" >&2
  exit 2
fi

EXP_NAME="$1"
INTERVAL="${2:-30}"
while true; do
  clear
  bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/show_target_search.sh" "$EXP_NAME"
  echo
  echo "Refresh every ${INTERVAL}s. Press Ctrl-C to stop."
  sleep "$INTERVAL"
done
