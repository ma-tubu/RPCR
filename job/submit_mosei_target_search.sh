#!/bin/bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=CMUMOSEI
exec bash "$JOB_DIR/submit_target_search.sh" "$@"
