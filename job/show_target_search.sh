#!/bin/bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-CHSIMS}"
export DATASET
source "$JOB_DIR/server.env"
[[ $# -eq 1 ]] || { echo "Usage: DATASET=CHSIMS|CMUMOSEI bash job/show_target_search.sh EXP_NAME" >&2; exit 2; }

RUN_DIR="$RUN_ROOT/$1"
[[ -d "$RUN_DIR" ]] || { echo "Experiment not found: $RUN_DIR" >&2; exit 3; }
if [[ -f "$RUN_DIR/submission.env" ]]; then
  source "$RUN_DIR/submission.env"
fi
source "$JOB_DIR/setup_env.sh"

echo "===== V2A2T Target Search ====="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "Dataset: $DATASET"
echo "Target specs: ${TARGET_SPECS:-}"
echo "Run directory: $RUN_DIR"
echo
squeue -u "$USER" || true
echo
python "$JOB_DIR/tools/summarize_target_search.py" --run-dir "$RUN_DIR"
