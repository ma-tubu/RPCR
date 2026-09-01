#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail
source "${CODE_DIR:?CODE_DIR is required}/job/setup_env.sh"
python "$CODE_DIR/job/tools/summarize_target_search.py" --run-dir "$RUN_DIR"
