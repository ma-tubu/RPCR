#!/bin/bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$JOB_DIR/server.env"

if [[ -f "$SERVER_ACTIVATE_SH" ]]; then
  source "$SERVER_ACTIVATE_SH"
fi

if type module >/dev/null 2>&1; then
  module reset
  module load "$PYTHON_MODULE"
fi

if [[ -f "$MOPE_VENV/bin/activate" ]]; then
  source "$MOPE_VENV/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "Cannot find virtualenv $MOPE_VENV or conda environment $CONDA_ENV." >&2
  exit 2
fi

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$CACHE_ROOT/tmp}"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$TMPDIR"

cd "$CODE_DIR"
echo "Python: $(command -v python)"
python --version
echo "CODE_DIR=$CODE_DIR"
echo "FEATURE_CACHE=$FEATURE_CACHE"
echo "CHSIMS_PKL=$CHSIMS_PKL"
echo "RUN_ROOT=$RUN_ROOT"
