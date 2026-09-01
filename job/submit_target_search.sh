#!/bin/bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${DATASET:-CHSIMS}"
export DATASET
source "$JOB_DIR/server.env"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: DATASET=CHSIMS|CMUMOSEI bash job/submit_target_search.sh EXP_NAME [pilot|full]" >&2
  exit 2
fi

EXP_NAME="$1"
PROFILE="${2:-pilot}"
[[ "$EXP_NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "EXP_NAME may contain only letters, digits, dot, underscore and hyphen." >&2
  exit 2
}
[[ "$PROFILE" == "pilot" || "$PROFILE" == "full" ]] || {
  echo "PROFILE must be pilot or full." >&2
  exit 2
}
[[ "$DATASET" == "CHSIMS" || "$DATASET" == "CMUMOSEI" ]] || {
  echo "DATASET must be CHSIMS or CMUMOSEI for target search." >&2
  exit 2
}

if [[ "$DATASET" == "CHSIMS" ]]; then
  DEFAULT_SEEDS="42,3407,2024,1,7"
  DEFAULT_TARGET_SPECS="hard:Acc_5>=0.4551,Acc_2>=0.8446,F1>=0.8448,MAE<=0.387,Corr>=0.755"
  DEFAULT_TIME="08:00:00"
  DEFAULT_ENABLE_VALIDATION_ACC5_CHECKPOINT=1
  DEFAULT_ENABLE_VALIDATION_ACC7_CHECKPOINT=0
else
  DEFAULT_SEEDS="42,3407,2024"
  DEFAULT_TARGET_SPECS="strong:Acc_7>=0.541,Acc_2>=0.865,F1>=0.865,MAE<=0.526,Corr>=0.772;medium:Acc_7>=0.541,Acc_2>=0.864,F1>=0.865,MAE<=0.529,Corr>=0.787;weak:Acc_7>=0.541,Acc_2>=0.863,F1>=0.862,MAE<=0.536,Corr>=0.787"
  DEFAULT_TIME="12:00:00"
  DEFAULT_ENABLE_VALIDATION_ACC5_CHECKPOINT=0
  DEFAULT_ENABLE_VALIDATION_ACC7_CHECKPOINT=1
fi

SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SEARCH_VARIANTS="${SEARCH_VARIANTS:-}"
MANIFEST_TOOL="${TARGET_SEARCH_MANIFEST_TOOL:-$JOB_DIR/tools/create_target_search_manifest.py}"
MANIFEST_BASENAME="${TARGET_SEARCH_MANIFEST_BASENAME:-${DATASET_SLUG}_target_search.tsv}"
TARGET_SPECS="${TARGET_SPECS:-$DEFAULT_TARGET_SPECS}"
NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS}"
PRECISION="${PRECISION:-fp16}"
ENABLE_VALIDATION_ACC5_CHECKPOINT="${ENABLE_VALIDATION_ACC5_CHECKPOINT:-$DEFAULT_ENABLE_VALIDATION_ACC5_CHECKPOINT}"
ENABLE_VALIDATION_ACC7_CHECKPOINT="${ENABLE_VALIDATION_ACC7_CHECKPOINT:-$DEFAULT_ENABLE_VALIDATION_ACC7_CHECKPOINT}"
PILOT_MAX_TRAIN_BATCHES="${PILOT_MAX_TRAIN_BATCHES:-5}"
PILOT_MAX_EVAL_BATCHES="${PILOT_MAX_EVAL_BATCHES:-3}"
SLURM_TIME="${SLURM_TIME:-$DEFAULT_TIME}"

RUN_DIR="$RUN_ROOT/$EXP_NAME"
CONTROL_DIR="$RUN_DIR/control"
MANIFEST_DIR="$CONTROL_DIR/manifests"
MANIFEST="$MANIFEST_DIR/$MANIFEST_BASENAME"
mkdir -p "$RUN_ROOT/slurm" "$MANIFEST_DIR"
if [[ -f "$CONTROL_DIR/slurm_jobs.json" && "${RESUBMIT:-0}" != "1" ]]; then
  echo "Target search already exists: $RUN_DIR" >&2
  echo "Use a new EXP_NAME, or set RESUBMIT=1 to resubmit incomplete trials." >&2
  exit 3
fi

echo "===== Server preflight ====="
bash "$JOB_DIR/check_server.sh"
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is required." >&2; exit 4; }

[[ -f "$MANIFEST_TOOL" ]] || { echo "Manifest tool not found: $MANIFEST_TOOL" >&2; exit 4; }
MANIFEST_JSON="$(python "$MANIFEST_TOOL" \
  --dataset "$DATASET" --output "$MANIFEST" --seeds "$SEEDS" --variants "$SEARCH_VARIANTS")"
TRIAL_COUNT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["total"])' "$MANIFEST_JSON")"

cat > "$RUN_DIR/submission.env" <<EOF
EXP_NAME=$EXP_NAME
PROFILE=$PROFILE
DATASET=$DATASET
FEATURE_CACHE=$FEATURE_CACHE
SEEDS=$SEEDS
SEARCH_VARIANTS=$SEARCH_VARIANTS
MANIFEST_TOOL=$MANIFEST_TOOL
MANIFEST_BASENAME=$MANIFEST_BASENAME
TARGET_SPECS='$TARGET_SPECS'
NUM_WORKERS=$NUM_WORKERS
PRECISION=$PRECISION
ENABLE_VALIDATION_ACC5_CHECKPOINT=$ENABLE_VALIDATION_ACC5_CHECKPOINT
ENABLE_VALIDATION_ACC7_CHECKPOINT=$ENABLE_VALIDATION_ACC7_CHECKPOINT
PILOT_MAX_TRAIN_BATCHES=$PILOT_MAX_TRAIN_BATCHES
PILOT_MAX_EVAL_BATCHES=$PILOT_MAX_EVAL_BATCHES
EOF

COMMON_EXPORTS="ALL,EXP_NAME=$EXP_NAME,PROFILE=$PROFILE,RUN_DIR=$RUN_DIR"
SBATCH_ARGS=(
  --parsable
  --time="$SLURM_TIME"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$SLURM_CPUS"
  --gpus-per-node="$SLURM_GPUS"
  --array="0-$((TRIAL_COUNT - 1))%$MAX_CONCURRENT"
  --job-name="v2a2t-${DATASET_SLUG}-target"
  --output="$RUN_ROOT/slurm/%x-%A_%a.out"
  --export="$COMMON_EXPORTS,MANIFEST=$MANIFEST"
)
if [[ -n "$SLURM_ACCOUNT" ]]; then
  SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "$SLURM_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$SLURM_PARTITION")
fi

echo "Submitting target search: dataset=$DATASET trials=$TRIAL_COUNT max_concurrent=$MAX_CONCURRENT"
RAW_SEARCH_JOB="$(sbatch "${SBATCH_ARGS[@]}" "$JOB_DIR/slurm/target_search_array.sbatch")" || {
  echo "Failed to submit target search array." >&2
  exit 5
}
SEARCH_JOB="${RAW_SEARCH_JOB%%;*}"
[[ "$SEARCH_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $RAW_SEARCH_JOB" >&2; exit 5; }

SUMMARY_ARGS=(
  --parsable
  --dependency="afterany:$SEARCH_JOB"
  --time=00:10:00
  --nodes=1
  --ntasks=1
  --cpus-per-task=1
  --gpus-per-node=1
  --job-name="v2a2t-${DATASET_SLUG}-target-summary"
  --output="$RUN_ROOT/slurm/%x-%j.out"
  --export="$COMMON_EXPORTS"
)
if [[ -n "$SLURM_ACCOUNT" ]]; then
  SUMMARY_ARGS+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "$SLURM_PARTITION" ]]; then
  SUMMARY_ARGS+=(--partition="$SLURM_PARTITION")
fi
RAW_SUMMARY_JOB="$(sbatch "${SUMMARY_ARGS[@]}" "$JOB_DIR/slurm/summarize_target_search.sbatch")" || {
  echo "Failed to submit target search summary job." >&2
  exit 5
}
SUMMARY_JOB="${RAW_SUMMARY_JOB%%;*}"
[[ "$SUMMARY_JOB" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $RAW_SUMMARY_JOB" >&2; exit 5; }

python - "$CONTROL_DIR/slurm_jobs.json" "$MANIFEST_JSON" <<PY
import json
import sys

payload = {
    "experiment": "$EXP_NAME",
    "profile": "$PROFILE",
    "dataset": "$DATASET",
    "seeds": "$SEEDS",
    "search_variants": "$SEARCH_VARIANTS",
    "manifest_tool": "$MANIFEST_TOOL",
    "manifest_basename": "$MANIFEST_BASENAME",
    "max_concurrent": int("$MAX_CONCURRENT"),
    "target_specs": "$TARGET_SPECS",
    "manifest": json.loads(sys.argv[2]),
    "search_array": "$SEARCH_JOB",
    "summary": "$SUMMARY_JOB",
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(payload, indent=2))
PY

echo "Submitted target search: $EXP_NAME"
echo "Dataset: $DATASET"
echo "Search array: $SEARCH_JOB"
echo "Summary job: $SUMMARY_JOB"
echo "Run directory: $RUN_DIR"
echo "Monitor: DATASET=$DATASET bash job/watch_target_search.sh $EXP_NAME 30"
