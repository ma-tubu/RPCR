#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

JOB_DIR="${CODE_DIR:?CODE_DIR is required}/job"
source "$JOB_DIR/setup_env.sh"
[[ -f "$RUN_DIR/submission.env" ]] || {
  echo "Missing submission configuration: $RUN_DIR/submission.env" >&2
  exit 3
}
source "$RUN_DIR/submission.env"

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 3; }
target_line=$((SLURM_ARRAY_TASK_ID + 2))
manifest_row="$(awk -F '\t' -v target="$target_line" 'NR == target {print $1 "\t" $2}' "$MANIFEST")"
[[ -n "$manifest_row" ]] || {
  echo "No manifest row for array index $SLURM_ARRAY_TASK_ID." >&2
  exit 3
}
IFS=$'\t' read -r TRIAL_ID SEED <<< "$manifest_row"

OUTPUT_DIR="$RUN_DIR/$TRIAL_ID"
mkdir -p "$OUTPUT_DIR"

echo "===== V2A2T Slurm Trial ====="
echo "Host: $(hostname)"
echo "Time: $(date --iso-8601=seconds)"
echo "Experiment: $EXP_NAME"
echo "Profile: $PROFILE"
echo "Trial: $TRIAL_ID"
echo "Seed: $SEED"
echo "Output: $OUTPUT_DIR"
nvidia-smi

if [[ -f "$OUTPUT_DIR/final_result.json" ]]; then
  echo "Trial already completed; skipping: $OUTPUT_DIR"
  exit 0
fi

COMMAND=(
  python -u "$CODE_DIR/main_feature_v2a2t.py"
  --dataset-name "$DATASET"
  --cache-path "$FEATURE_CACHE"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
  --precision "$PRECISION"
  --hidden-dim "$HIDDEN_DIM"
  --visual-depth "$VISUAL_DEPTH"
  --audio-depth "$AUDIO_DEPTH"
  --text-depth "$TEXT_DEPTH"
  --num-experts "$NUM_EXPERTS"
  --expert-dim "$EXPERT_DIM"
  --expansion "$EXPANSION"
  --dropout "$DROPOUT"
  --learning-rate "$LEARNING_RATE"
  --weight-decay "$WEIGHT_DECAY"
  --warmup-ratio "$WARMUP_RATIO"
  --grad-clip "$GRAD_CLIP"
  --auxiliary-weight "$AUXILIARY_WEIGHT"
  --correlation-weight "$CORRELATION_WEIGHT"
  --router-weight "$ROUTER_WEIGHT"
  --audio-condition-scale-init "$AUDIO_CONDITION_SCALE_INIT"
  --text-condition-scale-init "$TEXT_CONDITION_SCALE_INIT"
  --monitor "$MONITOR"
  --patience "$PATIENCE"
)

if [[ -n "${MAX_TRAIN_BATCHES:-}" ]]; then
  COMMAND+=(--max-train-batches "$MAX_TRAIN_BATCHES")
fi
if [[ -n "${MAX_EVAL_BATCHES:-}" ]]; then
  COMMAND+=(--max-eval-batches "$MAX_EVAL_BATCHES")
fi
if [[ "${EVAL_TEST_EVERY_EPOCH:-0}" == "1" ]]; then
  COMMAND+=(--eval-test-every-epoch)
fi
if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pt"
  COMMAND+=(--resume "$OUTPUT_DIR/last.pt")
fi

printf '%q ' "${COMMAND[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
"${COMMAND[@]}"

echo "Completed: $TRIAL_ID at $(date --iso-8601=seconds)"
