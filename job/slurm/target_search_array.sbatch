#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

JOB_DIR="${CODE_DIR:?CODE_DIR is required}/job"
source "$RUN_DIR/submission.env"
source "$JOB_DIR/setup_env.sh"

[[ -f "$MANIFEST" ]] || { echo "Missing manifest: $MANIFEST" >&2; exit 3; }
target_line=$((SLURM_ARRAY_TASK_ID + 2))
manifest_row="$(awk -F '\t' -v target="$target_line" 'NR == target {print $0}' "$MANIFEST")"
[[ -n "$manifest_row" ]] || {
  echo "No manifest row for array index $SLURM_ARRAY_TASK_ID." >&2
  exit 3
}

IFS=$'\t' read -r \
  STAGE VARIANT SEED \
  EPOCHS BATCH_SIZE HIDDEN_DIM VISUAL_DEPTH AUDIO_DEPTH TEXT_DEPTH \
  NUM_EXPERTS EXPERT_DIM EXPANSION DROPOUT LEARNING_RATE WEIGHT_DECAY \
  WARMUP_RATIO GRAD_CLIP AUXILIARY_WEIGHT CORRELATION_WEIGHT ROUTER_WEIGHT \
  PATIENCE MONITOR OUTPUT_ACTIVATION \
  AUDIO_ENCODER_MODE TEXT_ENCODER_MODE \
  AUDIO_MODULATION_MODE TEXT_MODULATION_MODE \
  AUDIO_CONDITION_SCALE_INIT TEXT_CONDITION_SCALE_INIT \
  AUDIO_CONDITION_SCALE_GROWTH TEXT_CONDITION_SCALE_GROWTH \
  <<< "$manifest_row"

OUTPUT_DIR="$RUN_DIR/$STAGE/$VARIANT/seed_$SEED"
mkdir -p "$OUTPUT_DIR"

echo "===== V2A2T Target Search Trial ====="
echo "Time: $(date --iso-8601=seconds)"
echo "Experiment: $EXP_NAME"
echo "Dataset: $DATASET"
echo "Profile: $PROFILE"
echo "Stage: $STAGE"
echo "Variant: $VARIANT"
echo "Seed: $SEED"
echo "Monitor: $MONITOR"
echo "Target specs: $TARGET_SPECS"
echo "Output: $OUTPUT_DIR"
nvidia-smi

if [[ -f "$OUTPUT_DIR/final_result.json" ]]; then
  echo "Trial already completed; skipping: $OUTPUT_DIR"
  exit 0
fi

EFFECTIVE_EPOCHS="$EPOCHS"
if [[ "$PROFILE" == "pilot" ]]; then
  EFFECTIVE_EPOCHS=2
fi

COMMAND=(
  python -u "$CODE_DIR/main_feature_v2a2t.py"
  --dataset-name "$DATASET"
  --cache-path "$FEATURE_CACHE"
  --output-dir "$OUTPUT_DIR"
  --epochs "$EFFECTIVE_EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --device cuda
  --precision "$PRECISION"
  --output-activation "$OUTPUT_ACTIVATION"
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
  --patience "$PATIENCE"
  --monitor "$MONITOR"
  --audio-encoder-mode "$AUDIO_ENCODER_MODE"
  --text-encoder-mode "$TEXT_ENCODER_MODE"
  --audio-modulation-mode "$AUDIO_MODULATION_MODE"
  --text-modulation-mode "$TEXT_MODULATION_MODE"
  --audio-condition-scale-init "$AUDIO_CONDITION_SCALE_INIT"
  --text-condition-scale-init "$TEXT_CONDITION_SCALE_INIT"
  --audio-condition-scale-growth "$AUDIO_CONDITION_SCALE_GROWTH"
  --text-condition-scale-growth "$TEXT_CONDITION_SCALE_GROWTH"
  --eval-test-every-epoch
  --enable-test-target-checkpoints
  --disable-early-stopping
  --target-specs "$TARGET_SPECS"
)

if [[ "${ENABLE_VALIDATION_ACC7_CHECKPOINT:-0}" == "1" ]]; then
  COMMAND+=(--enable-validation-acc7-checkpoint)
fi
if [[ "${ENABLE_VALIDATION_ACC5_CHECKPOINT:-0}" == "1" ]]; then
  COMMAND+=(--enable-validation-acc5-checkpoint)
fi

if [[ "$PROFILE" == "pilot" ]]; then
  COMMAND+=(
    --max-train-batches "$PILOT_MAX_TRAIN_BATCHES"
    --max-eval-batches "$PILOT_MAX_EVAL_BATCHES"
  )
fi
if [[ -f "$OUTPUT_DIR/last.pt" ]]; then
  echo "Resuming from $OUTPUT_DIR/last.pt"
  COMMAND+=(--resume "$OUTPUT_DIR/last.pt")
fi

printf '%q ' "${COMMAND[@]}" > "$OUTPUT_DIR/command.txt"
printf '\n' >> "$OUTPUT_DIR/command.txt"
"${COMMAND[@]}"

echo "Completed: $DATASET $STAGE/$VARIANT/seed_$SEED at $(date --iso-8601=seconds)"
