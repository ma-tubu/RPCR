#!/bin/bash
set -euo pipefail

JOB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$JOB_DIR/server.env"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash job/submit_feature_v2a2t.sh EXP_NAME [pilot|full]" >&2
  exit 2
fi

EXP_NAME="$1"
PROFILE="${2:-pilot}"

if [[ ! "$EXP_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "EXP_NAME may contain only letters, digits, dot, underscore and hyphen." >&2
  exit 2
fi
if [[ "$PROFILE" != "pilot" && "$PROFILE" != "full" ]]; then
  echo "PROFILE must be pilot or full, got: $PROFILE" >&2
  exit 2
fi

if [[ "$PROFILE" == "pilot" ]]; then
  SEEDS="${SEEDS:-42}"
  EPOCHS="${EPOCHS:-2}"
  HIDDEN_DIM="${HIDDEN_DIM:-64}"
  if [[ "$DATASET" == "CHSIMS" ]]; then
    NUM_EXPERTS="${NUM_EXPERTS:-2}"
    EXPERT_DIM="${EXPERT_DIM:-32}"
  else
    NUM_EXPERTS="${NUM_EXPERTS:-4}"
    EXPERT_DIM="${EXPERT_DIM:-0}"
  fi
  EXPANSION="${EXPANSION:-2}"
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-20}"
  MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-10}"
else
  SEEDS="${SEEDS:-42,3407,2024}"
  if [[ "$DATASET" == "CMUMOSI" ]]; then
    EPOCHS="${EPOCHS:-150}"
    HIDDEN_DIM="${HIDDEN_DIM:-256}"
    NUM_EXPERTS="${NUM_EXPERTS:-4}"
    EXPANSION="${EXPANSION:-2}"
    EXPERT_DIM="${EXPERT_DIM:-0}"
  elif [[ "$DATASET" == "CHSIMS" ]]; then
    EPOCHS="${EPOCHS:-60}"
    HIDDEN_DIM="${HIDDEN_DIM:-64}"
    NUM_EXPERTS="${NUM_EXPERTS:-2}"
    EXPERT_DIM="${EXPERT_DIM:-32}"
    EXPANSION="${EXPANSION:-2}"
  else
    EPOCHS="${EPOCHS:-60}"
    HIDDEN_DIM="${HIDDEN_DIM:-512}"
    NUM_EXPERTS="${NUM_EXPERTS:-8}"
    EXPANSION="${EXPANSION:-4}"
    EXPERT_DIM="${EXPERT_DIM:-0}"
  fi
  MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
  MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
fi

NUM_WORKERS="${NUM_WORKERS:-$SLURM_CPUS}"
PRECISION="${PRECISION:-fp16}"
if [[ "$DATASET" == "CHSIMS" ]]; then
  VISUAL_DEPTH="${VISUAL_DEPTH:-2}"
  AUDIO_DEPTH="${AUDIO_DEPTH:-3}"
  TEXT_DEPTH="${TEXT_DEPTH:-3}"
else
  VISUAL_DEPTH="${VISUAL_DEPTH:-3}"
  AUDIO_DEPTH="${AUDIO_DEPTH:-4}"
  TEXT_DEPTH="${TEXT_DEPTH:-4}"
fi
if [[ "$DATASET" == "CMUMOSI" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-64}"
  DROPOUT="${DROPOUT:-0.35}"
  LEARNING_RATE="${LEARNING_RATE:-0.0002}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0005}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
  PATIENCE="${PATIENCE:-25}"
elif [[ "$DATASET" == "CHSIMS" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-24}"
  DROPOUT="${DROPOUT:-0.35}"
  LEARNING_RATE="${LEARNING_RATE:-0.0002}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.002}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
  PATIENCE="${PATIENCE:-10}"
else
  BATCH_SIZE="${BATCH_SIZE:-256}"
  DROPOUT="${DROPOUT:-0.2}"
  LEARNING_RATE="${LEARNING_RATE:-0.0003}"
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
  WARMUP_RATIO="${WARMUP_RATIO:-0.08}"
  PATIENCE="${PATIENCE:-12}"
fi
if [[ "$DATASET" == "CHSIMS" ]]; then
  GRAD_CLIP="${GRAD_CLIP:-0.5}"
  AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.03}"
  CORRELATION_WEIGHT="${CORRELATION_WEIGHT:-0.05}"
  ROUTER_WEIGHT="${ROUTER_WEIGHT:-0.0}"
  AUDIO_CONDITION_SCALE_INIT="${AUDIO_CONDITION_SCALE_INIT:-0.03}"
  TEXT_CONDITION_SCALE_INIT="${TEXT_CONDITION_SCALE_INIT:-0.03}"
  MONITOR="${MONITOR:-Acc_3}"
  EVAL_TEST_EVERY_EPOCH="${EVAL_TEST_EVERY_EPOCH:-1}"
else
  GRAD_CLIP="${GRAD_CLIP:-1.0}"
  AUXILIARY_WEIGHT="${AUXILIARY_WEIGHT:-0.1}"
  CORRELATION_WEIGHT="${CORRELATION_WEIGHT:-0.05}"
  ROUTER_WEIGHT="${ROUTER_WEIGHT:-0.01}"
  AUDIO_CONDITION_SCALE_INIT="${AUDIO_CONDITION_SCALE_INIT:-0.01}"
  TEXT_CONDITION_SCALE_INIT="${TEXT_CONDITION_SCALE_INIT:-0.01}"
  MONITOR="${MONITOR:-MAE}"
  EVAL_TEST_EVERY_EPOCH="${EVAL_TEST_EVERY_EPOCH:-0}"
fi
if [[ "$EVAL_TEST_EVERY_EPOCH" != "0" && "$EVAL_TEST_EVERY_EPOCH" != "1" ]]; then
  echo "EVAL_TEST_EVERY_EPOCH must be 0 or 1." >&2
  exit 2
fi

RUN_DIR="$RUN_ROOT/$EXP_NAME"
MANIFEST="$RUN_DIR/manifest.tsv"
CONFIG_PATH="$RUN_DIR/submission.env"
mkdir -p "$RUN_ROOT/slurm" "$RUN_DIR"

if [[ -f "$MANIFEST" && "${RESUBMIT:-0}" != "1" ]]; then
  echo "Experiment already exists: $RUN_DIR" >&2
  echo "Use a new EXP_NAME, or set RESUBMIT=1 to submit the same manifest again." >&2
  exit 3
fi

echo "===== Server preflight ====="
bash "$JOB_DIR/check_server.sh"
command -v sbatch >/dev/null 2>&1 || { echo "sbatch is required." >&2; exit 4; }

IFS=',' read -r -a SEED_ARRAY <<< "$SEEDS"
if [[ ${#SEED_ARRAY[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one integer." >&2
  exit 2
fi

printf 'trial_id\tseed\n' > "$MANIFEST"
for raw_seed in "${SEED_ARRAY[@]}"; do
  seed="${raw_seed//[[:space:]]/}"
  [[ "$seed" =~ ^[0-9]+$ ]] || { echo "Invalid seed: $raw_seed" >&2; exit 2; }
  printf 'seed_%s\t%s\n' "$seed" "$seed" >> "$MANIFEST"
done

cat > "$CONFIG_PATH" <<EOF
EXP_NAME=$EXP_NAME
PROFILE=$PROFILE
DATASET=$DATASET
SEEDS=$SEEDS
FEATURE_CACHE=$FEATURE_CACHE
EPOCHS=$EPOCHS
BATCH_SIZE=$BATCH_SIZE
NUM_WORKERS=$NUM_WORKERS
PRECISION=$PRECISION
HIDDEN_DIM=$HIDDEN_DIM
VISUAL_DEPTH=$VISUAL_DEPTH
AUDIO_DEPTH=$AUDIO_DEPTH
TEXT_DEPTH=$TEXT_DEPTH
NUM_EXPERTS=$NUM_EXPERTS
EXPERT_DIM=$EXPERT_DIM
EXPANSION=$EXPANSION
DROPOUT=$DROPOUT
LEARNING_RATE=$LEARNING_RATE
WEIGHT_DECAY=$WEIGHT_DECAY
WARMUP_RATIO=$WARMUP_RATIO
GRAD_CLIP=$GRAD_CLIP
AUXILIARY_WEIGHT=$AUXILIARY_WEIGHT
CORRELATION_WEIGHT=$CORRELATION_WEIGHT
ROUTER_WEIGHT=$ROUTER_WEIGHT
AUDIO_CONDITION_SCALE_INIT=$AUDIO_CONDITION_SCALE_INIT
TEXT_CONDITION_SCALE_INIT=$TEXT_CONDITION_SCALE_INIT
MONITOR=$MONITOR
PATIENCE=$PATIENCE
EVAL_TEST_EVERY_EPOCH=$EVAL_TEST_EVERY_EPOCH
MAX_TRAIN_BATCHES=$MAX_TRAIN_BATCHES
MAX_EVAL_BATCHES=$MAX_EVAL_BATCHES
EOF

EXPORTS="ALL,EXP_NAME=$EXP_NAME,PROFILE=$PROFILE,RUN_DIR=$RUN_DIR,MANIFEST=$MANIFEST"

SBATCH_ARGS=(
  --parsable
  --time="$SLURM_TIME"
  --nodes=1
  --ntasks=1
  --cpus-per-task="$SLURM_CPUS"
  --gpus-per-node="$SLURM_GPUS"
  --array="0-$((${#SEED_ARRAY[@]} - 1))%$MAX_CONCURRENT"
  --job-name="v2a2t-$EXP_NAME"
  --output="$RUN_ROOT/slurm/%x-%A_%a.out"
  --export="$EXPORTS"
)
if [[ -n "$SLURM_ACCOUNT" ]]; then
  SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
fi
if [[ -n "$SLURM_PARTITION" ]]; then
  SBATCH_ARGS+=(--partition="$SLURM_PARTITION")
fi

if ! raw_job_id="$(sbatch "${SBATCH_ARGS[@]}" "$JOB_DIR/slurm/train_array.sbatch")"; then
  echo "Failed to submit V2A2T training array." >&2
  exit 5
fi
job_id="${raw_job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $raw_job_id" >&2; exit 5; }

cat > "$RUN_DIR/slurm_job.json" <<EOF
{
  "experiment": "$EXP_NAME",
  "profile": "$PROFILE",
  "dataset": "$DATASET",
  "job_id": "$job_id",
  "array_size": ${#SEED_ARRAY[@]},
  "manifest": "$MANIFEST",
  "run_dir": "$RUN_DIR"
}
EOF

echo "Submitted V2A2T experiment: $EXP_NAME"
echo "Profile: $PROFILE"
echo "Dataset: $DATASET"
echo "Seeds: $SEEDS"
echo "Slurm job: $job_id"
echo "Run directory: $RUN_DIR"
echo "Monitor: bash job/watch_experiment.sh $EXP_NAME"
