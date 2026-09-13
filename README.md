# RPCR: Role-Aware Progressive Chained Refinement for Multimodal Sentiment Analysis

This repository provides a lightweight research-preview implementation of **RPCR**
for multimodal sentiment analysis on CMU-MOSI, CMU-MOSEI, and CH-SIMS.

RPCR follows a role-aware progressive chain:

```text
Vision -> Audio -> Text
```

Vision acts as a nonverbal conditioner, audio forms an acoustic-centered
nonverbal carrier, and text provides the semantic anchor for final sentiment
prediction. Cross-modal evidence is injected progressively through adaptive
conditional refinement blocks.

> This is an early public code snapshot for paper review and project tracking.
> The final polished release may reorganize scripts, checkpoints, and pretrained
> feature links.

## Highlights

- **Role-aware chained reasoning**: organizes multimodal interaction as
  `V -> A -> T` rather than fully parallel fusion.
- **Progressive conditional refinement**: uses visual context to refine acoustic
  features, then uses visual-acoustic evidence to refine textual features.
- **Adaptive refinement composition**: combines dynamic sample-conditioned
  routing, static refinement bases, and feature-wise gated residual updates.
- **Feature-domain training**: supports utterance-level pretrained features for
  efficient experiments on CMU-MOSI, CMU-MOSEI, and CH-SIMS.
- **Server search scripts**: includes Slurm helpers for hyperparameter search,
  target checkpoint discovery, and result summarization.

## Repository Layout

```text
RPCR/
├── main_feature_v2a2t.py              # Main feature-domain RPCR training entry
├── main_chsims_unaligned_v2a2t.py     # CH-SIMS unaligned-feature baseline entry
├── models/
│   ├── feature_v2a2t.py               # RPCR model and refinement blocks
│   └── chsims_unaligned_v2a2t.py      # CH-SIMS sequence-feature model
├── utils/
│   ├── ebmc_feature_dataset.py        # Utterance-level feature cache loader
│   ├── chsims_unaligned_dataset.py    # CH-SIMS unaligned pickle loader
│   └── regression_metrics.py          # MOSI/MOSEI/CH-SIMS metrics
├── tools/
│   ├── build_ebmc_feature_cache.py    # Build compact .pt feature cache
│   ├── inspect_ebmc_features.py       # Inspect feature directories/cache files
│   └── run_feature_v2a2t.ps1          # Local Windows training helper
├── job/
│   ├── submit_feature_v2a2t.sh        # Generic Slurm training launcher
│   ├── submit_target_search.sh        # Slurm target-search launcher
│   ├── TARGET_SEARCH.md               # Search protocol notes
│   └── slurm/                         # Slurm array job templates
├── environment/
│   └── requirements_conda_mope_env.txt
└── assets/figures/                    # Paper figures used in this README
```

## Data

The main RPCR implementation expects a compact utterance-level feature cache:

```text
cache/ebmc_cmumosi_full.pt
cache/ebmc_cmumosei_full.pt
cache/ebmc_chsims_full.pt
```

Each sample contains:

| Modality | Feature type | Dim |
|---|---|---:|
| Vision | MANet utterance-level feature | 1024 |
| Audio | Wav2Vec utterance-level feature | 512 |
| Text | DeBERTa/MacBERT utterance-level feature | 1024 |

The raw datasets and feature files are not included in this repository. See
`DATASETS.md` for the expected layout and cache-building commands.

## Installation

```bash
git clone https://github.com/ma-tubu/RPCR.git
cd RPCR

conda create -n rpcr python=3.11 -y
conda activate rpcr
pip install -r requirements.txt
```

If you already have a project environment with PyTorch, NumPy, and
scikit-learn, you can use it directly.

## Build Feature Cache

Example for CMU-MOSEI:

```bash
python tools/build_ebmc_feature_cache.py \
  --dataset-root datasets/CMUMOSEI \
  --output cache/ebmc_cmumosei_full.pt \
  --dataset-name CMUMOSEI
```

Inspect the cache:

```bash
python tools/inspect_ebmc_features.py --cache-path cache/ebmc_cmumosei_full.pt
```

## Training

### CMU-MOSEI

```bash
python main_feature_v2a2t.py \
  --dataset-name CMUMOSEI \
  --cache-path cache/ebmc_cmumosei_full.pt \
  --output-dir train_log/cmumosei_seed42 \
  --epochs 60 \
  --batch-size 256 \
  --monitor MAE \
  --device cuda
```

### CMU-MOSI

```bash
python main_feature_v2a2t.py \
  --dataset-name CMUMOSI \
  --cache-path cache/ebmc_cmumosi_full.pt \
  --output-dir train_log/cmumosi_seed42 \
  --epochs 60 \
  --batch-size 64 \
  --monitor MAE \
  --device cuda
```

### CH-SIMS

For CH-SIMS, `Acc_5` is the recommended formal validation metric because it is
part of the reported metric set.

```bash
python main_feature_v2a2t.py \
  --dataset-name CHSIMS \
  --cache-path cache/ebmc_chsims_full.pt \
  --output-dir train_log/chsims_seed42 \
  --epochs 70 \
  --batch-size 24 \
  --monitor Acc_5 \
  --output-activation tanh \
  --enable-validation-acc5-checkpoint \
  --device cuda
```

## Checkpoint Selection

`best.pt` is selected **only by validation metrics** and should be used for
standard reporting. Per-epoch test-monitor checkpoints are intended for
diagnosis and follow-up verification.

Common checkpoint files:

| File | Selection source | Recommended use |
|---|---|---|
| `best.pt` | Validation monitor | Formal evaluation |
| `best_valid_acc5.pt` | Validation Acc-5 | CH-SIMS formal audit |
| `best_valid_acc7.pt` | Validation Acc-7 | MOSI/MOSEI audit |
| `best_test_acc5.pt`, `best_test_acc7.pt` | Per-epoch test monitor | Exploratory only |
| `*_hit_epoch_*.pt` | Per-epoch test target hit | Exploratory only |

## Server Search

The `job/` folder contains Slurm scripts used for large-scale searches.

```bash
cd /path/to/RPCR

DATASET=CMUMOSEI MAX_CONCURRENT=2 \
  bash job/submit_target_search.sh mosei_target_search_001 full

DATASET=CHSIMS MAX_CONCURRENT=4 \
  bash job/submit_target_search.sh chsims_target_search_acc5_001 full
```

Monitor:

```bash
DATASET=CMUMOSEI bash job/watch_target_search.sh mosei_target_search_001 30
DATASET=CHSIMS bash job/watch_target_search.sh chsims_target_search_acc5_001 30
```

## Notes

- The repository intentionally excludes raw datasets, pretrained feature caches,
  checkpoints, and training logs.
- Some scripts are research utilities from active experiments and may require
  path or cluster-specific adjustment.
- The code release is provided for transparency before a final cleaned version.
