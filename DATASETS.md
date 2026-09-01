# Dataset and Feature Cache Preparation

RPCR currently uses utterance-level pretrained feature vectors for efficient
training.

## Expected Feature Dimensions

| Dataset | Vision | Audio | Text | Label range |
|---|---:|---:|---:|---|
| CMU-MOSI | 1024 | 512 | 1024 | `[-3, 3]` |
| CMU-MOSEI | 1024 | 512 | 1024 | `[-3, 3]` |
| CH-SIMS | 1024 | 512 | 1024 | usually `[-1, 1]` |

The default feature names in `utils/ebmc_feature_dataset.py` are:

```text
vision: manet_UTT
audio : wav2vec-large-c-UTT
text  : deberta-large-4-UTT
text  : chinese-macbert-large-4-UTT for CH-SIMS
```

## Directory Layout

Place downloaded/pre-extracted features under `datasets/`:

```text
datasets/
├── CMUMOSI/
│   ├── CMUMOSI_features_raw_2way.pkl
│   ├── manet_UTT/
│   ├── wav2vec-large-c-UTT/
│   └── deberta-large-4-UTT/
├── CMUMOSEI/
│   ├── CMUMOSEI_features_raw_2way.pkl
│   ├── manet_UTT/
│   ├── wav2vec-large-c-UTT/
│   └── deberta-large-4-UTT/
└── CHSIMS/
    ├── CHSIMS_features_raw_2way.pkl
    ├── manet_UTT/
    ├── wav2vec-large-c-UTT/
    └── chinese-macbert-large-4-UTT/
```

Each feature folder should contain one `.npy` vector per utterance ID. The cache
builder checks feature dimensions and writes a compact PyTorch `.pt` cache.

## Build Caches

```bash
mkdir -p cache

python tools/build_ebmc_feature_cache.py \
  --dataset-root datasets/CMUMOSI \
  --output cache/ebmc_cmumosi_full.pt \
  --dataset-name CMUMOSI

python tools/build_ebmc_feature_cache.py \
  --dataset-root datasets/CMUMOSEI \
  --output cache/ebmc_cmumosei_full.pt \
  --dataset-name CMUMOSEI

python tools/build_ebmc_feature_cache.py \
  --dataset-root datasets/CHSIMS \
  --output cache/ebmc_chsims_full.pt \
  --dataset-name CHSIMS
```

## Inspect Caches

```bash
python tools/inspect_ebmc_features.py --cache-path cache/ebmc_cmumosei_full.pt
```

## What Is Not Included

This repository does not include:

- raw videos, audio, transcripts, or annotations;
- extracted `.npy` feature folders;
- compact `.pt` feature caches;
- trained checkpoints;
- large training logs.

Please obtain the datasets from their official sources and follow their licenses.
