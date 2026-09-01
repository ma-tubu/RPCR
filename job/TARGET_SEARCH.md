# V2A2T Target Search

This server workflow runs validation-selected training while also monitoring the
test split after every epoch for exploratory target discovery.

## Formal Selection Protocol

- CHSIMS formal checkpoint: validation `Acc_5` (`best.pt` when `monitor=Acc_5`,
  plus `best_valid_acc5.pt` for explicit audit).
- CMU-MOSEI formal checkpoint: validation `MAE` by default (`best.pt`), with
  optional validation `Acc_7` audit checkpoint.
- Per-epoch test checkpoints such as `best_test_acc5.pt`, `best_test_acc7.pt`,
  and `*_hit_epoch_*.pt` are diagnostic/exploratory only.
- Do not use CHSIMS validation `Acc_3` as the main checkpoint selector unless
  `Acc_3` is explicitly reported in the paper.

## Targets

CHSIMS hard target:

```text
Acc_5 >= 0.4551
Acc_2 >= 0.8446
F1    >= 0.8448
MAE   <= 0.387
Corr  >= 0.755
```

CMU-MOSEI target tiers. A checkpoint is saved when any tier is satisfied:

```text
strong: Acc_7 >= 0.541, Acc_2 >= 0.865, F1 >= 0.865, MAE <= 0.526, Corr >= 0.772
medium: Acc_7 >= 0.541, Acc_2 >= 0.864, F1 >= 0.865, MAE <= 0.529, Corr >= 0.787
weak:   Acc_7 >= 0.541, Acc_2 >= 0.863, F1 >= 0.862, MAE <= 0.536, Corr >= 0.787
```

## Submit

Run a CHSIMS pilot:

```bash
cd /path/to/RPCR
DATASET=CHSIMS MAX_CONCURRENT=4 bash job/submit_target_search.sh chsims_target_pilot pilot
```

Run a CHSIMS full search:

```bash
DATASET=CHSIMS MAX_CONCURRENT=4 bash job/submit_target_search.sh chsims_target_search_001 full
```

Run a CMU-MOSEI pilot:

```bash
DATASET=CMUMOSEI MAX_CONCURRENT=2 bash job/submit_target_search.sh mosei_target_pilot pilot
```

Run a CMU-MOSEI full search:

```bash
DATASET=CMUMOSEI MAX_CONCURRENT=2 bash job/submit_target_search.sh mosei_target_search_001 full
```

## Monitor

```bash
DATASET=CHSIMS bash job/show_target_search.sh chsims_target_search_001
DATASET=CHSIMS bash job/watch_target_search.sh chsims_target_search_001 30
```

```bash
DATASET=CMUMOSEI bash job/show_target_search.sh mosei_target_search_001
DATASET=CMUMOSEI bash job/watch_target_search.sh mosei_target_search_001 30
```

## Outputs

Each run directory contains:

- `target_search_trials.csv`
- `target_search_summary.csv`
- `target_search_epoch_trajectories.csv`
- `target_search_target_hit_trials.csv`
- `target_search_target_hit_epochs.csv`
- `target_search_summary.json`

Target-hit checkpoints are saved inside each trial directory using names such as
`hard_hit_epoch_018.pt`, `strong_hit_epoch_013.pt`, and
`weak_hit_best_acc7.pt` for MOSEI or `hard_hit_best_acc5.pt` for CHSIMS.

Formal CHSIMS runs also save `best_valid_acc5.pt` and record
`validation_acc5_selection` in `final_result.json`.

These target-hit checkpoints are exploratory test-monitor checkpoints. Use them
for discovery and follow-up verification, not as validation-selected formal
checkpoints without a clean selection protocol.
