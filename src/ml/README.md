# Machine Learning Scripts (Classical Models)

Script module that mirrors the deep-learning CLI workflow for expanding-window backtests using classical ML models.

## Run location

Run from this directory:

```bash
cd src/ml
python main.py --help
```

## Supported models

- decision-tree
- random-forest
- gradient-boosting
- linear-regression

Default --models runs all four.

## Strategy

- --strategy hourly
  - trains 24 separate models for each fold and model (one per hour 0..23)
  - each hour model trains only when it meets configured minimum sample thresholds

## Actions

- train: for --test-year N, trains folds from 2014..N
  - fold pattern: train 2013..(k-1), test k
  - if a compatible checkpoint exists, training is skipped and checkpoint is reused
  - default behavior is train-only
  - add --eval-after-train to evaluate each fold immediately after training
- test: evaluates only fold --test-year using existing checkpoints
- eval: evaluates all folds 2014..N using existing checkpoints

## Dataset loading

Default dataset:

- ../../data/preprocessed/merged_all_years_preprocessed.csv

Optional preprocessing split variant:

- pass --variant-stem [variant_name] and script resolves:
  - ../../data/preprocessed/splits/[variant_name]/merged_all_years_preprocessed.csv
  - reads run_params.json if present

## MLflow and artifacts

- MLflow backend: sqlite:///../../data/results/mlflow.db
- Experiment: ml-expanding-window
- Run granularity: one run per model per fold
- Fold metrics logged per segment: all, pre_conflict, post_conflict
- Hour-level metrics are stored as artifacts/files (compact run metrics stay fold-level)

Results:

- ../../data/results/ml/<run_id>/hour_metrics.csv
- ../../data/results/ml/<run_id>/fold_summary.csv
- ../../data/results/ml/<run_id>/summary.csv

Checkpoints:

- ../../data/models/ml/[model]/hourly/[dataset_tag]/train_2013-[end]__test-[year]__[hash]/
- contains hour_00.joblib ... hour_23.joblib (for trained hours)
- contains checkpoint_manifest.json

## Commands

```bash
cd src/ml

# Train one fold for one model and evaluate after training
python main.py train --test-year 2014 --models decision-tree --strategy hourly --eval-after-train

# Train all models through 2020 and evaluate each fold
python main.py train --test-year 2020 --models decision-tree,random-forest,gradient-boosting,linear-regression --strategy hourly --eval-after-train

# Evaluate all folds through 2020 from existing checkpoints
python main.py eval --test-year 2020 --models decision-tree,random-forest,gradient-boosting,linear-regression --strategy hourly
```

## Notes

- This module does not generate plots.
- Leakage guard excludes consumption_* columns except the target.
- High-missing columns are dropped when missing ratio exceeds --drop-high-missing-threshold.
