# Deep Learning Scripts (Foundation Models)

Script module that replicates the expanding-window deep-learning notebook workflow in a CLI form.

## Run location

Run **from this directory**:

```bash
cd src/dl
python main.py --help
```

## Supported models

- `chronos2` (`amazon/chronos-2`)
- `granite_ttm` (`ibm-granite/granite-timeseries-ttm-r2`)
- `moirai1_base` (`Salesforce/moirai-1.0-R-base`)
- `timesfm25` (`google/timesfm-2.5-200m-pytorch`)

Default `--models` runs `chronos2,moirai1_base,timesfm25`.

## Modes

- `--mode one-shot`: evaluate pretrained model weights only
- `--mode finetuned`: train fold-specific model and save checkpoint, or load saved checkpoint for testing/evaluation

### Covariate input mode

- `--training-input-mode univariate`: target only.
- `--training-input-mode covariate`: enables covariates for `chronos2`, `moirai1_base`, and `granite_ttm`.
- In covariate mode, selected covariates are split into past and known-future groups via:
  - `--covariate-columns`
  - `--future-covariate-columns`
  - `--past-covariate-columns`
- `timesfm25` currently does not support covariate mode in this CLI.

Default `--context-length` is `512`.

### Training loss parameter

- `--train-loss {mae,mse,rmse,mape,smape}` is supported only for custom models (currently `model_1`).
- Foundation models (`chronos2`, `moirai1_base`, `timesfm25`) reject `--train-loss`.
- If omitted, custom models default to `mse`.

### Training optimizer parameter

- `--train-optimizer {adamw,adam,sgd}` is supported only for custom models (currently `model_1`).
- Foundation models reject `--train-optimizer`.
- If omitted, custom models default to `adamw` (same as previous behavior).

### Checkpoint selection

- `--checkpoint-selection {best-train-loss,last}` controls which finetuned weights are saved.
- Default is `best-train-loss`: the model restores the epoch with the lowest training loss before checkpoint save.
- Use `last` to keep previous behavior and save final-epoch weights.

## Actions

- `train`: for `--test-year N`, runs all folds from 2014..N
  - fold pattern: train `2013..(k-1)`, test `k`
  - each fold trains only when a compatible checkpoint is missing
  - if a compatible checkpoint already exists, training is skipped and checkpoint is reused
  - default behavior is train-only (saves checkpoints, no metrics)
    - add `--eval-after-train` to evaluate each fold right after training
      - valid only with `--mode finetuned`
- `test`: evaluates only the fold for `--test-year`
  - finetuned mode loads checkpoint for `train_2013-(N-1)__test-N`
- `eval`: evaluates all folds from 2014..N
  - one-shot mode: evaluate pretrained model on all folds (no training)
  - finetuned mode: load fold checkpoints and evaluate
  - no training is performed in `test` or `eval`

`one-shot` is inference-only. `train --mode one-shot` is rejected by CLI.

For finetuned `test`/`eval`, checkpoint loading is strict: the script does not auto-load the latest model. If a checkpoint is missing, it fails with an error that includes the expected path and a command to create it.

`checkpoint_manifest.json` is required and validated before loading a finetuned checkpoint (model slug, fold test year, dataset tag, and key compatibility params).

## Dataset loading

Default preprocessing split variant:

- `base`

DL resolves split artifacts from:

- `../../data/preprocessed/splits/<variant_stem>/`

For each fold with test year `T`, DL loads:

- training data: `ranges_from_2013_to_*/range_2013_<T-1>.csv`
- test data: `single_years/year_<T>.csv`

DL does not build train/test splits internally from a merged CSV anymore. It consumes the precomputed split files directly.

If the split root or required split files are missing, script fails fast with this remediation:

```bash
cd ../preprocessing
python main.py
```

## Logging and artifacts

- Save/artifact locations are fixed in this module and are not configurable via CLI flags.
- MLflow backend default:
  - `sqlite:///../../data/results/mlflow.db`
- MLflow experiments (auto-selected per model family):
  - `deep-learning-foundation-expanding-window`
  - `deep-learning-custom-expanding-window`
- Metrics logged per segment: `SMAPE`, `MAPE`, `MAE`, `MSE`, `R²`
- Dataset provenance logged to MLflow artifacts/tags:
  - `dataset_profile.json`
  - `dataset_columns_preview.csv`
  - `run_context.json`
  - tags: dataset path/hash + model/fold/run-kind
- Results:
  - `../../data/results/deep_learning/<run_id>/results.csv`
  - `../../data/results/deep_learning/<run_id>/summary.csv`
- Fine-tuned checkpoints:
  - `../../data/models/deep_learning/<model>/finetuned/<dataset_tag>/train_2013-<end>__test-<year>__<hash>/`
  - includes `checkpoint_manifest.json` for strict compatibility checks

## Model organization

- Foundation adapters:
  - `adapters/foundation/chronos2.py`
  - `adapters/foundation/moirai1_base.py`
  - `adapters/foundation/timesfm25.py`
- Custom adapters:
  - `adapters/custom/` (register your own models in `adapters/__init__.py`)
- Backward compatibility shim:
  - `models.py` re-exports adapter API from the new structure

## Commands

### Full run set (all models, all folds/ranges, context length 512)

Use `--test-year 2025` to cover all currently available folds (2014..2025) in this dataset.

```bash
cd src/dl

# 1) Fine-tune checkpoints for all models on all folds (train-only)
python main.py train --mode finetuned --test-year 2025

# 2) Evaluate fine-tuned checkpoints for all models on all folds
python main.py eval --mode finetuned --test-year 2025

# 3) One-shot evaluation for all models on all folds
python main.py eval --mode one-shot --test-year 2025
```

### 1) Train all folds up to 2020 (fine-tuned)

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2020
```

### 2) Train all folds and evaluate immediately after each fold

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2020 --eval-after-train
```

### 3) Test one-shot pretrained on 2021 only

```bash
cd src/dl
python main.py test --mode one-shot --test-year 2021
```

### 4) Test fine-tuned checkpoint on 2021 only

```bash
cd src/dl
python main.py test --mode finetuned --test-year 2021
```

### 5) Evaluate all folds (expanding window) with existing fine-tuned checkpoints

```bash
cd src/dl
python main.py eval --mode finetuned --test-year 2021
```

### 6) Use preprocessing split variant

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2020 --variant-stem drop-year__cyc-hour-month-day-of-week-src-dropped
```

### 7) Quick smoke run (train-only)

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2014 --models chronos2 --max-origins-per-year 2 --train-epochs 1 --train-steps-per-epoch 2 --train-batch-size 2
```

### 7b) Granite TTM quick smoke (all mode/input combinations)

```bash
cd src/dl

# one-shot + univariate
python main.py test --mode one-shot --test-year 2014 --models granite_ttm --training-input-mode univariate --max-origins-per-year 2

# one-shot + covariate
python main.py test --mode one-shot --test-year 2014 --models granite_ttm --training-input-mode covariate --max-origins-per-year 2

# finetuned + univariate
python main.py train --mode finetuned --test-year 2014 --models granite_ttm --training-input-mode univariate --train-epochs 1 --train-steps-per-epoch 2 --train-batch-size 2 --eval-after-train --max-origins-per-year 2

# finetuned + covariate
python main.py train --mode finetuned --test-year 2014 --models granite_ttm --training-input-mode covariate --train-epochs 1 --train-steps-per-epoch 2 --train-batch-size 2 --eval-after-train --max-origins-per-year 2
```

### 8) Custom model with explicit training loss

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2014 --models model_1 --train-loss smape --train-optimizer adamw
```

## Notes

- No plots are generated in this module.
- No additional preprocessing/scaling/imputation or train/test split calculation is done here.
- GPU is used automatically when available (`torch.cuda.is_available()`).
- Finetune support in script:
  - implemented: `chronos2`, `moirai1_base`, `timesfm25`
  - note: `timesfm25` uses a lightweight head-only fine-tuning loop for runtime practicality
