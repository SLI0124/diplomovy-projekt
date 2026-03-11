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
- `lag-llama` (`time-series-foundation-models/Lag-Llama`)
- `moirai` (`Salesforce/moirai-1.0-R-base`)
- `timesfm2.5` (`google/timesfm-2.5-200m-pytorch`)

Default `--models` runs all four.

## Modes

- `--mode one-shot`: evaluate pretrained model weights only
- `--mode finetuned`: train fold-specific model and save checkpoint, or load saved checkpoint for testing/evaluation

Default `--context-length` is `512`.

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

Default dataset:

- `../../data/preprocessed/merged_all_years_preprocessed.csv`

Optional preprocessing split variant:

- pass `--variant-stem <name>` and script resolves:
  - `../../data/preprocessed/splits/<name>/merged_all_years_preprocessed.csv`
  - reads shared preprocessing params from `run_params.json` if present

If dataset file is missing, script fails with a clear command hint to create it via `src/preprocessing/main.py`.

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
  - `adapters/foundation/lag_llama.py`
  - `adapters/foundation/moirai.py`
  - `adapters/foundation/timesfm.py`
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

## Notes

- No plots are generated in this module.
- No additional preprocessing/scaling/imputation is done here.
- GPU is used automatically when available (`torch.cuda.is_available()`).
- Finetune support in script:
  - implemented: `chronos2`, `lag-llama`, `moirai`, `timesfm2.5`
  - note: `timesfm2.5` uses a lightweight head-only fine-tuning loop for runtime practicality
