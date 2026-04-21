# Deep Learning Scripts (Foundation & Custom Models)

Script module that replicates the expanding-window deep-learning notebook workflow in a CLI form.

## Run location

Run **from this directory**:

```bash
cd src/dl
python main.py --help
```

## Supported models

### Foundation models

- `chronos2` (`amazon/chronos-2`)
- `granite` (`ibm-granite/granite-timeseries-ttm-r2`)
- `moirai1` (`Salesforce/moirai-1.0-R-base`)

### Custom models

- `model_1` (BiGRU baseline with target + covariate support)
- `model_2` (Conv1D + BiGRU with target + covariate support)
- `model_3` (Conv1D + BiGRU + attention refinement with target + covariate support)

Default `--models` runs `chronos2,moirai1,granite` in this order.

## Modes

- `--mode one-shot`: evaluate pretrained model weights only
- `--mode finetuned`: train fold-specific model and save checkpoint, or load saved checkpoint for testing/evaluation

### Required Parameters

- `action {train,test,eval}`: the execution mode
- `--training-input-mode {univariate,covariate}`: the input data mode
- `--test-year YYYY`: the target year for the backtest or single-fold evaluation

### Covariate input mode

- `--training-input-mode univariate`: target only (supported for all models).
- `--training-input-mode covariate`: enables covariates for `chronos2`, `moirai1`, `granite`, `model_1`, `model_2`, and `model_3`.
- In covariate mode, selected covariates are split into past and known-future groups via:
  - `--covariate-columns`
  - `--future-covariate-columns`
  - `--past-covariate-columns`

Default `--context-length` is `512`.

### Evaluation and Performance Tuning

- `--num-samples N`: number of samples for probabilistic forecasts (Moirai, Chronos). Default is `20`.
- `--window-stride N`: stride for the expanding window. Default is `24`.
- `--prediction-length N`: forecast horizon. Default is `24`.
- `--target-col NAME`: the name of the target column. Default is `consumption_total`.
- `--max-origins-per-year N`: cap the number of windows per fold for faster experiments.
- `--seed N`: random seed for reproducibility. Default is `42`.

### Evaluation Logic: Expanding Window

Evaluation follows an **expanding window** (walk-forward) approach for each fold:

- **Origins**: The script iterates through the test year, placing a forecast "origin" every `--window-stride` steps (default 24).
- **Context Construction**: For each origin, the script concatenates the entire available history (all training years from 2013 + the elapsed portion of the test year). The model then receives the most recent `--context-length` (default 512) points from this history as its input.
- **Horizon**: From each origin, the model predicts the next `--prediction-length` (default 24) steps.
- **Metrics**: Results are aggregated across all windows to produce the final fold metrics (e.g., if a year has 8760 hours and stride is 24, it evaluates ~365 windows).

### Training loss parameter

- `--train-loss {mae,mse,rmse,mape,smape}` is supported only for custom models (`model_1`, `model_2`, `model_3`).
- Foundation models (`chronos2`, `granite`, `moirai1`) reject `--train-loss`.
- If omitted, custom models default to `mse`.

### Training optimizer parameter

- `--train-optimizer {adamw,adam,sgd}` is supported only for custom models (`model_1`, `model_2`, `model_3`).
- Foundation models reject `--train-optimizer`.
- If omitted, custom models default to `adamw` (same as previous behavior).

### Checkpoint selection and identification

- `--checkpoint-selection {best-train-loss,last}` controls which finetuned weights are saved.
- Default is `best-train-loss`: the model restores the epoch with the lowest training loss before checkpoint save.
- Use `last` to save final-epoch weights.
- Checkpoint directory paths include a short MD5 hash of training parameters and model architecture signature to ensure that existing checkpoints are only reused if the configuration matches exactly.
- Important: in `--mode finetuned`, `test` and `eval` must use the same training hyperparameters that produced the checkpoint hash (for example `--train-epochs`, `--train-batch-size`, `--train-steps-per-epoch`, `--train-lr`, `--train-weight-decay`). If they differ, strict loading will report checkpoint-not-found for the computed hash path.

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
  - `adapters/foundation/granite.py`
  - `adapters/foundation/moirai1.py`
- Custom adapters:
  - `adapters/custom/` (register your own models in `adapters/__init__.py`)
- Backward compatibility shim:
  - `models.py` re-exports adapter API from the new structure

## Commands

### Full run set (Foundation models, all folds, context length 512)

Use `--test-year 2025` to cover all available folds (2014..2025).

```bash
# 1) Train fine-tune: Univariate (Train from target only)
python main.py train --mode finetuned --training-input-mode univariate --test-year 2025

# 2) Train fine-tune: Covariate (Train with multiple features)
python main.py train --mode finetuned --training-input-mode covariate --test-year 2025

# 3) Evaluate fine-tuned checkpoints (Univariate)
python main.py eval --mode finetuned --training-input-mode univariate --test-year 2025

# 4) Evaluate fine-tuned checkpoints (Covariate)
python main.py eval --mode finetuned --training-input-mode covariate --test-year 2025

# 5) Evaluate one-shot pretrained weights (Covariate)
python main.py eval --mode one-shot --training-input-mode covariate --test-year 2025

# 6) Evaluate one-shot pretrained weights (Univariate)
python main.py eval --mode one-shot --training-input-mode univariate --test-year 2025
```

or run the whole sequence with a single command as one-liner: (bit complex but those are all the combinations I've used for paper)

```bash
python main.py train --mode finetuned --training-input-mode univariate --test-year 2025 && python main.py train --mode finetuned --training-input-mode covariate --test-year 2025 && python main.py eval --mode finetuned --training-input-mode univariate --test-year 2025 && python main.py eval --mode finetuned --training-input-mode covariate --test-year 2025 && python main.py eval --mode one-shot --training-input-mode covariate --test-year 2025 && python main.py eval --mode one-shot --training-input-mode univariate --test-year 2025
```

### Common Workflows

#### 1) Train all folds up to 2022 with custom epochs

```bash
python main.py train --mode finetuned --training-input-mode univariate --test-year 2022 --train-epochs 20
```

#### 2) Train and evaluate immediately (single fold)

```bash
python main.py train --mode finetuned --training-input-mode covariate --test-year 2021 --eval-after-train
```

#### 3) Test fine-tuned checkpoint on a specific year

```bash
python main.py test --mode finetuned --training-input-mode univariate --test-year 2021
```

#### 4) Use specific covariate columns

```bash
python main.py train --mode finetuned --training-input-mode covariate --test-year 2020 --covariate-columns consumption_total,temp,holiday --future-covariate-columns holiday
```

#### 5) Custom model (LSTM) training

```bash
# Custom models support univariate and covariate training
python main.py train --mode finetuned --training-input-mode covariate --test-year 2014 --models model_1 --train-loss smape --train-epochs 50

#### 6) Full run for custom model (50 epochs)

```bash
# Train all folds (2014..2025), univariate
python main.py train --mode finetuned --training-input-mode univariate --test-year 2025 --models model_1 --train-epochs 50

# Train all folds (2014..2025), covariate
python main.py train --mode finetuned --training-input-mode covariate --test-year 2025 --models model_1 --train-epochs 50

# Evaluate trained checkpoints, univariate
python main.py eval --mode finetuned --training-input-mode univariate --test-year 2025 --models model_1 --train-epochs 50

# Evaluate trained checkpoints, covariate
python main.py eval --mode finetuned --training-input-mode covariate --test-year 2025 --models model_1 --train-epochs 50
```

### Quick smoke runs (Performance check)

```bash
# Foundation smoke (Chronos2)
python main.py train --mode finetuned --training-input-mode univariate --test-year 2014 --models chronos2 --max-origins-per-year 2 --train-epochs 1 --train-steps-per-epoch 2

# Granite TTM smoke (Covariate)
python main.py train --mode finetuned --training-input-mode covariate --test-year 2014 --models granite --max-origins-per-year 5 --train-epochs 1 --eval-after-train
```

## Notes

- No plots are generated in this module.
- No additional train/test split calculation is done here.
- Foundation adapters do not add custom preprocessing in this module.
- Custom adapters `model_1`, `model_2`, and `model_3` consume the split data as-is and differ only by cumulatively added architecture blocks.
- GPU is used automatically when available (`torch.cuda.is_available()`).
- Finetune support in script:
  - implemented: `chronos2`, `granite`, `moirai1`, `model_1`, `model_2`, `model_3`
