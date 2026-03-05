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
- `--mode finetuned`: train fold-specific model and save checkpoint, or load saved checkpoint for testing

## Actions

- `train`: for `--test-year N`, runs all folds from 2014..N
  - fold pattern: train `2013..(k-1)`, test `k`
  - each fold starts from scratch
  - valid only with `--mode finetuned`
- `test`: evaluates only the fold for `--test-year`
  - finetuned mode loads checkpoint for `train_2013-(N-1)__test-N`
- `eval`: evaluates all folds from 2014..N
  - one-shot mode: evaluate pretrained model on all folds (no training)
  - finetuned mode: load fold checkpoints and evaluate

`one-shot` is inference-only. `train --mode one-shot` is rejected by CLI.

## Dataset loading

Default dataset:

- `../../data/preprocessed/merged_all_years_preprocessed.csv`

Optional preprocessing split variant:

- pass `--variant-stem <name>` and script resolves:
  - `../../data/preprocessed/splits/<name>/merged_all_years_preprocessed.csv`
  - reads shared preprocessing params from `run_params.json` if present

If dataset file is missing, script fails with a clear command hint to create it via `src/preprocessing/main.py`.

## Logging and artifacts

- MLflow backend default:
  - `sqlite:///../../data/results/deep_learning/mlflow.db`
- Metrics logged per segment: `SMAPE`, `MAPE`, `MAE`, `MSE`, `R²`
- Dataset provenance logged to MLflow artifacts/tags:
  - `dataset_profile.json`
  - `run_context.json`
  - tags: dataset path/hash + model/fold/run-kind
- Results:
  - `../../data/results/deep_learning/<run_id>/results.csv`
  - `../../data/results/deep_learning/<run_id>/summary.csv`
- Fine-tuned checkpoints:
  - `../../data/models/deep_learning/<model>/finetuned/<dataset_tag>/train_2013-<end>__test-<year>__<hash>/`

## Commands

### 1) Train all folds up to 2020 (fine-tuned)

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2020
```

### 2) Test one-shot pretrained on 2021 only

```bash
cd src/dl
python main.py test --mode one-shot --test-year 2021
```

### 3) Test fine-tuned checkpoint on 2021 only

```bash
cd src/dl
python main.py test --mode finetuned --test-year 2021
```

### 4) Use preprocessing split variant

```bash
cd src/dl
python main.py train --mode finetuned --test-year 2020 --variant-stem drop-year__cyc-hour-month-day-of-week-src-dropped
```

### 5) Quick smoke run

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
