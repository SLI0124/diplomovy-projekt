# Deep Learning Runner 🤖

This module is the CLI version of the deep-learning workflow used in the paper. It runs expanding-window experiments for both foundation models and custom models, logs results to MLflow, saves fold outputs, and reuses checkpoints when possible.

If you only need the short version, it is this:

1. Preprocessing prepares split files.
2. This module trains, tests, or evaluates models on those splits.
3. Results go to `data/results/deep_learning`, checkpoints go to `data/models/deep_learning`, and MLflow logs go to `data/results/mlflow.db`.

## Run from here 📍

```bash
cd src/dl
python main.py --help
```

## What it supports

### Model families

- Foundation: `chronos2`, `granite`, `moirai1`
- Custom: `model_1`, `model_2`, `model_3`

### Main modes

- `--mode one-shot`: evaluate pretrained weights only
- `--mode finetuned`: train or load fold-specific checkpoints

### Main actions

- `train`: train all folds up to `--test-year`
- `test`: evaluate only the selected `--test-year`
- `eval`: evaluate all folds up to `--test-year`

## What was used in the paper ✨

The paper workflow mainly used:

- foundation models in `one-shot` and `finetuned`
- `univariate` and `covariate` input modes
- custom models with longer training, typically `50` epochs
- expanding-window evaluation over all folds up to `2025`

In practice, the most important flags are:

- `--mode`
- `--training-input-mode`
- `--test-year`
- `--models`
- `--train-epochs` for finetuned/custom runs

## The commands that actually matter

### Foundation models: full paper-style sweep

Use `--test-year 2025` to cover all folds from `2014` to `2025`.

```bash
# Train finetuned checkpoints, univariate
python main.py train --mode finetuned --training-input-mode univariate --test-year 2025

# Train finetuned checkpoints, covariate
python main.py train --mode finetuned --training-input-mode covariate --test-year 2025

# Evaluate finetuned checkpoints, univariate
python main.py eval --mode finetuned --training-input-mode univariate --test-year 2025

# Evaluate finetuned checkpoints, covariate
python main.py eval --mode finetuned --training-input-mode covariate --test-year 2025

# Evaluate one-shot models, covariate
python main.py eval --mode one-shot --training-input-mode covariate --test-year 2025

# Evaluate one-shot models, univariate
python main.py eval --mode one-shot --training-input-mode univariate --test-year 2025
```

### Custom models: common paper-style run

```bash
# Train custom model, univariate
python main.py train --mode finetuned --training-input-mode univariate --test-year 2025 --models model_1 --train-epochs 50

# Train custom model, covariate
python main.py train --mode finetuned --training-input-mode covariate --test-year 2025 --models model_1 --train-epochs 50

# Evaluate custom model, univariate
python main.py eval --mode finetuned --training-input-mode univariate --test-year 2025 --models model_1 --train-epochs 50

# Evaluate custom model, covariate
python main.py eval --mode finetuned --training-input-mode covariate --test-year 2025 --models model_1 --train-epochs 50
```

### One simple single-fold example

```bash
python main.py train --mode finetuned --training-input-mode covariate --test-year 2021 --eval-after-train
```

## Three slightly absurd examples 🎭

These are mostly here to show what the CLI can express.

### 1) Tiny smoke run

```bash
python main.py train --mode finetuned --training-input-mode univariate --test-year 2014 --models chronos2 --max-origins-per-year 2 --train-epochs 1 --train-steps-per-epoch 2
```

### 2) Very specific covariate setup

```bash
python main.py train --mode finetuned --training-input-mode covariate --test-year 2020 --models model_3 --covariate-columns consumption_total,temp,holiday --future-covariate-columns holiday --train-loss smape --train-epochs 50
```

### 3) One-shot evaluation for just one foundation model

```bash
python main.py eval --mode one-shot --training-input-mode covariate --test-year 2025 --models granite
```

## What goes in, what comes out

### Inputs

This module expects precomputed preprocessing splits in:

```text
../../data/preprocessed/splits/<variant_stem>/
```

For fold year `T`, it loads:

- training range ending in `T-1`
- test file for year `T`

If the splits are missing, run:

```bash
cd ../preprocessing
python main.py
```

### Outputs

- Run outputs: `../../data/results/deep_learning/<run_id>/`
- Checkpoints: `../../data/models/deep_learning/<model>/finetuned/...`
- MLflow DB: `../../data/results/mlflow.db`
- MLflow artifacts: `../../data/results/mlflow-artifacts/...`

## A few things worth knowing 💡

- `one-shot` is inference-only, so `train --mode one-shot` is not valid.
- Finetuned `test` and `eval` load checkpoints strictly. If the checkpoint config does not match, loading fails instead of guessing.
- Checkpoint compatibility is tracked through a hash and `checkpoint_manifest.json`.
- Default split variant is `base`.
- GPU is used automatically when available.
- This module does not generate plots.

## If you want to explore further

Look at:

- `main.py`: CLI entrypoint
- `runner.py`: execution flow
- `dataset.py`: loading split data
- `mlflow_logging.py`: MLflow integration
- `adapters/foundation/`: foundation-model adapters
- `adapters/custom/`: custom-model adapters
