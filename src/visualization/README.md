# Visualization Scripts 📊

This folder is `src/visualization` in the current repo layout.

## Recommended way to run

Some newer scripts resolve paths from `__file__`, but older plotting scripts use relative paths like `../../data/...`. The safest pattern is:

```bash
cd src/visualization
python <script>.py ...
```

Use the project Python environment. The MLflow scripts also require `mlflow` to be installed.

## What is here ✨

- `mlflow_metrics_summary.py`: filtered MLflow export to compact JSON files
- `export_mlflow_results.py`: full MLflow export for experiments 1 and 2
- `plot_delta_improvment.py`: plots and CSV summaries from MLflow JSON exports
- `export_conflict_summaries.py`: pre/post-conflict comparison CSV exports
- `plot_run_artifact_result.py`: plots prediction or training-loss artifacts for one deep-learning run
- `plot_sarimax_results.py`: plots yearly SARIMAX predictions
- `eda_plots.py`: general EDA plots from the merged preprocessed dataset
- `pre_eda_plots.py`: one early consumption plot for `consumption_vcpnet`
- `plot_global_temperature.py`: downloads NOAA temperature anomalies and plots them
- `__init__.py`: empty package marker

## Script reference

### `mlflow_metrics_summary.py` 📦

Exports filtered runs from `data/results/mlflow.db` into compact JSON files under `data/data-exports/mlflow_exports/<experiment>/`.

What it filters by:

- `--experiments`: `foundation`, `custom`, `1`, `2`
- `--action`
- `--mode`
- `--training-input-mode`
- `--stem`
- `--model`
- `--status`
- `--epochs`
- repeated `--filter key=value`

Useful notes:

- `foundation -> experiment_id=1`
- `custom -> experiment_id=2`
- output filenames include the applied filters and a UTC timestamp
- it prunes the export to mostly `all.*` metrics plus selected params and tags

Example matching your current usage pattern:

```bash
python mlflow_metrics_summary.py \
  --experiments foundation \
  --action eval \
  --mode finetuned \
  --training-input-mode covariate \
  --stem base \
  --model granite \
  --epochs 10
```

Another example for custom models:

```bash
python mlflow_metrics_summary.py \
  --experiments custom \
  --action eval \
  --mode finetuned \
  --training-input-mode univariate \
  --stem base \
  --model model_2 \
  --epochs 50
```

Repo-specific caveat:

- in the current exported filenames, `one-shot` foundation runs do not appear to carry `train_epochs`; if `--mode one-shot --epochs ...` returns no runs, retry without `--epochs` 🔎

### `export_mlflow_results.py`

Exports full, unfiltered MLflow experiment dumps from `data/results/mlflow.db`.

Default outputs:

- `data/data-exports/mlflow_experiment_1_foundation_model.json`
- `data/data-exports/mlflow_experiment_2_own_model.json`

Run:

```bash
python export_mlflow_results.py
```

Custom paths:

```bash
python export_mlflow_results.py \
  --db-path ../../data/results/mlflow.db \
  --output-dir ../../data/data-exports
```

Use this when you want a raw archive of all runs, not a thesis-ready filtered export.

### `plot_delta_improvment.py` 📈

Consumes MLflow JSON exports and generates yearly MAPE plots, delta plots, and summary CSVs.

Default inputs:

- foundation exports: `data/data-exports/mlflow_exports/foundation`
- custom exports: `data/data-exports/mlflow_exports/custom`

Default outputs:

- foundation plots and CSVs: `data/plots/foundation`
- custom plots and CSVs: `data/plots/custom`

Run:

```bash
python plot_delta_improvment.py
```

With explicit folders:

```bash
python plot_delta_improvment.py \
  --input-dir ../../data/data-exports/mlflow_exports/foundation \
  --custom-input-dir ../../data/data-exports/mlflow_exports/custom \
  --output-dir ../../data/plots/foundation \
  --custom-output-dir ../../data/plots/custom
```

What it produces:

- foundation yearly MAPE plots for both modes and both input modes
- foundation delta plots for `one-shot - finetuned`
- foundation delta plots for `univariate - covariate`
- custom yearly MAPE plots for finetuned 10-epoch runs
- custom covariate delta plot
- custom year-by-year comparison of `model_2` vs `model_3` in `finetuned + covariate`
- summary CSV tables next to the plots

### `export_conflict_summaries.py` ⚔️

Builds CSV summaries for the thesis conflict analysis chapter.

Data sources:

- deep learning runs from `data/results/deep_learning/<run_dir>/predictions/*.csv`
- SARIMAX results from `data/results/sarimax_stepup`

Default output folder:

- `data/data-exports/conflict_analysis`

Run:

```bash
python export_conflict_summaries.py
```

Useful options:

- `--include-extra-epochs`: also include foundation 20-epoch and custom 50-epoch runs
- `--include-granite-oneshot-covariate`: include the Granite one-shot covariate case that is excluded by default
- `--year-of-interest 2022`: focused split year
- `--conflict-date 2022-02-24 00:00:00`: split date

Outputs:

- `conflict_2022_segments_all.csv`
- `conflict_2022_summary_all.csv`
- `conflict_2022_summary_latest.csv`
- `conflict_regime_segments_all.csv`
- `conflict_regime_summary_all.csv`
- `conflict_regime_summary_latest.csv`

By default it keeps only the configurations used in the thesis:

- foundation models with 10 epochs
- custom models with 10 epochs
- SARIMAX
- Granite `one-shot + covariate` excluded

### `plot_run_artifact_result.py` 🧪

Plots artifacts from one deep-learning run directory inside `data/results/deep_learning`.

It supports:

- prediction plots from `predictions/<model>__test-<year>.csv`
- training-loss plots from `training_losses/<model>__test-<year>.csv`

It also reads `runtime_config.json` from the run directory.

Basic example:

```bash
python plot_run_artifact_result.py 20260421_115905 --model model_1 --year 2022
```

Plot around the RU-UA conflict window:

```bash
python plot_run_artifact_result.py \
  20260421_115905 \
  --model model_1 \
  --year 2022 \
  --conflict-window \
  --days-before 45 \
  --days-after 45
```

Plot only a selected date range:

```bash
python plot_run_artifact_result.py \
  20260421_115905 \
  --model model_1 \
  --year 2022 \
  --start-date 2022-01-01 \
  --end-date 2022-03-31
```

Plot training losses instead of predictions:

```bash
python plot_run_artifact_result.py \
  20260421_115905 \
  --kind training_losses \
  --model model_1 \
  --year 2022
```

Default output location:

- `data/plots/deep_learning/<run_dir>/...png`

Useful options:

- `--kind auto|prediction|training_losses`
- `--model`
- `--year`
- `--output`
- `--plot-difference`
- `--difference-y-offset auto|<number>`

### `plot_sarimax_results.py`

Plots yearly SARIMAX predictions from:

- `data/results/sarimax_stepup/predictions_by_year/sarimax_predictions_<year>.csv`

Default output folder:

- `data/plots/sarimax_stepup`

Run:

```bash
python plot_sarimax_results.py
```

Specific years:

```bash
python plot_sarimax_results.py --years 2021 2022
```

The script also prints how many negative predictions occurred in each plotted year.

### `eda_plots.py` 🌿

Creates the main EDA figure set from the merged preprocessed dataset.

Default direct-run input:

- `../../data/preprocessed/merged_all_years_preprocessed.csv`

Default output folder:

- `../../data/plots`

Run:

```bash
python eda_plots.py
```

What it generates:

- daily consumption over time
- boxplot of absolute total consumption
- histogram of absolute total consumption
- monthly average consumption vs temperature
- year-over-year average consumption vs average price
- weighted gas price series for 2020-2025
- yearly gas price distribution

If you want different paths, import it and call:

```python
from pathlib import Path
from eda_plots import plot_all

plot_all(
    csv_path=Path("../../data/preprocessed/merged_all_years_preprocessed.csv"),
    save_path=Path("../../data/plots"),
)
```

### `pre_eda_plots.py`

Creates one earlier exploratory plot for `consumption_vcpnet` from:

- `../../data/processed/merged/merged_all_years.csv`

Run:

```bash
python pre_eda_plots.py
```

Output:

- `../../data/plots/pre_eda_consumption_vcpnet_2014_2018.png`

Important code detail:

- despite the output filename, the current script filters only `2016-01-01` to `2016-12-31` ⚠️

### `plot_global_temperature.py` 🌍

Downloads NOAA global temperature anomaly data and plots it.

Run:

```bash
python plot_global_temperature.py
```

What it does:

- downloads `global_temperature.csv` into `../../data/`
- saves `../../data/plots/global_temperature_plot.png`

Requirements:

- internet access
- `requests`
- `tqdm`

The dataset URL is hard-coded in the script.

## Typical workflow 🚀

If your goal is thesis figures from MLflow runs, the common flow is:

1. Run `mlflow_metrics_summary.py` for the configurations you want to export.
2. Run `plot_delta_improvment.py` on the exported JSON files.
3. Use `plot_run_artifact_result.py` for selected run-level prediction or loss figures.
4. Use `export_conflict_summaries.py` for the chapter 6 pre/post-conflict CSV summaries.
