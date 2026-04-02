from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

RUN_DIR_PATTERN = re.compile(r"^(?P<model>.+)__test-(?P<year>\d{4})\.csv$")
PlotKind = Literal["prediction", "training_losses"]


@dataclass(frozen=True)
class RunMetadata:
    run_dir: str
    models: list[str]
    training_input_mode: str
    mode_short: str
    plot_kind: PlotKind
    csv_by_model: dict[str, dict[int, Path]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot prediction or training-loss curves from a deep learning run "
            "directory in data/results/deep_learning."
        )
    )
    parser.add_argument(
        "run_dir",
        help="Run directory name, for example: 20260401_152259",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help=(
            "Optional test year to plot. If omitted, plots are generated for all "
            "available test years for the selected model."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=["auto", "prediction", "training_losses"],
        default="auto",
        help=(
            "What to plot. auto detects from run content, otherwise force "
            "prediction or training_losses."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name used in artifact file names (for example model_1). "
            "If omitted, all discovered models are plotted."
        ),
    )
    parser.add_argument(
        "--results-root",
        default=None,
        help=(
            "Optional root path that contains run directories. "
            "Default resolves to <project_root>/data/results/deep_learning."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output path for the saved figure. "
            "Default is <project_root>/data/plots/deep_learning/<run_dir>/<model>__test-<year>__true_vs_pred.png"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Saved figure DPI (default: 200)",
    )
    parser.add_argument(
        "--conflict-window",
        action="store_true",
        help=(
            "Prediction only: plot around the RU-UA conflict date instead of the full year."
        ),
    )
    parser.add_argument(
        "--conflict-date",
        default="2022-02-24",
        help="Reference date for --conflict-window (default: 2022-02-24).",
    )
    parser.add_argument(
        "--days-before",
        type=int,
        default=45,
        help="Days before conflict date for --conflict-window (default: 45).",
    )
    parser.add_argument(
        "--days-after",
        type=int,
        default=45,
        help="Days after conflict date for --conflict-window (default: 45).",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_results_root(cli_results_root: str | None) -> Path:
    if cli_results_root:
        return Path(cli_results_root).resolve()
    return _project_root() / "data" / "results" / "deep_learning"


def _load_runtime_config(run_path: Path) -> dict:
    runtime_config_path = run_path / "runtime_config.json"
    if not runtime_config_path.exists():
        raise FileNotFoundError(f"Missing runtime config: {runtime_config_path}")

    with runtime_config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_prediction_files(predictions_dir: Path) -> list[tuple[str, int, Path]]:
    discovered: list[tuple[str, int, Path]] = []
    for path in sorted(predictions_dir.glob("*.csv")):
        match = RUN_DIR_PATTERN.match(path.name)
        if not match:
            continue
        model = match.group("model")
        year = int(match.group("year"))
        discovered.append((model, year, path))
    return discovered


def _build_csv_by_model(
    discovered: list[tuple[str, int, Path]],
    selected_models: list[str],
) -> dict[str, dict[int, Path]]:
    csv_by_model: dict[str, dict[int, Path]] = {model: {} for model in selected_models}
    for model, year, path in discovered:
        if model in csv_by_model:
            csv_by_model[model][year] = path
    return csv_by_model


def _detect_plot_kind(run_path: Path, requested_kind: str) -> PlotKind:
    predictions_dir = run_path / "predictions"
    training_losses_dir = run_path / "training_losses"

    match requested_kind:
        case "prediction":
            if not predictions_dir.exists():
                raise FileNotFoundError(
                    f"Missing predictions directory: {predictions_dir}"
                )
            return "prediction"
        case "training_losses":
            if not training_losses_dir.exists():
                raise FileNotFoundError(
                    f"Missing training_losses directory: {training_losses_dir}"
                )
            return "training_losses"
        case "auto":
            if predictions_dir.exists():
                return "prediction"
            if training_losses_dir.exists():
                return "training_losses"
            raise FileNotFoundError(
                f"Run {run_path.name} has neither predictions/ nor training_losses/"
            )
        case _:
            raise ValueError(f"Unsupported --kind value: {requested_kind}")


def _artifact_dir_for_kind(run_path: Path, plot_kind: PlotKind) -> Path:
    if plot_kind == "prediction":
        return run_path / "predictions"
    return run_path / "training_losses"


def _short_mode(training_input_mode: str) -> str:
    normalized = training_input_mode.strip().lower()
    if normalized == "univariate":
        return "uni"
    if normalized == "covariate":
        return "co"
    return normalized[:3] if normalized else "unk"


def _resolve_metadata(args: argparse.Namespace) -> RunMetadata:
    results_root = _resolve_results_root(args.results_root)
    run_path = results_root / args.run_dir

    if not run_path.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_path}")

    plot_kind = _detect_plot_kind(run_path=run_path, requested_kind=args.kind)
    artifact_dir = _artifact_dir_for_kind(run_path=run_path, plot_kind=plot_kind)

    runtime_config = _load_runtime_config(run_path)
    discovered = _discover_prediction_files(artifact_dir)
    if not discovered:
        raise FileNotFoundError(f"No CSV files found in: {artifact_dir}")

    discovered_models = sorted({model for model, _, _ in discovered})
    selected_models = [args.model] if args.model else discovered_models

    csv_by_model = _build_csv_by_model(discovered, selected_models)
    empty_models = [model for model, years in csv_by_model.items() if not years]

    if empty_models:
        available = sorted({(model, year) for model, year, _ in discovered})
        available_str = ", ".join(f"{model}:{year}" for model, year in available)
        missing_str = ", ".join(empty_models)
        raise FileNotFoundError(
            "Could not find artifact CSV files for requested model(s). "
            f"Requested: {missing_str}. Available: {available_str}"
        )

    training_input_mode = str(runtime_config.get("training_input_mode", "unknown"))
    mode_short = _short_mode(training_input_mode)

    return RunMetadata(
        run_dir=args.run_dir,
        models=selected_models,
        training_input_mode=training_input_mode,
        mode_short=mode_short,
        plot_kind=plot_kind,
        csv_by_model=csv_by_model,
    )


def _prepare_timeseries(predictions_path: Path) -> pd.DataFrame:
    df = pd.read_csv(predictions_path)

    required_cols = {"target_timestamp", "y_true", "y_pred"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {predictions_path.name}: {', '.join(missing)}"
        )

    ts = df[["target_timestamp", "y_true", "y_pred"]].copy()
    ts["target_timestamp"] = pd.to_datetime(ts["target_timestamp"], errors="coerce")
    ts = ts.dropna(subset=["target_timestamp", "y_true", "y_pred"])

    # Handle potential overlapping forecast origins by averaging predictions per timestamp.
    ts = (
        ts.groupby("target_timestamp", as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
        .sort_values("target_timestamp")
    )

    if ts.empty:
        raise ValueError(f"No valid rows to plot after cleaning: {predictions_path}")

    # Scale to thousands for readability and to match axis units.
    ts["y_true"] = ts["y_true"] / 1000.0
    ts["y_pred"] = ts["y_pred"] / 1000.0

    return ts


def _slice_conflict_window(
    ts: pd.DataFrame,
    conflict_date_text: str,
    days_before: int,
    days_after: int,
) -> pd.DataFrame:
    try:
        conflict_date = pd.Timestamp(conflict_date_text)
    except Exception as exc:  # pragma: no cover - defensive parsing guard
        raise ValueError(
            f"Invalid --conflict-date value: {conflict_date_text}"
        ) from exc

    if days_before < 0 or days_after < 0:
        raise ValueError("--days-before and --days-after must be non-negative.")

    start = conflict_date - pd.Timedelta(days=days_before)
    end = conflict_date + pd.Timedelta(days=days_after)

    sliced = ts[
        (ts["target_timestamp"] >= start) & (ts["target_timestamp"] <= end)
    ].copy()
    if sliced.empty:
        raise ValueError(
            "Conflict window slice is empty. Adjust --conflict-date/--days-before/--days-after."
        )

    return sliced


def _prepare_training_losses(training_loss_path: Path) -> pd.DataFrame:
    df = pd.read_csv(training_loss_path)

    required_cols = {"epoch", "loss"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {training_loss_path.name}: {', '.join(missing)}"
        )

    ts = df[["epoch", "loss"]].copy()
    ts = ts.dropna(subset=["epoch", "loss"])
    ts["epoch"] = pd.to_numeric(ts["epoch"], errors="coerce")
    ts["loss"] = pd.to_numeric(ts["loss"], errors="coerce")
    ts = ts.dropna(subset=["epoch", "loss"])
    ts = ts.sort_values("epoch")

    if ts.empty:
        raise ValueError(f"No valid rows to plot after cleaning: {training_loss_path}")

    return ts


def _default_output_path(
    meta: RunMetadata,
    model: str,
    year: int,
    args: argparse.Namespace,
) -> Path:
    out_dir = _project_root() / "data" / "plots" / "deep_learning" / meta.run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if meta.plot_kind == "prediction":
        if args.conflict_window:
            return out_dir / (
                f"{model}__{meta.mode_short}__test-{year}"
                "__okoli_konfliktu_ru_ua_3m__skutecnost_vs_predikce.png"
            )
        return (
            out_dir
            / f"{model}__{meta.mode_short}__test-{year}__skutecnost_vs_predikce.png"
        )
    return out_dir / f"{model}__{meta.mode_short}__test-{year}__treninkova_ztrata.png"


def _resolve_output_path(
    args: argparse.Namespace,
    meta: RunMetadata,
    model: str,
    year: int,
    plotting_multiple_outputs: bool,
) -> Path:
    default_path = _default_output_path(meta, model, year, args)
    if not args.output:
        return default_path

    cli_output = Path(args.output).resolve()
    default_name = default_path.name

    if plotting_multiple_outputs:
        if cli_output.suffix.lower() == ".png":
            raise ValueError(
                "--output points to a single PNG file but multiple plots are being generated. "
                "Provide a directory path for --output or reduce model/year selection."
            )
        cli_output.mkdir(parents=True, exist_ok=True)
        return cli_output / default_name

    if cli_output.suffix.lower() == ".png":
        cli_output.parent.mkdir(parents=True, exist_ok=True)
        return cli_output

    cli_output.mkdir(parents=True, exist_ok=True)
    return cli_output / default_name


def _plot_and_save(
    data: pd.DataFrame,
    model: str,
    meta: RunMetadata,
    year: int,
    output_path: Path,
    dpi: int,
) -> None:
    plt.figure(figsize=(16, 7))
    if meta.plot_kind == "prediction":
        plt.plot(
            data["target_timestamp"],
            data["y_true"],
            label="Skutečnost",
            linewidth=1.4,
            alpha=0.95,
        )
        plt.plot(
            data["target_timestamp"],
            data["y_pred"],
            label="Predikce",
            linewidth=1.2,
            alpha=0.9,
        )
        plt.xlabel("Datum")
        plt.ylabel("Spotřeba [tis.]")
    else:
        plt.plot(
            data["epoch"],
            data["loss"],
            label="Tréninková ztráta",
            linewidth=1.6,
            alpha=0.95,
        )
        plt.xlabel("Epocha")
        plt.ylabel("Ztráta")

    plt.title(f"{model} | {meta.mode_short} | rok {year}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = _parse_args()
    meta = _resolve_metadata(args)

    if args.conflict_window and meta.plot_kind != "prediction":
        raise ValueError("--conflict-window can be used only with prediction plots.")

    years_by_model: dict[str, list[int]] = {}
    for model in meta.models:
        available_years = sorted(meta.csv_by_model[model].keys())
        if args.conflict_window:
            years_to_plot = [args.year] if args.year is not None else [2022]
        else:
            years_to_plot = [args.year] if args.year is not None else available_years

        missing_years = [
            year for year in years_to_plot if year not in meta.csv_by_model[model]
        ]
        if missing_years:
            available_years_text = ", ".join(str(y) for y in available_years)
            missing_text = ", ".join(str(y) for y in sorted(missing_years))
            raise FileNotFoundError(
                "No artifact CSV found for requested year(s). "
                f"model={model}, missing={missing_text}, available={available_years_text}"
            )

        years_by_model[model] = years_to_plot

    total_plots = sum(len(years) for years in years_by_model.values())
    plotting_multiple_outputs = total_plots > 1

    for model in meta.models:
        for year in years_by_model[model]:
            source_csv = meta.csv_by_model[model][year]
            if meta.plot_kind == "prediction":
                data = _prepare_timeseries(source_csv)
                if args.conflict_window:
                    data = _slice_conflict_window(
                        ts=data,
                        conflict_date_text=args.conflict_date,
                        days_before=args.days_before,
                        days_after=args.days_after,
                    )
            else:
                data = _prepare_training_losses(source_csv)
            output_path = _resolve_output_path(
                args=args,
                meta=meta,
                model=model,
                year=year,
                plotting_multiple_outputs=plotting_multiple_outputs,
            )
            _plot_and_save(
                data=data,
                model=model,
                meta=meta,
                year=year,
                output_path=output_path,
                dpi=args.dpi,
            )

            print(f"Saved plot: {output_path}")
            print(f"run_dir={meta.run_dir}")
            print(f"model={model}")
            print(f"training_input_mode={meta.training_input_mode}")
            print(f"mode_short={meta.mode_short}")
            print(f"plot_kind={meta.plot_kind}")
            print(f"year={year}")
            print(f"source_csv={source_csv}")


if __name__ == "__main__":
    main()
