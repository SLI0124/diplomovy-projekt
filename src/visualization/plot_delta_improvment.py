from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

FOUNDATION_MODEL_ORDER = ["chronos2", "granite", "moirai1"]
CUSTOM_MODEL_ORDER = ["model_1", "model_2", "model_3"]
MODE_ORDER = ["one-shot", "finetuned"]
INPUT_MODE_ORDER = ["univariate", "covariate"]
FOUNDATION_MODEL_LABELS = {
    "chronos2": "Chronos2",
    "granite": "Granite",
    "moirai1": "Moirai1",
}
CUSTOM_MODEL_LABELS = {
    "model_1": "Rekurentní model",
    "model_2": "Konvoluční model",
    "model_3": "Model s pozornosti",
}
INPUT_MODE_LABELS = {
    "univariate": "Jednorozměrný vstup",
    "covariate": "Vícerozměrný vstup",
}
MODE_LABELS = {
    "one-shot": "Bez doladění",
    "finetuned": "Po doladění",
}
FOUNDATION_MODEL_COLORS = {
    "chronos2": "#1b9e77",
    "granite": "#d95f02",
    "moirai1": "#7570b3",
}
CUSTOM_MODEL_COLORS = {
    "model_1": "#4c78a8",
    "model_2": "#f58518",
    "model_3": "#54a24b",
}
INPUT_MODE_COLORS = {
    "univariate": "#4c78a8",
    "covariate": "#f58518",
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate foundation-model MAPE and delta plots from MLflow JSON exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root
        / "data"
        / "data-exports"
        / "mlflow_exports"
        / "foundation",
        help="Folder containing foundation MLflow JSON exports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "plots" / "foundation",
        help="Output folder for generated plots.",
    )
    parser.add_argument(
        "--custom-input-dir",
        type=Path,
        default=project_root / "data" / "data-exports" / "mlflow_exports" / "custom",
        help="Folder containing custom-model MLflow JSON exports.",
    )
    parser.add_argument(
        "--custom-output-dir",
        type=Path,
        default=project_root / "data" / "plots" / "custom",
        help="Output folder for generated custom-model plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="Output DPI for saved figures.",
    )
    return parser.parse_args()


def _normalize_mode(raw_mode: str | None) -> str | None:
    if raw_mode is None:
        return None
    normalized = str(raw_mode).strip().lower()
    if normalized in {"one-shot", "oneshot"}:
        return "one-shot"
    if normalized == "finetuned":
        return "finetuned"
    return normalized


def _read_export_rows(
    input_dir: Path,
    allowed_models: list[str],
    train_epochs: str | None = None,
) -> pd.DataFrame:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for run in payload.get("runs", []):
            params = run.get("params", {})
            metrics = run.get("metrics", {})
            tags = run.get("tags", {})

            model = tags.get("run.model.resolved") or payload.get(
                "applied_filters", {}
            ).get("model")
            mode = _normalize_mode(
                params.get("mode") or payload.get("applied_filters", {}).get("mode")
            )
            input_mode = params.get("training_input_mode") or payload.get(
                "applied_filters", {}
            ).get("training_input_mode")
            year = params.get("test_year")
            mape = metrics.get("all.mape")
            epochs = params.get("train_epochs") or payload.get(
                "applied_filters", {}
            ).get("train_epochs")

            if not model or not mode or not input_mode or year is None or mape is None:
                continue
            if train_epochs is not None and str(epochs) != str(train_epochs):
                continue

            rows.append(
                {
                    "source_file": path.name,
                    "model": str(model).strip().lower(),
                    "mode": mode,
                    "training_input_mode": str(input_mode).strip().lower(),
                    "test_year": int(year),
                    "mape": float(mape),
                    "train_epochs": None if epochs is None else int(epochs),
                }
            )

    if not rows:
        raise ValueError(f"No usable runs found in {input_dir}")

    df = pd.DataFrame(rows)
    df = df[df["model"].isin(allowed_models)]
    df = df[df["mode"].isin(MODE_ORDER)]
    df = df[df["training_input_mode"].isin(INPUT_MODE_ORDER)]
    if train_epochs is not None:
        df = df[df["train_epochs"] == int(train_epochs)]
    if df.empty:
        raise ValueError(f"No matching runs found in {input_dir}")

    df = (
        df.sort_values(["model", "mode", "training_input_mode", "test_year"])
        .drop_duplicates(
            subset=["model", "mode", "training_input_mode", "test_year"], keep="last"
        )
        .reset_index(drop=True)
    )
    return df


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 23,
            "axes.titlesize": 23,
            "axes.labelsize": 25,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 21,
        }
    )


def _plot_yearly_mape(
    df: pd.DataFrame,
    mode: str,
    input_mode: str,
    model_order: list[str],
    model_labels: dict[str, str],
    model_colors: dict[str, str],
    output_dir: Path,
    dpi: int,
    filename_prefix: str,
) -> Path:
    plot_df = df[
        (df["mode"] == mode) & (df["training_input_mode"] == input_mode)
    ].copy()
    if plot_df.empty:
        raise ValueError(f"No rows found for mode={mode}, input_mode={input_mode}")

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)

    for model in model_order:
        model_df = plot_df[plot_df["model"] == model].sort_values("test_year")
        ax.plot(
            model_df["test_year"],
            model_df["mape"],
            marker="o",
            linewidth=3.0,
            markersize=7.0,
            color=model_colors[model],
            label=model_labels[model],
        )

    ax.set_xlabel("Testovací rok")
    ax.set_ylabel("MAPE (%)")
    years = sorted(plot_df["test_year"].unique())
    ax.set_xticks(years[::2])
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=1)

    output_path = output_dir / (
        f"{filename_prefix}_mape_by_year_{mode.replace('-', '_')}_{input_mode}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _aggregate_deltas(
    df: pd.DataFrame,
    model_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = df.pivot_table(
        index=["model", "test_year"],
        columns=["mode", "training_input_mode"],
        values="mape",
        aggfunc="mean",
    ).sort_index()

    finetuning_rows: list[dict[str, Any]] = []
    for model in model_order:
        if model not in pivot.index.get_level_values("model"):
            continue
        model_pivot = pivot.xs(model, level="model")
        for input_mode in INPUT_MODE_ORDER:
            oneshot = model_pivot.get(("one-shot", input_mode))
            finetuned = model_pivot.get(("finetuned", input_mode))
            if oneshot is None or finetuned is None:
                continue
            comparison = pd.concat(
                [oneshot, finetuned], axis=1, keys=["oneshot", "finetuned"]
            ).dropna()
            if comparison.empty:
                continue
            improvement = comparison["oneshot"] - comparison["finetuned"]
            finetuning_rows.append(
                {
                    "model": model,
                    "training_input_mode": input_mode,
                    "mean_delta_mape": float(improvement.mean()),
                }
            )

    covariate_rows: list[dict[str, Any]] = []
    for model in model_order:
        if model not in pivot.index.get_level_values("model"):
            continue
        model_pivot = pivot.xs(model, level="model")
        for mode in MODE_ORDER:
            univariate = model_pivot.get((mode, "univariate"))
            covariate = model_pivot.get((mode, "covariate"))
            if univariate is None or covariate is None:
                continue
            comparison = pd.concat(
                [univariate, covariate], axis=1, keys=["univariate", "covariate"]
            ).dropna()
            if comparison.empty:
                continue
            improvement = comparison["univariate"] - comparison["covariate"]
            covariate_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "mean_delta_mape": float(improvement.mean()),
                }
            )

    return pd.DataFrame(finetuning_rows), pd.DataFrame(covariate_rows)


def _plot_finetuning_delta(delta_df: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    x_positions = list(range(len(FOUNDATION_MODEL_ORDER)))
    width = 0.34

    for idx, input_mode in enumerate(INPUT_MODE_ORDER):
        subset = delta_df[delta_df["training_input_mode"] == input_mode].set_index(
            "model"
        )
        values = [
            subset.loc[model, "mean_delta_mape"] for model in FOUNDATION_MODEL_ORDER
        ]
        shift = (-width / 2) if idx == 0 else (width / 2)
        ax.bar(
            [x + shift for x in x_positions],
            values,  # type: ignore
            width=width,
            color=INPUT_MODE_COLORS[input_mode],
            label=INPUT_MODE_LABELS[input_mode],
        )

    ax.axhline(0, color="#666666", linewidth=1.1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [FOUNDATION_MODEL_LABELS[model] for model in FOUNDATION_MODEL_ORDER]
    )
    ax.set_ylabel("One-shot minus finetuned (MAPE bodu)")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(frameon=False)

    output_path = output_dir / "foundation_delta_finetuning.png"
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_covariate_delta(
    delta_df: pd.DataFrame,
    mode: str,
    model_order: list[str],
    model_labels: dict[str, str],
    model_colors: dict[str, str],
    output_dir: Path,
    dpi: int,
    filename_prefix: str,
) -> Path:
    subset = delta_df[delta_df["mode"] == mode].set_index("model")
    values = [subset.loc[model, "mean_delta_mape"] for model in model_order]

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    ax.bar(
        [model_labels[model] for model in model_order],
        values,  # type: ignore
        color=[model_colors[model] for model in model_order],
        width=0.62,
    )
    ax.axhline(0, color="#666666", linewidth=1.1)
    ax.set_xlabel("Model")
    ax.set_ylabel("Rozdíl MAPE mezi režimy (%)")
    ax.grid(True, axis="y", alpha=0.22)

    output_path = (
        output_dir / f"{filename_prefix}_delta_covariate_{mode.replace('-', '_')}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_custom_yearly_mape(
    df: pd.DataFrame,
    input_mode: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    plot_df = df[
        (df["mode"] == "finetuned") & (df["training_input_mode"] == input_mode)
    ].copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.8), constrained_layout=True)
    for model in CUSTOM_MODEL_ORDER:
        model_df = plot_df[plot_df["model"] == model].sort_values("test_year")
        ax.plot(
            model_df["test_year"],
            model_df["mape"],
            marker="o",
            linewidth=3.0,
            markersize=7.0,
            color=CUSTOM_MODEL_COLORS[model],
            label=CUSTOM_MODEL_LABELS[model],
        )

    years = sorted(plot_df["test_year"].unique())
    ax.set_xticks(years[::2])
    ax.set_xlabel("Testovací rok")
    ax.set_ylabel("MAPE (%)")
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=1)

    output_path = (
        output_dir / f"custom_mape_by_year_finetuned_10_epochs_{input_mode}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_custom_model_improvement_by_year(
    df: pd.DataFrame,
    baseline_model: str,
    improved_model: str,
    mode: str,
    input_mode: str,
) -> pd.DataFrame:
    subset = df[
        (df["model"].isin([baseline_model, improved_model]))
        & (df["mode"] == mode)
        & (df["training_input_mode"] == input_mode)
    ].copy()
    if subset.empty:
        raise ValueError(
            "No rows found for custom-model improvement comparison: "
            f"baseline={baseline_model}, improved={improved_model}, "
            f"mode={mode}, input_mode={input_mode}"
        )

    pivot = (
        subset.pivot_table(
            index="test_year", columns="model", values="mape", aggfunc="mean"
        )
        .sort_index()
        .dropna(subset=[baseline_model, improved_model])
    )
    if pivot.empty:
        raise ValueError(
            "No overlapping yearly rows found for custom-model improvement comparison: "
            f"baseline={baseline_model}, improved={improved_model}, "
            f"mode={mode}, input_mode={input_mode}"
        )

    comparison_df = pivot.reset_index().rename_axis(columns=None)
    comparison_df["delta_mape"] = (
        comparison_df[baseline_model] - comparison_df[improved_model]
    )
    comparison_df["baseline_model"] = baseline_model
    comparison_df["improved_model"] = improved_model
    comparison_df["mode"] = mode
    comparison_df["training_input_mode"] = input_mode
    comparison_df = comparison_df.rename(
        columns={
            baseline_model: "baseline_mape",
            improved_model: "improved_mape",
        }
    )
    return comparison_df[
        [
            "test_year",
            "baseline_model",
            "improved_model",
            "mode",
            "training_input_mode",
            "baseline_mape",
            "improved_mape",
            "delta_mape",
        ]
    ]


def _plot_custom_model_improvement_by_year(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    filename: str,
) -> Path:
    years = comparison_df["test_year"].tolist()
    values = comparison_df["delta_mape"].tolist()
    colors = ["#54a24b" if value >= 0 else "#e45756" for value in values]

    fig, ax = plt.subplots(figsize=(13.0, 7.0), constrained_layout=True)
    ax.bar(years, values, color=colors, width=0.72)
    ax.axhline(0, color="#666666", linewidth=1.1)
    ax.set_xlabel("Testovací rok")
    ax.set_ylabel("Rozdíl MAPE mezi modely (%)")
    ax.set_xticks(years)
    ax.grid(True, axis="y", alpha=0.22)

    output_path = output_dir / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _write_summary_tables(
    yearly_df: pd.DataFrame,
    finetuning_delta_df: pd.DataFrame,
    covariate_delta_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    yearly_path = output_dir / "foundation_mape_by_year.csv"
    finetuning_path = output_dir / "foundation_delta_finetuning.csv"
    covariate_path = output_dir / "foundation_delta_covariate.csv"

    yearly_df.to_csv(yearly_path, index=False)
    finetuning_delta_df.to_csv(finetuning_path, index=False)
    covariate_delta_df.to_csv(covariate_path, index=False)
    return [yearly_path, finetuning_path, covariate_path]


def _write_custom_summary_tables(
    yearly_df: pd.DataFrame,
    covariate_delta_df: pd.DataFrame,
    comparison_by_year_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    yearly_path = output_dir / "custom_mape_by_year_10_epochs.csv"
    covariate_path = output_dir / "custom_delta_covariate_10_epochs.csv"
    comparison_by_year_path = (
        output_dir / "custom_delta_conv_vs_attention_by_year_10_epochs.csv"
    )

    yearly_df.to_csv(yearly_path, index=False)
    covariate_delta_df.to_csv(covariate_path, index=False)
    comparison_by_year_df.to_csv(comparison_by_year_path, index=False)
    return [yearly_path, covariate_path, comparison_by_year_path]


def _print_delta_summary(
    finetuning_delta_df: pd.DataFrame,
    covariate_delta_df: pd.DataFrame,
) -> None:
    print("Delta finetuning: one-shot minus finetuned")
    for model in FOUNDATION_MODEL_ORDER:
        subset = finetuning_delta_df[finetuning_delta_df["model"] == model].set_index(
            "training_input_mode"
        )
        uni = float(subset.loc["univariate", "mean_delta_mape"])  # type: ignore
        cov = float(subset.loc["covariate", "mean_delta_mape"])  # type: ignore
        print(
            f"  {FOUNDATION_MODEL_LABELS[model]}: "
            f"univariate={uni:.4f}, covariate={cov:.4f}"
        )

    print("Delta covariates: univariate minus covariate")
    for model in FOUNDATION_MODEL_ORDER:
        subset = covariate_delta_df[covariate_delta_df["model"] == model].set_index(
            "mode"
        )
        one_shot = float(subset.loc["one-shot", "mean_delta_mape"])  # type: ignore
        finetuned = float(subset.loc["finetuned", "mean_delta_mape"])  # type: ignore
        print(
            f"  {FOUNDATION_MODEL_LABELS[model]}: one-shot={one_shot:.4f}, "
            f"finetuned={finetuned:.4f}"
        )


def _print_custom_delta_summary(covariate_delta_df: pd.DataFrame) -> None:
    print("Custom delta covariates (10 epochs): univariate minus covariate")
    for model in CUSTOM_MODEL_ORDER:
        subset = covariate_delta_df[covariate_delta_df["model"] == model].set_index(
            "mode"
        )
        finetuned = float(subset.loc["finetuned", "mean_delta_mape"])  # type: ignore
        print(f"  {CUSTOM_MODEL_LABELS[model]}: finetuned={finetuned:.4f}")


def _print_custom_model_improvement_by_year(comparison_df: pd.DataFrame) -> None:
    print("Custom delta by year (10 epochs, covariate): model_2 minus model_3")
    for row in comparison_df.itertuples(index=False):
        print(
            f"  {row.test_year}: "
            f"{CUSTOM_MODEL_LABELS[row.improved_model]} better by {row.delta_mape:.4f} MAPE points"  # type: ignore
        )


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.custom_output_dir.mkdir(parents=True, exist_ok=True)
    _apply_plot_style()

    yearly_df = _read_export_rows(
        args.input_dir,
        allowed_models=FOUNDATION_MODEL_ORDER,
    )
    finetuning_delta_df, covariate_delta_df = _aggregate_deltas(
        yearly_df,
        model_order=FOUNDATION_MODEL_ORDER,
    )
    custom_yearly_df = _read_export_rows(
        args.custom_input_dir,
        allowed_models=CUSTOM_MODEL_ORDER,
        train_epochs="10",
    )
    _, custom_covariate_delta_df = _aggregate_deltas(
        custom_yearly_df,
        model_order=CUSTOM_MODEL_ORDER,
    )
    custom_comparison_by_year_df = _build_custom_model_improvement_by_year(
        custom_yearly_df,
        baseline_model="model_2",
        improved_model="model_3",
        mode="finetuned",
        input_mode="covariate",
    )

    output_paths = [
        _plot_yearly_mape(
            yearly_df,
            mode="one-shot",
            input_mode="univariate",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_yearly_mape(
            yearly_df,
            mode="one-shot",
            input_mode="covariate",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_yearly_mape(
            yearly_df,
            mode="finetuned",
            input_mode="univariate",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_yearly_mape(
            yearly_df,
            mode="finetuned",
            input_mode="covariate",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_finetuning_delta(
            finetuning_delta_df, output_dir=args.output_dir, dpi=args.dpi
        ),
        _plot_covariate_delta(
            covariate_delta_df,
            mode="one-shot",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_covariate_delta(
            covariate_delta_df,
            mode="finetuned",
            model_order=FOUNDATION_MODEL_ORDER,
            model_labels=FOUNDATION_MODEL_LABELS,
            model_colors=FOUNDATION_MODEL_COLORS,
            output_dir=args.output_dir,
            dpi=args.dpi,
            filename_prefix="foundation",
        ),
        _plot_custom_yearly_mape(
            custom_yearly_df,
            input_mode="univariate",
            output_dir=args.custom_output_dir,
            dpi=args.dpi,
        ),
        _plot_custom_yearly_mape(
            custom_yearly_df,
            input_mode="covariate",
            output_dir=args.custom_output_dir,
            dpi=args.dpi,
        ),
        _plot_covariate_delta(
            custom_covariate_delta_df,
            mode="finetuned",
            model_order=CUSTOM_MODEL_ORDER,
            model_labels=CUSTOM_MODEL_LABELS,
            model_colors=CUSTOM_MODEL_COLORS,
            output_dir=args.custom_output_dir,
            dpi=args.dpi,
            filename_prefix="custom_10_epochs",
        ),
        _plot_custom_model_improvement_by_year(
            custom_comparison_by_year_df,
            output_dir=args.custom_output_dir,
            dpi=args.dpi,
            filename="custom_delta_conv_vs_attention_by_year_10_epochs.png",
        ),
    ]
    output_paths.extend(
        _write_summary_tables(
            yearly_df=yearly_df,
            finetuning_delta_df=finetuning_delta_df,
            covariate_delta_df=covariate_delta_df,
            output_dir=args.output_dir,
        )
    )
    output_paths.extend(
        _write_custom_summary_tables(
            yearly_df=custom_yearly_df,
            covariate_delta_df=custom_covariate_delta_df,
            comparison_by_year_df=custom_comparison_by_year_df,
            output_dir=args.custom_output_dir,
        )
    )
    _print_delta_summary(
        finetuning_delta_df=finetuning_delta_df,
        covariate_delta_df=covariate_delta_df,
    )
    _print_custom_delta_summary(custom_covariate_delta_df)
    _print_custom_model_improvement_by_year(custom_comparison_by_year_df)

    for path in output_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
