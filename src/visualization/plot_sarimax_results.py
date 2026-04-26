
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.ticker import FuncFormatter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot yearly SARIMAX prediction results from "
            "data/results/sarimax_stepup/predictions_by_year."
        )
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2021, 2022],
        help="Year or years to plot. Defaults to 2021 2022.",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_results_root() -> Path:
    return _project_root() / "data" / "results" / "sarimax_stepup"


def _resolve_output_dir() -> Path:
    output_dir = _project_root() / "data" / "plots" / "sarimax_stepup"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _csv_path_for_year(results_root: Path, year: int) -> Path:
    return results_root / "predictions_by_year" / f"sarimax_predictions_{year}.csv"


def _load_year_predictions(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing SARIMAX predictions file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"timestamp", "y_true", "y_pred"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {', '.join(missing)}"
        )

    ts = df[["timestamp", "y_true", "y_pred"]].copy()
    ts["timestamp"] = pd.to_datetime(ts["timestamp"], errors="coerce")
    ts["y_true"] = pd.to_numeric(ts["y_true"], errors="coerce")
    ts["y_pred"] = pd.to_numeric(ts["y_pred"], errors="coerce")
    ts = ts.dropna(subset=["timestamp", "y_true", "y_pred"])
    ts = ts.sort_values("timestamp")

    if ts.empty:
        raise ValueError(f"No valid rows to plot after cleaning: {csv_path}")

    return ts


def _format_thousands_with_spaces(value: float, _: int) -> str:
    return f"{value:,.0f}".replace(",", " ")


def _negative_prediction_summary(data: pd.DataFrame) -> tuple[int, float]:
    negative_count = int((data["y_pred"] < 0).sum())
    total_count = len(data)
    percentage = (negative_count / total_count * 100.0) if total_count else 0.0
    return negative_count, percentage


def _plot_year(data: pd.DataFrame, year: int, output_dir: Path, dpi: int = 200) -> Path:
    output_path = output_dir / f"sarimax_{year}_skutecnost_vs_predikce.png"

    plt.rcParams.update(
        {
            "font.size": 28,
            "axes.titlesize": 30,
            "axes.labelsize": 28,
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
            "legend.fontsize": 26,
        }
    )
    plt.figure(figsize=(16, 8))
    plt.plot(
        data["timestamp"],
        data["y_true"],
        label="Skutečnost",
    )
    plt.plot(
        data["timestamp"],
        data["y_pred"],
        label="Předikce",
    )
    plt.xlabel("Datum")
    plt.ylabel("Spotreba (MWh)")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FuncFormatter(_format_thousands_with_spaces))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.grid(True, alpha=0.25)
    plt.legend()
    # plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()

    return output_path


def main() -> None:
    args = _parse_args()
    results_root = _resolve_results_root()
    output_dir = _resolve_output_dir()

    for year in args.years:
        csv_path = _csv_path_for_year(results_root, year)
        data = _load_year_predictions(csv_path)
        negative_count, negative_percentage = _negative_prediction_summary(data)
        output_path = _plot_year(
            data=data,
            year=year,
            output_dir=output_dir,
        )
        print(f"Saved plot: {output_path}")
        print(f"source_csv={csv_path}")
        print(
            f"negative_predictions={negative_count} "
            f"({negative_percentage:.2f}% of total)"
        )


if __name__ == "__main__":
    main()
