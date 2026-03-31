from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = {"year", "month", "day", "hour", "apparent_temperature"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]].rename(columns={"hour": "hour"}),
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp", "apparent_temperature"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("timestamp")

    return df


def plot_consumption_total_by_year(df: pd.DataFrame) -> None:
    years = sorted(df["year"].dropna().unique().astype(int).tolist())
    if not years:
        raise ValueError("No year information found in dataset.")

    df = df[df["year"].isin(years)].copy()
    if df.empty:
        raise ValueError("No rows found for selected years.")

    cmap = plt.get_cmap("tab20", len(years))

    plt.figure(figsize=(14, 7))
    for idx, year in enumerate(years):
        year_data = df[df["year"] == year]
        if year_data.empty:
            continue
        plt.plot(
            year_data["timestamp"],
            year_data["apparent_temperature"],
            label=str(year),
            color=cmap(idx),
            linewidth=0.9,
            alpha=0.9,
        )

    # Add a global trend line across all data points
    df_all = df.dropna(subset=["timestamp", "apparent_temperature"]).copy()
    x = (
        (df_all["timestamp"] - df_all["timestamp"].min())
        .dt.total_seconds()
        .astype(float)
    )
    y = df_all["apparent_temperature"].astype(float)
    if len(x) > 1:
        coeffs = np.polyfit(x, y, 1)
        trend_line = coeffs[0] * x + coeffs[1]
        plt.plot(
            df_all["timestamp"],
            trend_line,
            label="Trend",
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )

    plt.xlabel("Rok")
    plt.ylabel("Pocitová teplota (°C)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "apparent_temperature_all_years_trend.png"
    plt.savefig(output_path, dpi=150)
    plt.show()


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    csv_path = (
        project_root / "data" / "preprocessed" / "merged_all_years_preprocessed.csv"
    )

    df = load_dataset(csv_path)
    plot_consumption_total_by_year(df)


if __name__ == "__main__":
    main()
