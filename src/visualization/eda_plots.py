from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

FONT_SCALE = 1.6
LABEL_FONT_SIZE = int(15 * FONT_SCALE)
TICK_FONT_SIZE = int(13 * FONT_SCALE)
LEGEND_FONT_SIZE = int(13 * FONT_SCALE)


def nacti_a_priprav_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(
        df[["year", "month", "day", "hour"]], errors="coerce"
    )
    df = df.sort_values("datetime").reset_index(drop=True)

    print(f"Loaded: {csv_path}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Range: {df['datetime'].min()} -> {df['datetime'].max()}")

    return df


def plot_eda_consumption_over_time(df: pd.DataFrame, save_path: Path) -> None:
    daily = (
        df.set_index("datetime")["consumption_total"]
        .abs()
        .resample("D")
        .mean()
        .dropna()
    )
    daily_x = daily.index.to_numpy()
    daily_y = daily.to_numpy(dtype=float)

    _, ax = plt.subplots(figsize=(18, 8))
    ax.plot(
        daily_x,
        daily_y,
        linewidth=2.2,
        alpha=0.95,
    )

    ax.set_xlabel("Datum", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Celková spotřeba (kWh)", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="y", style="plain")
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    plt.tight_layout()
    plt.savefig(save_path / "eda_consumption_over_time.png", dpi=300)
    plt.show()


def plot_boxplot_spotreba_total(df: pd.DataFrame, save_path: Path) -> None:
    spotreba_abs = df["consumption_total"].abs().dropna()

    _, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(x=spotreba_abs, ax=ax)

    ax.set_xlabel("Celková spotřeba (kWh)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    plt.tight_layout()
    plt.savefig(save_path / "eda_boxplot_spotreba_total_absolutni.png", dpi=300)
    plt.show()


def plot_distribuce_spotreba_total(df: pd.DataFrame, save_path: Path) -> None:
    absolute_consumption = df["consumption_total"].abs().dropna()
    absolute_consumption_array = absolute_consumption.to_numpy(dtype=float)

    _, ax = plt.subplots(figsize=(14, 7))
    sns.histplot(x=absolute_consumption_array, bins=60, kde=True, ax=ax)

    ax.set_xlabel("Celková spotřeba (kWh)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Četnost", fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    plt.tight_layout()
    plt.savefig(save_path / "eda_distribuce_spotreba_total_absolutni.png", dpi=300)
    plt.show()


def plot_mesicni_spotreba_a_teplota(df: pd.DataFrame, save_path: Path) -> None:

    monthly_consumption_temperature = (
        df.groupby("month", observed=True)
        .agg(
            spotreba_prumer=("consumption_total", lambda s: s.abs().mean()),
            teplota_prumer=("temperature_2m", lambda s: s.abs().mean()),
        )
        .reindex(range(1, 13))
    )

    _, ax1 = plt.subplots(figsize=(18, 8))
    ax1.bar(
        monthly_consumption_temperature.index,
        monthly_consumption_temperature["spotreba_prumer"],
        alpha=0.9,
        label="Průměrná celková spotřeba",
    )
    ax1.set_xlabel("Měsíc", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("Průměrná celková spotřeba (kWh)", fontsize=LABEL_FONT_SIZE)
    ax1.tick_params(axis="both", labelsize=TICK_FONT_SIZE)
    ax1.grid(alpha=0.25)
    ax1.ticklabel_format(axis="y", style="plain")
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )

    ax1.set_xticks(
        range(1, 13),
        [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
        ],
    )

    ax2 = ax1.twinx()
    ax2.plot(
        monthly_consumption_temperature.index,
        monthly_consumption_temperature["teplota_prumer"],
        color="#d61b1b",
        linewidth=3,
        marker="o",
        label="Průměrná teplota",
    )
    ax2.set_ylabel("Průměrná teplota (°C)", fontsize=LABEL_FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    handles_left, labels_left = ax1.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()
    plt.savefig(save_path / "eda_mesicni_spotreba_teplota_absolutni.png", dpi=300)
    plt.show()


def plot_all(csv_path: Path, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    df = nacti_a_priprav_data(csv_path)

    plot_eda_consumption_over_time(df, save_path)
    plot_boxplot_spotreba_total(df, save_path)
    plot_distribuce_spotreba_total(df, save_path)
    plot_mesicni_spotreba_a_teplota(df, save_path)


if __name__ == "__main__":
    plot_all(
        csv_path=Path("../../data/preprocessed/merged_all_years_preprocessed.csv"),
        save_path=Path("../../data/plots"),
    )
