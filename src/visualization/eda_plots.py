from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

FONT_SCALE = 1.6
LABEL_FONT_SIZE = int(15 * FONT_SCALE)
TICK_FONT_SIZE = int(13 * FONT_SCALE)
LEGEND_FONT_SIZE = int(13 * FONT_SCALE)
DEFAULT_BLUE = "#1f77b4"
ACCENT_RED = "#d62728"
TEXT_COLOR = "#000000"


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
    ax.set_ylabel("Celková spotřeba (MWh)", fontsize=LABEL_FONT_SIZE)
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

    bigger_font_size = 1.2
    ax.set_xlabel("Celková spotřeba (MWh)", fontsize=LABEL_FONT_SIZE * bigger_font_size)
    ax.set_ylabel("", fontsize=LABEL_FONT_SIZE * bigger_font_size)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE * bigger_font_size)
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

    font_size_multiplier = 1.2
    ax.set_xlabel("Celková spotřeba (MWh)", fontsize=LABEL_FONT_SIZE * font_size_multiplier)
    ax.set_ylabel("Četnost", fontsize=LABEL_FONT_SIZE * font_size_multiplier)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE * font_size_multiplier)
    ax.grid(alpha=0.25)
    ax.ticklabel_format(axis="x", style="plain")
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " "))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:,.0f}".replace(",", " "))
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
    ax1.set_ylabel("Průměrná celková spotřeba (MWh)", fontsize=LABEL_FONT_SIZE)
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


def _set_black_text_axes(ax) -> None:
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE, colors=TEXT_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)

def _prepare_daily_prices(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    daily_price = (
        df.set_index("datetime")[columns].resample("D").first().dropna().reset_index()
    )
    daily_price["year"] = daily_price["datetime"].dt.year
    return daily_price


def plot_mezirocni_trend_spotreba_a_cena(df: pd.DataFrame, save_path: Path) -> None:
    annual_summary = (
        df.groupby("year", observed=True)
        .agg(
            avg_consumption=("consumption_total", "mean"),
            avg_price=("weighted_avg_price_eur_mwh", "mean"),
        )
        .reset_index()
    )

    _, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(
        annual_summary["year"],
        annual_summary["avg_consumption"],
        color=DEFAULT_BLUE,
        marker="o",
        linewidth=3,
        label="Průměrná spotřeba",
    )
    ax1.set_xlabel("Rok", fontsize=LABEL_FONT_SIZE)
    ax1.set_ylabel("Průměrná spotřeba (MWh)", fontsize=LABEL_FONT_SIZE)
    ax1.grid(alpha=0.25)
    ax1.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y:,.0f}".replace(",", " "))
    )
    _set_black_text_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        annual_summary["year"],
        annual_summary["avg_price"],
        color=ACCENT_RED,
        marker="s",
        linewidth=3,
        label="Průměrná cena",
    )
    ax2.set_ylabel(
        "Průměrná cena (EUR/MWh)", fontsize=LABEL_FONT_SIZE, color=TEXT_COLOR
    )
    ax2.tick_params(axis="y", labelsize=TICK_FONT_SIZE, colors=TEXT_COLOR)

    handles_left, labels_left = ax1.get_legend_handles_labels()
    handles_right, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()
    plt.savefig(save_path / "eda_mezirocni_trend_spotreba_a_cena.png", dpi=300)
    plt.show()


def plot_vazena_cena_plynu_2020_2025(df: pd.DataFrame, save_path: Path) -> None:
    daily_price = _prepare_daily_prices(df, ["weighted_avg_price_eur_mwh"])
    daily_price_window = daily_price[
        (daily_price["year"] >= 2020) & (daily_price["year"] <= 2025)
    ].copy()

    if daily_price_window.empty:
        print("No daily gas price data available for years 2020-2025.")
        return

    daily_price_window["price_30d"] = (
        daily_price_window["weighted_avg_price_eur_mwh"]
        .rolling(30, min_periods=1)
        .mean()
    )

    _, ax = plt.subplots(figsize=(17, 7))
    ax.plot(
        daily_price_window["datetime"],
        daily_price_window["weighted_avg_price_eur_mwh"],
        color=DEFAULT_BLUE,
        alpha=0.4,
        linewidth=1.5,
        label="Denní hodnota",
    )
    ax.plot(
        daily_price_window["datetime"],
        daily_price_window["price_30d"],
        color=ACCENT_RED,
        linewidth=3,
        label="30denní průměr",
    )

    ax.set_xlabel("Datum", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Cena (EUR/MWh)", fontsize=LABEL_FONT_SIZE)
    ax.grid(alpha=0.25)
    _set_black_text_axes(ax)
    ax.legend(loc="upper left", fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(save_path / "eda_vazena_cena_plynu_2020_2025.png", dpi=300)
    plt.show()


def plot_prumerny_denni_cenovy_rozptyl_podle_roku(
    df: pd.DataFrame, save_path: Path
) -> None:
    daily_price = _prepare_daily_prices(df, ["min_price_eur_mwh", "max_price_eur_mwh"])
    price_spread_year = (
        (daily_price["max_price_eur_mwh"] - daily_price["min_price_eur_mwh"])
        .groupby(daily_price["year"])
        .mean()
    )

    colors = [
        ACCENT_RED if y == 2022 else DEFAULT_BLUE for y in price_spread_year.index
    ]

    _, ax = plt.subplots(figsize=(14, 5))
    ax.bar(
        price_spread_year.index.astype(str),
        price_spread_year.to_numpy(dtype=float),
        color=colors,
        alpha=0.9,
    )
    ax.set_xlabel("Rok", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Průměrný rozptyl (EUR/MWh)", fontsize=LABEL_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    _set_black_text_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path / "eda_prumerny_denni_cenovy_rozptyl_podle_roku.png", dpi=300)
    plt.show()


def plot_prumerna_cena_plynu_podle_roku(df: pd.DataFrame, save_path: Path) -> None:
    daily_price = _prepare_daily_prices(df, ["weighted_avg_price_eur_mwh"])
    yearly_avg_price = daily_price.groupby("year", observed=True)[
        "weighted_avg_price_eur_mwh"
    ].mean()

    bar_colors = [
        ACCENT_RED if y == 2022 else DEFAULT_BLUE for y in yearly_avg_price.index
    ]

    _, ax = plt.subplots(figsize=(14, 6))
    ax.bar(
        yearly_avg_price.index.astype(str),
        yearly_avg_price.to_numpy(dtype=float),
        color=bar_colors,
        alpha=0.95,
    )
    ax.set_xlabel("Rok", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Průměrná cena (EUR/MWh)", fontsize=LABEL_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    _set_black_text_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path / "eda_prumerna_cena_plynu_podle_roku.png", dpi=300)
    plt.show()


def plot_distribuce_ceny_plynu_podle_roku(df: pd.DataFrame, save_path: Path) -> None:
    daily_price = _prepare_daily_prices(df, ["weighted_avg_price_eur_mwh"])

    _, ax = plt.subplots(figsize=(15, 7))
    sns.boxplot(
        data=daily_price,
        x="year",
        y="weighted_avg_price_eur_mwh",
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("Rok", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel("Vážená průměrná cena (EUR/MWh)", fontsize=LABEL_FONT_SIZE)
    ax.grid(axis="y", alpha=0.25)
    _set_black_text_axes(ax)

    plt.tight_layout()
    plt.savefig(save_path / "eda_distribuce_ceny_plynu_podle_roku.png", dpi=300)
    plt.show()


def plot_all(csv_path: Path, save_path: Path) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    df = nacti_a_priprav_data(csv_path)

    plot_eda_consumption_over_time(df, save_path)
    plot_boxplot_spotreba_total(df, save_path)
    plot_distribuce_spotreba_total(df, save_path)
    plot_mesicni_spotreba_a_teplota(df, save_path)
    plot_mezirocni_trend_spotreba_a_cena(df, save_path)
    plot_vazena_cena_plynu_2020_2025(df, save_path)
    # plot_prumerny_denni_cenovy_rozptyl_podle_roku(df, save_path)
    # plot_prumerna_cena_plynu_podle_roku(df, save_path)
    plot_distribuce_ceny_plynu_podle_roku(df, save_path)


if __name__ == "__main__":
    plot_all(
        csv_path=Path("../../data/preprocessed/merged_all_years_preprocessed.csv"),
        save_path=Path("../../data/plots"),
    )
