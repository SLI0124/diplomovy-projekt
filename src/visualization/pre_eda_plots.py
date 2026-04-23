from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "figure.titlesize": 20,
    }
)

source_file = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "merged"
    / "merged_all_years.csv"
)
target_col = "consumption_vcpnet"
timestamp_col = "timestamp"

if not source_file.exists():
    raise FileNotFoundError(f"Source file not found: {source_file}")

df = pd.read_csv(source_file)
required_cols = [
    "year",
    "month",
    "day",
    "hour",
    target_col,
]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing required columns for plotting: {missing_cols}")

df[timestamp_col] = pd.to_datetime(
    df[["year", "month", "day", "hour"]], errors="coerce"
)
df = df.dropna(subset=[timestamp_col])

start_ts = pd.Timestamp("2016-01-01")
end_ts = pd.Timestamp("2016-12-31")
consumption_df = df.dropna(subset=[target_col])
consumption_df = consumption_df[
    (consumption_df[timestamp_col] >= start_ts)
    & (consumption_df[timestamp_col] < end_ts)
]
consumption_df = consumption_df.sort_values(timestamp_col)

if consumption_df.empty:
    raise ValueError("No rows available in selected period: 2014-2018")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    consumption_df[timestamp_col],
    consumption_df[target_col],
    linewidth=0.8,
    label=target_col,
)
consumption_outliers = consumption_df[target_col] > 800_000
if consumption_outliers.any():
    ax.scatter(
        consumption_df.loc[consumption_outliers, timestamp_col],
        consumption_df.loc[consumption_outliers, target_col],
        color="red",
        s=75,
        label="Odlehlá pozorování",
        zorder=3,
    )
    ax.legend()
ax.ticklabel_format(style="plain", axis="y", useOffset=False)
ax.set_xlabel("Datum")
ax.set_ylabel("Spotřeba (MWh)")
plt.tight_layout()
plt.savefig("../../data/plots/pre_eda_consumption_vcpnet_2014_2018.png", dpi=300)
plt.show()
