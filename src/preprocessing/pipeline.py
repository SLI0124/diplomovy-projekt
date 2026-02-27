from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


@dataclass
class PreprocessConfig:
    input_csv: Path
    output_csv: Path
    war_start: str = "2022-02-24"
    target_col: str = "consumption_total"
    random_state: int = 42


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dt_values = pd.to_datetime(out["datetime"], errors="coerce")
    dt_index = pd.DatetimeIndex(dt_values)

    hour_arr = np.asarray(out["hour"], dtype=float)
    dow_arr = np.asarray(out["day_of_week"], dtype=float)
    month_arr = np.asarray(out["month"], dtype=float)
    day_arr = np.asarray(out["day"], dtype=float)
    week_arr = np.asarray(dt_index.strftime("%V"), dtype=float)

    out["day_of_year"] = np.asarray(dt_index.strftime("%j"), dtype=float)
    out["week_of_year"] = week_arr
    out["is_weekend"] = np.where(dow_arr >= 5, 1, 0).astype(int)

    out["hour_sin"] = np.sin((2.0 * np.pi * hour_arr) / 24.0)
    out["hour_cos"] = np.cos((2.0 * np.pi * hour_arr) / 24.0)
    out["day_of_week_sin"] = np.sin((2.0 * np.pi * dow_arr) / 7.0)
    out["day_of_week_cos"] = np.cos((2.0 * np.pi * dow_arr) / 7.0)
    out["month_sin"] = np.sin((2.0 * np.pi * (month_arr - 1.0)) / 12.0)
    out["month_cos"] = np.cos((2.0 * np.pi * (month_arr - 1.0)) / 12.0)
    out["day_sin"] = np.sin((2.0 * np.pi * (day_arr - 1.0)) / 31.0)
    out["day_cos"] = np.cos((2.0 * np.pi * (day_arr - 1.0)) / 31.0)
    return out


def add_target_surrounding_features(frame: pd.DataFrame, col: str) -> pd.DataFrame:
    out = frame.copy()
    out[f"{col}_lag_1"] = out[col].shift(1)
    out[f"{col}_lag_24"] = out[col].shift(24)
    out[f"{col}_lag_168"] = out[col].shift(168)
    out[f"{col}_lead_1"] = out[col].shift(-1)
    out[f"{col}_lead_24"] = out[col].shift(-24)
    out[f"{col}_lead_168"] = out[col].shift(-168)
    out[f"{col}_roll_24_mean"] = out[col].shift(1).rolling(24, min_periods=1).mean()
    out[f"{col}_roll_168_mean"] = out[col].shift(1).rolling(168, min_periods=1).mean()
    return out


def fit_decision_tree_imputer(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    random_state: int,
) -> tuple[SimpleImputer, DecisionTreeRegressor, float]:
    model = DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=20,
        random_state=random_state,
    )

    imputer = SimpleImputer(strategy="median")
    x_train_imp = imputer.fit_transform(x_train)
    x_val_imp = imputer.transform(x_val)

    model.fit(x_train_imp, y_train)
    pred = model.predict(x_val_imp)
    mae = mean_absolute_error(y_val, pred)
    return imputer, model, float(mae)


def impute_single_column(
    frame: pd.DataFrame,
    target_col: str,
    random_state: int,
) -> tuple[pd.Series, dict]:
    work = add_target_surrounding_features(frame, target_col)

    base_feature_cols = [
        c
        for c in work.columns
        if c not in [target_col, "datetime", "row_id"]
        and pd.api.types.is_numeric_dtype(work[c])
    ]
    usable_feature_cols = [c for c in base_feature_cols if work[c].notna().any()]

    train_mask = work[target_col].notna()
    miss_mask = work[target_col].isna()

    if train_mask.sum() < 500 or miss_mask.sum() == 0 or len(usable_feature_cols) == 0:
        return frame[target_col], {
            "column": target_col,
            "model": "fallback_ffill_bfill",
            "masked_mae": np.nan,
            "n_missing_filled": int(miss_mask.sum()),
            "n_train": int(train_mask.sum()),
            "n_features_used": int(len(usable_feature_cols)),
            "n_features_dropped_all_nan": int(
                len(base_feature_cols) - len(usable_feature_cols)
            ),
        }

    train_idx = work.index[train_mask]
    val_n = min(4000, max(300, int(0.2 * len(train_idx))))
    val_idx = pd.Index(
        np.random.choice(np.asarray(train_idx), size=val_n, replace=False)
    )
    fit_idx = train_idx.difference(val_idx)

    x_train = work.loc[fit_idx, usable_feature_cols]
    y_train = work.loc[fit_idx, target_col]
    x_val = work.loc[val_idx, usable_feature_cols]
    y_val = work.loc[val_idx, target_col]

    imputer, tree, masked_mae = fit_decision_tree_imputer(
        x_train,
        y_train,
        x_val,
        y_val,
        random_state,
    )

    x_all_train = work.loc[train_mask, usable_feature_cols]
    y_all_train = work.loc[train_mask, target_col]
    x_all_train_imp = imputer.fit_transform(x_all_train)
    tree.fit(x_all_train_imp, y_all_train)

    out_series = frame[target_col].copy()
    if miss_mask.sum():
        x_missing = work.loc[miss_mask, usable_feature_cols]
        x_missing_imp = imputer.transform(x_missing)
        pred_missing = tree.predict(x_missing_imp)
        out_series.loc[miss_mask] = pred_missing

    lag24 = work[f"{target_col}_lag_24"]
    lag168 = work[f"{target_col}_lag_168"]
    lead24 = work[f"{target_col}_lead_24"]
    out_series = out_series.fillna(lag24).fillna(lead24).fillna(lag168).ffill().bfill()

    report = {
        "column": target_col,
        "model": "decision_tree_surroundings",
        "masked_mae": round(masked_mae, 4),
        "n_missing_filled": int(miss_mask.sum()),
        "n_train": int(train_mask.sum()),
        "n_features_used": int(len(usable_feature_cols)),
        "n_features_dropped_all_nan": int(
            len(base_feature_cols) - len(usable_feature_cols)
        ),
    }
    return out_series, report


def run_preprocessing(config: PreprocessConfig) -> dict:
    np.random.seed(config.random_state)

    if not config.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {config.input_csv}"
            + "\nPlease run the merging step to create the merged dataset."
        )

    df = pd.read_csv(config.input_csv)
    war_start = pd.Timestamp(config.war_start)

    df_work = df.copy()
    df_work["datetime"] = pd.to_datetime(
        df_work[["year", "month", "day", "hour"]],
        errors="coerce",
    )
    df_work = df_work.sort_values("datetime").reset_index(drop=True)
    df_work["row_id"] = np.arange(len(df_work), dtype=int)
    df_work["is_war_period"] = (df_work["datetime"] >= war_start).astype(int)
    print(f"Loaded {len(df_work):,} rows")

    consumption_cols = [c for c in df_work.columns if c.startswith("consumption_")]
    component_consumption_cols = [
        c
        for c in consumption_cols
        if c != config.target_col and c.startswith("consumption_")
    ]
    numeric_cols = [
        c
        for c in df_work.select_dtypes(include=[np.number]).columns
        if c not in ["row_id"]
    ]

    df_prepared = add_time_features(df_work.copy())
    print("Time features created")

    imputation_candidates = [
        c
        for c in numeric_cols
        if c != config.target_col and df_prepared[c].isna().any()
    ]
    imputation_report_rows: list[dict] = []
    correction_report_rows: list[dict] = []

    for col in component_consumption_cols:
        if col not in df_prepared.columns:
            continue

        series = df_prepared[col]
        neg_mask = series < 0
        abs_diff = series.diff().abs()
        diff_thr = abs_diff.quantile(0.999)
        val_thr = series.abs().quantile(0.999)
        spike_mask = (abs_diff > 15 * diff_thr) & (series.abs() > 8 * val_thr)

        impossible_mask = (neg_mask | spike_mask.fillna(False)) & series.notna()

        n_invalid = int(impossible_mask.sum())
        if n_invalid > 0:
            df_prepared.loc[impossible_mask, col] = np.nan

        correction_report_rows.append(
            {
                "column": col,
                "negative_count": int(neg_mask.sum()),
                "spike_count": int(spike_mask.fillna(False).sum()),
                "corrected_to_nan": n_invalid,
            }
        )

    imputation_candidates = sorted(
        set(
            imputation_candidates
            + [r["column"] for r in correction_report_rows if r["corrected_to_nan"] > 0]
        )
    )
    total_corrected = int(sum(r["corrected_to_nan"] for r in correction_report_rows))
    print(f"Anomaly correction: {total_corrected:,} values set to NaN")

    date_cols_to_encode = ["month", "day", "hour", "day_of_week"]
    encoded_date_cols = [
        "month_sin",
        "month_cos",
        "day_sin",
        "day_cos",
        "hour_sin",
        "hour_cos",
        "day_of_week_sin",
        "day_of_week_cos",
    ]
    export_cols = [c for c in df.columns if c not in date_cols_to_encode] + [
        c for c in encoded_date_cols if c in df_prepared.columns
    ]
    total_missing_before = int(df_prepared[export_cols].isna().sum().sum())
    print(f"Total missing before imputation: {total_missing_before:,}")
    print(f"Imputing {len(imputation_candidates)} columns with missing values")

    for col in imputation_candidates:
        imputed_series, rep = impute_single_column(
            df_prepared, col, config.random_state
        )
        df_prepared[col] = imputed_series
        imputation_report_rows.append(rep)

    df_prepared[config.target_col] = df_prepared[component_consumption_cols].sum(
        axis=1,
        min_count=1,
    )

    total_missing_after = int(df_prepared[export_cols].isna().sum().sum())
    print(f"Total missing after imputation: {total_missing_after:,}")

    imputation_report = (
        pd.DataFrame(imputation_report_rows)
        .sort_values(["n_missing_filled", "masked_mae"], ascending=[False, True])
        .reset_index(drop=True)
    )
    correction_report = (
        pd.DataFrame(correction_report_rows)
        .sort_values("corrected_to_nan", ascending=False)
        .reset_index(drop=True)
    )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_export = df_prepared[export_cols].copy()

    cleaned_path = config.output_csv
    df_export.to_csv(cleaned_path, index=False)

    return {
        "cleaned_path": str(cleaned_path),
        "rows": int(len(df_export)),
        "target_missing_after": int(df_export[config.target_col].isna().sum()),
        "imputed_columns": int(len(imputation_report)),
        "corrected_columns": int(len(correction_report)),
    }
