from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoldMetrics:
    model: str
    mode: str
    train_years: str
    test_year: int
    segment: str
    n_windows: int
    n_points: int
    smape: float
    mape: float
    mae: float
    mse: float
    r2: float


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
