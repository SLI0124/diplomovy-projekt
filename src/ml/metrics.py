from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HourMetrics:
    model: str
    strategy: str
    train_years: str
    test_year: int
    hour: int
    train_samples: int
    test_samples: int
    smape: float
    mape: float
    mae: float
    mse: float
    r2: float


@dataclass(frozen=True)
class FoldMetrics:
    model: str
    strategy: str
    train_years: str
    test_year: int
    segment: str
    n_hours: int
    n_points: int
    smape: float
    mape: float
    mae: float
    mse: float
    r2: float


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) == 0.0, 1.0, np.abs(y_true))
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]

    if y_true.size == 0:
        raise ValueError("No finite predictions available for metric computation.")

    return {
        "smape": smape(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
