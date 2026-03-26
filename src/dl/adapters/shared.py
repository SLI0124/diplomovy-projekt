from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomWindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        prediction_length: int,
        n_samples: int,
    ) -> None:
        self.series = np.asarray(series, dtype=np.float32)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.n_samples = int(n_samples)

        min_total = self.context_length + self.prediction_length
        if len(self.series) < min_total:
            raise ValueError(
                f"Need at least {min_total} points, got {len(self.series)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        max_start = len(self.series) - (self.context_length + self.prediction_length)
        start = int(np.random.randint(0, max_start + 1))

        past = self.series[start : start + self.context_length]
        future = self.series[
            start
            + self.context_length : start
            + self.context_length
            + self.prediction_length
        ]

        return {
            "context": torch.from_numpy(past),
            "future_target": torch.from_numpy(future),
        }


class RandomWindowDatasetWithCovariates(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        covariates: np.ndarray,
        context_length: int,
        prediction_length: int,
        n_samples: int,
        future_covariates: np.ndarray | None = None,
    ) -> None:
        self.series = np.asarray(series, dtype=np.float32)
        self.covariates = np.asarray(covariates, dtype=np.float32)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.n_samples = int(n_samples)
        self.future_covariates = (
            np.asarray(future_covariates, dtype=np.float32)
            if future_covariates is not None
            else self.covariates
        )

        min_total = self.context_length + self.prediction_length
        if len(self.series) < min_total:
            raise ValueError(
                f"Need at least {min_total} points, got {len(self.series)}"
            )
        if len(self.covariates) != len(self.series):
            raise ValueError(
                "Series and covariates length mismatch. "
                f"series={len(self.series)} covariates={len(self.covariates)}"
            )
        if len(self.future_covariates) != len(self.series):
            raise ValueError(
                "Series and future covariates length mismatch. "
                f"series={len(self.series)} future_covariates={len(self.future_covariates)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        max_start = len(self.series) - (self.context_length + self.prediction_length)
        start = int(np.random.randint(0, max_start + 1))

        context_start = start
        context_end = start + self.context_length
        future_end = context_end + self.prediction_length

        past_target = self.series[context_start:context_end]
        future_target = self.series[context_end:future_end]
        past_covariates = self.covariates[context_start:context_end]
        future_covariates = self.future_covariates[context_end:future_end]

        return {
            "context": torch.from_numpy(past_target),
            "future_target": torch.from_numpy(future_target),
            "context_covariates": torch.from_numpy(past_covariates),
            "future_covariates": torch.from_numpy(future_covariates),
        }
