from __future__ import annotations

import numpy as np
from config import parse_args
from dataset import load_dataset
from runner import run


def _log(message: str) -> None:
    print(f"[main] {message}")


def main() -> None:
    config = parse_args()

    np.random.seed(config.seed)

    _log(
        f"action={config.action} strategy={config.strategy} test_year={config.test_year}"
    )
    _log(f"models={config.models}")
    _log(f"dataset_path={config.dataset_path}")

    bundle = load_dataset(config)
    _log(f"resolved_dataset={bundle.dataset_path}")
    _log(f"feature_count={len(bundle.feature_columns)}")

    if bundle.run_params_path is not None:
        _log(f"preprocessing_run_params={bundle.run_params_path}")

    hour_df, fold_df = run(config, bundle)

    print("\nFold summary (segment=all):")
    if fold_df.empty:
        print("No evaluation metrics produced (train-only run).")
        return

    print(
        fold_df[fold_df["segment"] == "all"]
        .sort_values(["model", "test_year"])
        .reset_index(drop=True)
    )
    print(f"\nHour metric rows: {len(hour_df)}")


if __name__ == "__main__":
    main()
