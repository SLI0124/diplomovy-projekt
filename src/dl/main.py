from __future__ import annotations

import numpy as np
import torch

try:
    from .config import parse_args
    from .dataset import load_dataset
    from .runner import run
except ImportError:
    from config import parse_args
    from dataset import load_dataset
    from runner import run


def _log(message: str) -> None:
    print(f"[main] {message}")


def main() -> None:
    config = parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    _log(f"action={config.action} mode={config.mode} test_year={config.test_year}")
    _log(f"models={config.models}")
    _log(f"dataset_path={config.dataset_path}")

    bundle = load_dataset(config)
    _log(f"resolved_dataset={bundle.dataset_path}")

    if bundle.run_params_path is not None:
        _log(f"preprocessing_run_params={bundle.run_params_path}")

    results = run(config, bundle)

    print("\nPer-fold results (segment=all):")
    print(
        results[results["segment"] == "all"]
        .sort_values(["model", "mode", "test_year"])
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    main()
