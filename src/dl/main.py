from __future__ import annotations

import numpy as np
import torch
from config import parse_args
from dataset import load_dataset
from runner import run


def _log(message: str) -> None:
    print(f"[main] {message}")


def main() -> None:
    config = parse_args()

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    _log(f"action={config.action} mode={config.mode} test_year={config.test_year}")
    _log(f"models={config.models}")
    _log(f"variant_stem={config.variant_stem}")

    bundle = load_dataset(config)
    _log(f"resolved_split_root={bundle.split_root}")

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
