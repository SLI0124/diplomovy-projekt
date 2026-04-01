from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

YEARS = list(range(2014, 2026))
FOUNDATION_MODELS = ("granite", "chronos2", "moirai1")
MODES = ("finetuned", "oneshot")
TRAINING_INPUT_MODES = ("covariate", "univariate")


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return float(value)


def _fmt(value: float | None) -> float | str:
    return value if value is not None else ""


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _read_csv_rows(path: Path) -> dict[int, dict[str, float | None]]:
    if not path.exists():
        return {}

    by_year: dict[int, dict[str, float | None]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            year_raw = row.get("testing_year")
            if not year_raw:
                continue
            year = int(year_raw)
            by_year[year] = {
                k: _parse_float(v) for k, v in row.items() if k != "testing_year"
            }
    return by_year


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_foundation_mode_comparison(
    input_root: Path,
    output_root: Path,
    training_input_mode: str,
) -> None:
    finetuned_path = (
        input_root / "foundation" / f"foundation_finetuned_{training_input_mode}.csv"
    )
    oneshot_path = (
        input_root / "foundation" / f"foundation_oneshot_{training_input_mode}.csv"
    )

    finetuned = _read_csv_rows(finetuned_path)
    oneshot = _read_csv_rows(oneshot_path)

    rows: list[dict[str, Any]] = []
    for year in YEARS:
        row: dict[str, Any] = {"testing_year": year}
        finetuned_metrics = finetuned.get(year, {})
        oneshot_metrics = oneshot.get(year, {})

        for model in FOUNDATION_MODELS:
            ft_mae = finetuned_metrics.get(f"{model}_mae")
            os_mae = oneshot_metrics.get(f"{model}_mae")
            ft_mape = finetuned_metrics.get(f"{model}_mape")
            os_mape = oneshot_metrics.get(f"{model}_mape")

            row[f"{model}_finetuned_mae"] = _fmt(ft_mae)
            row[f"{model}_oneshot_mae"] = _fmt(os_mae)
            row[f"{model}_delta_mae_finetuned_minus_oneshot"] = _fmt(
                _delta(ft_mae, os_mae)
            )
            row[f"{model}_finetuned_mape"] = _fmt(ft_mape)
            row[f"{model}_oneshot_mape"] = _fmt(os_mape)
            row[f"{model}_delta_mape_finetuned_minus_oneshot"] = _fmt(
                _delta(ft_mape, os_mape)
            )

        rows.append(row)

    fieldnames = ["testing_year"]
    for model in FOUNDATION_MODELS:
        fieldnames.extend(
            [
                f"{model}_finetuned_mae",
                f"{model}_oneshot_mae",
                f"{model}_delta_mae_finetuned_minus_oneshot",
                f"{model}_finetuned_mape",
                f"{model}_oneshot_mape",
                f"{model}_delta_mape_finetuned_minus_oneshot",
            ]
        )

    output_path = (
        output_root
        / "foundation"
        / f"foundation_compare_finetuned_vs_oneshot_{training_input_mode}.csv"
    )
    _write_csv(output_path, fieldnames, rows)
    print(f"[foundation] wrote {output_path}")


def _build_foundation_training_input_comparison(
    input_root: Path,
    output_root: Path,
    mode: str,
) -> None:
    covariate_path = input_root / "foundation" / f"foundation_{mode}_covariate.csv"
    univariate_path = input_root / "foundation" / f"foundation_{mode}_univariate.csv"

    covariate = _read_csv_rows(covariate_path)
    univariate = _read_csv_rows(univariate_path)

    rows: list[dict[str, Any]] = []
    for year in YEARS:
        row: dict[str, Any] = {"testing_year": year}
        covariate_metrics = covariate.get(year, {})
        univariate_metrics = univariate.get(year, {})

        for model in FOUNDATION_MODELS:
            cov_mae = covariate_metrics.get(f"{model}_mae")
            uni_mae = univariate_metrics.get(f"{model}_mae")
            cov_mape = covariate_metrics.get(f"{model}_mape")
            uni_mape = univariate_metrics.get(f"{model}_mape")

            row[f"{model}_covariate_mae"] = _fmt(cov_mae)
            row[f"{model}_univariate_mae"] = _fmt(uni_mae)
            row[f"{model}_delta_mae_covariate_minus_univariate"] = _fmt(
                _delta(cov_mae, uni_mae)
            )
            row[f"{model}_covariate_mape"] = _fmt(cov_mape)
            row[f"{model}_univariate_mape"] = _fmt(uni_mape)
            row[f"{model}_delta_mape_covariate_minus_univariate"] = _fmt(
                _delta(cov_mape, uni_mape)
            )

        rows.append(row)

    fieldnames = ["testing_year"]
    for model in FOUNDATION_MODELS:
        fieldnames.extend(
            [
                f"{model}_covariate_mae",
                f"{model}_univariate_mae",
                f"{model}_delta_mae_covariate_minus_univariate",
                f"{model}_covariate_mape",
                f"{model}_univariate_mape",
                f"{model}_delta_mape_covariate_minus_univariate",
            ]
        )

    output_path = (
        output_root
        / "foundation"
        / f"foundation_compare_covariate_vs_univariate_{mode}.csv"
    )
    _write_csv(output_path, fieldnames, rows)
    print(f"[foundation] wrote {output_path}")


def _build_custom_training_input_comparison(
    input_root: Path,
    output_root: Path,
    mode: str,
) -> None:
    covariate_path = input_root / "custom" / f"custom_{mode}_covariate.csv"
    univariate_path = input_root / "custom" / f"custom_{mode}_univariate.csv"

    covariate = _read_csv_rows(covariate_path)
    univariate = _read_csv_rows(univariate_path)

    rows: list[dict[str, Any]] = []
    for year in YEARS:
        covariate_metrics = covariate.get(year, {})
        univariate_metrics = univariate.get(year, {})

        cov_mae = covariate_metrics.get("custom_mae")
        uni_mae = univariate_metrics.get("custom_mae")
        cov_mape = covariate_metrics.get("custom_mape")
        uni_mape = univariate_metrics.get("custom_mape")

        rows.append(
            {
                "testing_year": year,
                "covariate_mae": _fmt(cov_mae),
                "univariate_mae": _fmt(uni_mae),
                "delta_mae_covariate_minus_univariate": _fmt(_delta(cov_mae, uni_mae)),
                "covariate_mape": _fmt(cov_mape),
                "univariate_mape": _fmt(uni_mape),
                "delta_mape_covariate_minus_univariate": _fmt(
                    _delta(cov_mape, uni_mape)
                ),
            }
        )

    fieldnames = [
        "testing_year",
        "covariate_mae",
        "univariate_mae",
        "delta_mae_covariate_minus_univariate",
        "covariate_mape",
        "univariate_mape",
        "delta_mape_covariate_minus_univariate",
    ]
    output_path = (
        output_root / "custom" / f"custom_compare_covariate_vs_univariate_{mode}.csv"
    )
    _write_csv(output_path, fieldnames, rows)
    print(f"[custom] wrote {output_path}")


def export_comparisons(input_root: Path, output_root: Path) -> None:
    for training_input_mode in TRAINING_INPUT_MODES:
        _build_foundation_mode_comparison(
            input_root=input_root,
            output_root=output_root,
            training_input_mode=training_input_mode,
        )

    for mode in MODES:
        _build_foundation_training_input_comparison(
            input_root=input_root,
            output_root=output_root,
            mode=mode,
        )

    # Keep custom outputs fixed to two comparison files (finetuned + oneshot).
    for mode in MODES:
        _build_custom_training_input_comparison(
            input_root=input_root,
            output_root=output_root,
            mode=mode,
        )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create comparison CSV tables from aggregated MLflow table snapshots."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "data" / "data-exports" / "mlflow_tables_csv",
        help="Root folder containing generated foundation/custom MLflow table CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root
        / "data"
        / "data-exports"
        / "mlflow_tables_comparisons_csv",
        help="Output folder for generated comparison CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_comparisons(input_root=args.input_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
