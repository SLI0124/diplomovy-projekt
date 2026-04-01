from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

YEARS = list(range(2014, 2026))
FOUNDATION_MODELS = ("granite", "chronos2", "moirai1")


@dataclass(frozen=True)
class RunsFileMeta:
    experiment: str
    mode: str
    training_input_mode: str
    model: str | None


def _extract_prefixed_segment(parts: list[str], prefix: str) -> str | None:
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _normalize_mode_name(mode: str) -> str:
    return mode.replace("one-shot", "oneshot")


def _parse_runs_file_meta(file_path: Path, input_root: Path) -> RunsFileMeta:
    rel_parts = list(file_path.relative_to(input_root).parts)

    experiment = "foundation" if "foundation" in rel_parts else "custom"
    mode = _extract_prefixed_segment(rel_parts, "mode-")
    if mode is None:
        raise ValueError(f"Cannot resolve mode from path: {file_path}")

    training_input_mode = _extract_prefixed_segment(rel_parts, "training_input_mode-")
    if training_input_mode is None:
        raise ValueError(f"Cannot resolve training_input_mode from path: {file_path}")

    model = _extract_prefixed_segment(rel_parts, "model-")
    return RunsFileMeta(
        experiment=experiment,
        mode=mode,
        training_input_mode=training_input_mode,
        model=model,
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_year_metrics(payload: dict[str, Any]) -> dict[int, tuple[float, float]]:
    by_year: dict[int, tuple[float, float]] = {}
    for run in payload.get("runs", []):
        params = run.get("params", {})
        metrics = run.get("metrics", {})

        test_year_raw = params.get("test_year")
        mae_raw = metrics.get("all.mae")
        mape_raw = metrics.get("all.mape")
        if test_year_raw is None or mae_raw is None or mape_raw is None:
            continue

        year = int(test_year_raw)
        by_year[year] = (float(mae_raw), float(mape_raw))
    return by_year


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_tables(input_root: Path, output_root: Path) -> None:
    runs_files = sorted(input_root.glob("**/runs.json"))
    if not runs_files:
        raise FileNotFoundError(f"No runs.json files found under: {input_root}")

    foundation_data: dict[
        tuple[str, str], dict[str, dict[int, tuple[float, float]]]
    ] = {}
    custom_data: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}

    for runs_file in runs_files:
        meta = _parse_runs_file_meta(runs_file, input_root)
        payload = _load_json(runs_file)
        year_metrics = _extract_year_metrics(payload)

        combo_key = (meta.mode, meta.training_input_mode)

        if meta.experiment == "foundation":
            if meta.model is None:
                raise ValueError(
                    f"Foundation file is missing model in path: {runs_file}"
                )
            if meta.model not in FOUNDATION_MODELS:
                continue
            foundation_data.setdefault(combo_key, {})[meta.model] = year_metrics
            continue

        custom_data[combo_key] = year_metrics

    for (mode, training_input_mode), model_payloads in sorted(foundation_data.items()):
        rows: list[dict[str, Any]] = []
        for year in YEARS:
            row: dict[str, Any] = {"testing_year": year}
            for model in FOUNDATION_MODELS:
                metrics = model_payloads.get(model, {}).get(year)
                row[f"{model}_mae"] = metrics[0] if metrics else ""
                row[f"{model}_mape"] = metrics[1] if metrics else ""
            rows.append(row)

        fieldnames = ["testing_year"]
        for model in FOUNDATION_MODELS:
            fieldnames.append(f"{model}_mae")
            fieldnames.append(f"{model}_mape")

        output_name = (
            f"foundation_{_normalize_mode_name(mode)}_{training_input_mode}.csv"
        )
        output_path = output_root / "foundation" / output_name
        _write_csv(output_path, fieldnames, rows)
        print(f"[foundation] wrote {output_path}")

    for (mode, training_input_mode), year_metrics in sorted(custom_data.items()):
        rows = []
        for year in YEARS:
            metrics = year_metrics.get(year)
            rows.append(
                {
                    "testing_year": year,
                    "custom_mae": metrics[0] if metrics else "",
                    "custom_mape": metrics[1] if metrics else "",
                }
            )

        fieldnames = ["testing_year", "custom_mae", "custom_mape"]
        output_name = f"custom_{_normalize_mode_name(mode)}_{training_input_mode}.csv"
        output_path = output_root / "custom" / output_name
        _write_csv(output_path, fieldnames, rows)
        print(f"[custom] wrote {output_path}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Create aggregated CSV tables from exported MLflow runs.json snapshots."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=project_root / "data" / "data-exports" / "mlflow_exports",
        help="Root folder containing foundation/custom MLflow runs.json exports.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "data-exports" / "mlflow_tables_csv",
        help="Output folder for generated CSV tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_tables(input_root=args.input_root, output_root=args.output_root)


if __name__ == "__main__":
    main()
