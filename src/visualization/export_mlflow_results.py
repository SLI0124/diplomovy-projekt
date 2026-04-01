from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT_OUTPUTS: dict[str, str] = {
    "1": "mlflow_experiment_1_foundation_model.json",
    "2": "mlflow_experiment_2_own_model.json",
}


def _to_iso8601(timestamp_ms: int | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC).isoformat()


def _run_to_dict(run: Any) -> dict[str, Any]:
    info = run.info
    data = run.data

    return {
        "run_id": info.run_id,
        "run_name": data.tags.get("mlflow.runName"),
        "status": info.status,
        "lifecycle_stage": info.lifecycle_stage,
        "artifact_uri": info.artifact_uri,
        "start_time_ms": info.start_time,
        "end_time_ms": info.end_time,
        "start_time_utc": _to_iso8601(info.start_time),
        "end_time_utc": _to_iso8601(info.end_time),
        "metrics": dict(data.metrics),
        "params": dict(data.params),
        "tags": dict(data.tags),
    }


def _get_all_runs(client: MlflowClient, experiment_id: str) -> list[Any]:
    runs: list[Any] = []
    page_token: str | None = None

    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=ViewType.ALL,
            max_results=50_000,
            order_by=["attribute.start_time DESC"],
            page_token=page_token,
        )
        runs.extend(page)

        page_token = page.token
        if page_token is None:
            break

    return runs


def export_experiment_to_json(
    client: MlflowClient,
    experiment_id: str,
    output_path: Path,
) -> None:
    experiment = client.get_experiment(experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment with ID {experiment_id} does not exist.")

    runs = _get_all_runs(client, experiment_id)

    payload = {
        "exported_at_utc": datetime.now(tz=UTC).isoformat(),
        "tracking_uri": mlflow.get_tracking_uri(),
        "experiment": {
            "experiment_id": experiment.experiment_id,
            "name": experiment.name,
            "artifact_location": experiment.artifact_location,
            "lifecycle_stage": experiment.lifecycle_stage,
            "creation_time_ms": experiment.creation_time,
            "last_update_time_ms": experiment.last_update_time,
            "creation_time_utc": _to_iso8601(experiment.creation_time),
            "last_update_time_utc": _to_iso8601(experiment.last_update_time),
            "tags": dict(experiment.tags or {}),
        },
        "run_count": len(runs),
        "runs": [_run_to_dict(run) for run in runs],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Export MLflow experiment results from SQLite tracking DB to JSON files."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=project_root / "data" / "results" / "mlflow.db",
        help="Path to MLflow SQLite database file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "data-exports",
        help="Directory where JSON exports will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = args.db_path.resolve()
    if not db_path.exists():
        raise FileNotFoundError(f"MLflow database file not found: {db_path}")

    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    client = MlflowClient()

    for experiment_id, output_name in DEFAULT_EXPERIMENT_OUTPUTS.items():
        output_path = args.output_dir.resolve() / output_name
        export_experiment_to_json(client, experiment_id, output_path)
        print(f"Exported experiment {experiment_id} to {output_path}")


if __name__ == "__main__":
    main()
