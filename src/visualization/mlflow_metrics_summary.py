from __future__ import annotations

import argparse
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Run, ViewType
from mlflow.tracking import MlflowClient


@dataclass(frozen=True)
class FilterSpec:
    """Describes how to resolve a filter value from a run payload."""

    aliases: tuple[str, ...]
    extractors: tuple[Callable[[dict[str, Any]], Any], ...]


def _from_params(key: str) -> Callable[[dict[str, Any]], Any]:
    return lambda payload: payload.get("params", {}).get(key)


def _from_tags(key: str) -> Callable[[dict[str, Any]], Any]:
    return lambda payload: payload.get("tags", {}).get(key)


def _from_top_level(key: str) -> Callable[[dict[str, Any]], Any]:
    return lambda payload: payload.get(key)


FILTER_REGISTRY: dict[str, FilterSpec] = {
    "action": FilterSpec(
        aliases=("action",),
        extractors=(_from_params("action"),),
    ),
    "mode": FilterSpec(
        aliases=("mode",),
        extractors=(_from_params("mode"),),
    ),
    "training_input_mode": FilterSpec(
        aliases=("training_input_mode", "training-input-mode"),
        extractors=(
            _from_params("training_input_mode"),
            _from_tags("run.training_input_mode"),
        ),
    ),
    "stem": FilterSpec(
        aliases=("stem", "steam", "variant_stem", "variant-stem"),
        extractors=(
            _from_params("variant_stem"),
            _from_params("dataset_tag"),
            _from_tags("run.dataset.tag"),
        ),
    ),
    "model": FilterSpec(
        aliases=("model",),
        extractors=(
            _from_params("model"),
            _from_tags("run.model.resolved"),
            _from_tags("run.model.requested"),
            lambda payload: str(payload.get("run_name", "")).split("__", 1)[0] or None,
        ),
    ),
    "status": FilterSpec(
        aliases=("status",),
        extractors=(_from_top_level("status"),),
    ),
    "train_epochs": FilterSpec(
        aliases=("train_epochs", "train-epochs", "epochs"),
        extractors=(_from_params("train_epochs"),),
    ),
}

EXPERIMENT_MAP = {
    "foundation": "1",
    "custom": "2",
}


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip()


def _normalize_for_compare(value: Any) -> str | None:
    txt = _coerce_text(value)
    if txt is None or txt == "":
        return None
    return txt.lower()


def _parse_key_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --filter '{raw}'. Expected key=value.")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Invalid --filter '{raw}'. Empty key.")
    if value == "":
        raise ValueError(f"Invalid --filter '{raw}'. Empty value.")
    return key, value


def _sanitize_segment(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"[^a-z0-9._-]+", "-", cleaned)
    cleaned = re.sub(r"-+", "-", cleaned)
    return cleaned.strip("-") or "na"


def _to_tracking_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _ms_to_utc_iso(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=UTC).isoformat()


def _run_to_payload(run: Run) -> dict[str, Any]:
    info = run.info
    data = run.data
    return {
        "run_id": info.run_id,
        "run_name": data.tags.get("mlflow.runName", ""),
        "status": info.status,
        "lifecycle_stage": info.lifecycle_stage,
        "artifact_uri": info.artifact_uri,
        "start_time_ms": info.start_time,
        "end_time_ms": info.end_time,
        "start_time_utc": _ms_to_utc_iso(info.start_time),
        "end_time_utc": _ms_to_utc_iso(info.end_time),
        "metrics": dict(data.metrics),
        "params": dict(data.params),
        "tags": dict(data.tags),
    }


RUN_KEYS_TO_DROP = {
    "run_id",
    "run_name",
    "status",
    "lifecycle_stage",
    "artifact_uri",
    "start_time_ms",
    "end_time_ms",
}

PARAM_KEYS_TO_DROP = {
    "eval_after_train",
    "target_col",
    "max_origins_per_year",
    "dataset_tag",
    "covariate_column_digest",
    "status",
}

TAG_EXACT_KEYS_TO_DROP = {
    "run.kind",
    "run.training_input_mode",
    "run.model.requested",
    "run.model.family",
    "run.dataset.tag",
    "run.fold",
}


def _prune_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k.startswith("all.")}


def _prune_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k not in PARAM_KEYS_TO_DROP}


def _prune_tags(tags: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v
        for k, v in tags.items()
        if not k.startswith("mlflow.")
        and not k.startswith("dataset.")
        and k not in TAG_EXACT_KEYS_TO_DROP
    }


def _prune_run_for_export(run_payload: dict[str, Any]) -> dict[str, Any]:
    pruned = {k: v for k, v in run_payload.items() if k not in RUN_KEYS_TO_DROP}
    pruned["metrics"] = _prune_metrics(dict(run_payload.get("metrics", {})))
    pruned["params"] = _prune_params(dict(run_payload.get("params", {})))
    pruned["tags"] = _prune_tags(dict(run_payload.get("tags", {})))
    return pruned


def _resolve_filter_key(raw_key: str) -> str:
    normalized = raw_key.strip().lower().replace("-", "_")
    for canonical, spec in FILTER_REGISTRY.items():
        if normalized == canonical:
            return canonical
        alias_norm = {alias.lower().replace("-", "_") for alias in spec.aliases}
        if normalized in alias_norm:
            return canonical
    return normalized


def _build_filters(args: argparse.Namespace) -> dict[str, str]:
    filters: dict[str, str] = {}

    fixed_filters = {
        "action": args.action,
        "mode": args.mode,
        "training_input_mode": args.training_input_mode,
        "stem": args.stem,
        "model": args.model,
        "status": args.status,
        "train_epochs": args.epochs,
    }
    for key, value in fixed_filters.items():
        if value is not None:
            filters[key] = value

    for raw in args.filter or []:
        key, value = _parse_key_value(raw)
        canonical = _resolve_filter_key(key)
        filters[canonical] = value
    return filters


def _payload_matches(payload: dict[str, Any], filters: dict[str, str]) -> bool:
    for key, expected in filters.items():
        expected_norm = _normalize_for_compare(expected)
        if expected_norm is None:
            continue

        spec = FILTER_REGISTRY.get(key)
        if spec is None:
            candidate_values = [
                payload.get(key),
                payload.get("params", {}).get(key),
                payload.get("tags", {}).get(key),
            ]
        else:
            candidate_values = [extract(payload) for extract in spec.extractors]

        if not any(
            _normalize_for_compare(v) == expected_norm for v in candidate_values
        ):
            return False
    return True


def _search_all_runs(
    client: MlflowClient, experiment_id: str, view_type: int
) -> list[Run]:
    runs: list[Run] = []
    page_token: str | None = None
    while True:
        page = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=view_type,
            max_results=50000,
            page_token=page_token,
        )
        runs.extend(page)
        page_token = page.token
        if not page_token:
            break
    return runs


def _build_output_filename(experiment_name: str, filters: dict[str, str]) -> str:
    parts = [f"experiment-{_sanitize_segment(experiment_name)}"]
    for key in sorted(filters):
        value = filters[key]
        parts.append(f"{_sanitize_segment(key)}-{_sanitize_segment(value)}")
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    parts.append(timestamp)
    return "__".join(parts) + ".json"


def _resolve_experiments(requested: list[str]) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    for name in requested:
        key = name.strip().lower()
        if key in EXPERIMENT_MAP:
            resolved.append((key, EXPERIMENT_MAP[key]))
            continue
        if key in EXPERIMENT_MAP.values():
            canonical = next(k for k, v in EXPERIMENT_MAP.items() if v == key)
            resolved.append((canonical, key))
            continue
        raise ValueError(
            f"Unknown experiment '{name}'. Use one of {sorted(EXPERIMENT_MAP)} or IDs {sorted(EXPERIMENT_MAP.values())}."
        )
    return resolved


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Export raw MLflow runs with optional parameter/tag filters."
    )

    parser.add_argument(
        "--db-path",
        type=Path,
        default=project_root / "data" / "results" / "mlflow.db",
        help="Path to MLflow sqlite db file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "data" / "data-exports" / "mlflow_exports",
        help="Base output folder. Experiment subfolders are created under this path.",
    )

    parser.add_argument(
        "--experiments",
        "--experiemnts",
        nargs="+",
        default=["foundation", "custom"],
        help="Experiments to export. Accepted values: foundation custom 1 2.",
    )

    parser.add_argument("--action", help="Filter by params.action")
    parser.add_argument("--mode", help="Filter by params.mode")
    parser.add_argument(
        "--training-input-mode",
        dest="training_input_mode",
        help="Filter by training_input_mode (param/tag).",
    )
    parser.add_argument(
        "--stem",
        "--steam",
        dest="stem",
        help="Filter by variant stem (supports both --stem and typo --steam).",
    )
    parser.add_argument("--model", help="Filter by model/tag/run_name prefix")
    parser.add_argument("--status", help="Filter by run status")
    parser.add_argument(
        "--epochs",
        help="Filter by params.train_epochs (example: --epochs 10).",
    )

    parser.add_argument(
        "--filter",
        action="append",
        help="Additional extensible filters as key=value. Can be used multiple times.",
    )

    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Include deleted runs. By default, deleted runs are excluded.",
    )

    return parser.parse_args()


def export_runs(args: argparse.Namespace) -> None:
    db_path = args.db_path
    if not db_path.exists():
        raise FileNotFoundError(f"MLflow db file not found: {db_path}")

    tracking_uri = _to_tracking_uri(db_path)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    experiments = _resolve_experiments(args.experiments)
    filters = _build_filters(args)
    view_type = ViewType.ALL if args.include_deleted else ViewType.ACTIVE_ONLY

    for experiment_name, experiment_id in experiments:
        runs = _search_all_runs(
            client=client, experiment_id=experiment_id, view_type=view_type
        )
        payload_rows = [_run_to_payload(run) for run in runs]
        filtered_rows = [row for row in payload_rows if _payload_matches(row, filters)]
        export_rows = [_prune_run_for_export(row) for row in filtered_rows]

        output_dir = args.output_root / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / _build_output_filename(experiment_name, filters)

        export_payload = {
            "applied_filters": filters,
            "run_count": len(export_rows),
            "runs": export_rows,
        }

        output_file.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")
        print(
            f"[{experiment_name}] exported {len(filtered_rows)} runs -> {output_file}"
        )


def main() -> None:
    args = parse_args()
    export_runs(args)


if __name__ == "__main__":
    main()
