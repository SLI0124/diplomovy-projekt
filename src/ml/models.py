from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

if TYPE_CHECKING:
    from config import RuntimeConfig


ModelBuilder = Callable[["RuntimeConfig"], object]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: ModelBuilder
    needs_scaling: bool = False


def _build_decision_tree(config: RuntimeConfig) -> object:
    return DecisionTreeRegressor(
        max_depth=config.dt_max_depth,
        min_samples_split=config.dt_min_samples_split,
        random_state=config.seed,
    )


def _build_random_forest(config: RuntimeConfig) -> object:
    return RandomForestRegressor(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        random_state=config.seed,
        n_jobs=-1,
    )


def _build_gradient_boosting(config: RuntimeConfig) -> object:
    return GradientBoostingRegressor(
        n_estimators=config.gb_n_estimators,
        learning_rate=config.gb_learning_rate,
        max_depth=config.gb_max_depth,
        random_state=config.seed,
    )


def _build_linear_regression(config: RuntimeConfig) -> object:
    _ = config
    return LinearRegression()


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "decision-tree": ModelSpec(name="decision-tree", builder=_build_decision_tree),
    "random-forest": ModelSpec(name="random-forest", builder=_build_random_forest),
    "gradient-boosting": ModelSpec(
        name="gradient-boosting", builder=_build_gradient_boosting
    ),
    "linear-regression": ModelSpec(
        name="linear-regression", builder=_build_linear_regression, needs_scaling=True
    ),
}


def supported_model_ids() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def build_pipeline(model_name: str, config: RuntimeConfig) -> Pipeline:
    key = model_name.strip().lower()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(supported_model_ids())
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if spec.needs_scaling:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", spec.builder(config)))
    return Pipeline(steps)
