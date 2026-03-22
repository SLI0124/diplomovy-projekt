from __future__ import annotations

from dataclasses import dataclass

import torch
from adapters.base import BaseFoundationModelAdapter, ForecastResult, ModelContext
from adapters.custom import Model1Adapter
from adapters.foundation import (
    Chronos2Adapter,
    Moirai1BaseAdapter,
    TimesFM25Adapter,
)


@dataclass(frozen=True)
class AdapterSpec:
    adapter_cls: type[BaseFoundationModelAdapter]
    family: str


MODEL_REGISTRY: dict[str, AdapterSpec] = {
    "chronos2": AdapterSpec(adapter_cls=Chronos2Adapter, family="foundation"),
    "moirai1_base": AdapterSpec(adapter_cls=Moirai1BaseAdapter, family="foundation"),
    "timesfm25": AdapterSpec(adapter_cls=TimesFM25Adapter, family="foundation"),
    "model_1": AdapterSpec(adapter_cls=Model1Adapter, family="custom"),
}


def supported_model_ids() -> tuple[str, ...]:
    return tuple(MODEL_REGISTRY.keys())


def resolve_model_family(model_name: str) -> str:
    key = model_name.lower()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(supported_model_ids())
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")
    return spec.family


def build_model_adapter(
    model_name: str, model_ctx: ModelContext, device: torch.device
) -> BaseFoundationModelAdapter:
    key = model_name.lower()
    spec = MODEL_REGISTRY.get(key)
    if spec is None:
        supported = ", ".join(supported_model_ids())
        raise ValueError(f"Unsupported model '{model_name}'. Supported: {supported}")

    adapter = spec.adapter_cls(model_ctx=model_ctx, device=device)
    return adapter


__all__ = [
    "AdapterSpec",
    "BaseFoundationModelAdapter",
    "ForecastResult",
    "ModelContext",
    "MODEL_REGISTRY",
    "build_model_adapter",
    "resolve_model_family",
    "supported_model_ids",
    "Chronos2Adapter",
    "Moirai1BaseAdapter",
    "Model1Adapter",
    "TimesFM25Adapter",
]
