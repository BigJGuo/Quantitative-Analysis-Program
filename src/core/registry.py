"""Global model registry.

Importing a model module that uses `@register_model` is enough to make it
discoverable; the orchestration layer never needs to know individual model
class names. Tests can use `clear_registry()` to start from a clean slate.
"""

from __future__ import annotations

from typing import TypeVar

from src.core.base_model import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}

ModelT = TypeVar("ModelT", bound=type[BaseModel])


def register_model(cls: ModelT) -> ModelT:
    """Class decorator. Adds `cls` to the global registry keyed by `cls.name`."""
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError("@register_model can only decorate BaseModel subclasses")
    name = cls.name
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(
            f"Model name {name!r} already registered to {existing.__qualname__}"
        )
    _REGISTRY[name] = cls
    return cls


def get_model(name: str) -> type[BaseModel]:
    if name not in _REGISTRY:
        raise KeyError(
            f"Model {name!r} not registered. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_REGISTRY)


def models_by_layer(layer: int) -> list[type[BaseModel]]:
    return [cls for cls in _REGISTRY.values() if cls.layer == layer]


def clear_registry() -> None:
    """Wipe the registry. Intended for test setup/teardown."""
    _REGISTRY.clear()
