# datakhanon/model/registry.py
from typing import Dict, Type
from datakhanon.model.base import BaseModel

_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str):
    def decorator(cls: Type[BaseModel]):
        key = name.lower()
        if key in _MODEL_REGISTRY:
            raise KeyError(f"Modelo '{name}' já registrado.")
        _MODEL_REGISTRY[key] = cls
        return cls
    return decorator

def get_model(name: str, **kwargs) -> BaseModel:
    key = name.lower()
    cls = _MODEL_REGISTRY.get(key)
    if cls is None:
        raise KeyError(f"Modelo '{name}' não encontrado. Disponíveis: {list(_MODEL_REGISTRY.keys())}")
    return cls(kwargs)

def available_models():
    return list(_MODEL_REGISTRY.keys())
