# datakhanon/model/base.py
from __future__ import annotations
import abc
from typing import Any, Dict, Optional
import joblib
import os
import json

class Serializable(abc.ABC):
    @abc.abstractmethod
    def save(self, path: str) -> None:
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str):
        ...

class BaseModel(Serializable, abc.ABC):
    """
    Contrato simples para modelos no datakhanon.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    @abc.abstractmethod
    def fit(self, X, y, **fit_kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def predict_proba(self, X):
        """Opcional — pode ser sobrescrito por wrappers."""
        raise NotImplementedError

    def score(self, X, y, scorer=None):
        if scorer:
            return scorer(self, X, y)
        try:
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, self.predict(X))
        except Exception:
            return None

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        joblib.dump(self, os.path.join(path, "model.joblib"))
        # salvar metadata mínima
        meta = {"class": self.__class__.__name__, "params": self.params}
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: str):
        obj = joblib.load(os.path.join(path, "model.joblib"))
        return obj
