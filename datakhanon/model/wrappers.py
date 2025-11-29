# datakhanon/model/wrappers.py
from typing import Any, Dict, Optional
from datakhanon.model.base import BaseModel
from datakhanon.model.registry import register_model
import numpy as np

@register_model("sklearn_estimator")
class SKLearnWrapper(BaseModel):
    """
    Wrapper leve para estimadores scikit-learn.
    Aceita params: {"estimator": <sklearn estimator>, "fit_params": {...}}
    Implementa get_params/set_params para compatibilidade com sklearn.clone.
    """
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        est = None
        if params:
            est = params.get("estimator")
        self.estimator = est
        self.fit_params = params.get("fit_params", {}) if params else {}

    def fit(self, X, y, **fit_kwargs):
        if self.estimator is None:
            raise ValueError("Nenhum estimator fornecido.")
        self.estimator.fit(X, y, **{**self.fit_params, **fit_kwargs})
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(X)
        preds = self.predict(X)
        return np.asarray(preds)

    def get_params(self, deep: bool = True):
        params = {"estimator": self.estimator, "fit_params": self.fit_params}
        if deep and self.estimator is not None and hasattr(self.estimator, "get_params"):
            for k, v in self.estimator.get_params(deep=True).items():
                params[f"estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params.pop("estimator")
        if "fit_params" in params:
            self.fit_params = params.pop("fit_params")
        est_updates = {}
        for k in list(params.keys()):
            if k.startswith("estimator__"):
                est_key = k.split("estimator__", 1)[1]
                est_updates[est_key] = params.pop(k)
        if est_updates and self.estimator and hasattr(self.estimator, "set_params"):
            self.estimator.set_params(**est_updates)
        return self
