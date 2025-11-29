# datakhanon/model/trainer.py
from typing import Optional, Callable, Dict, Any
import time
import numpy as np
from datakhanon.model.base import BaseModel
from sklearn.model_selection import cross_validate
from datakhanon.model import report as report_module

class Trainer:
    def __init__(self, model: BaseModel, scorer: Optional[Callable] = None):
        self.model = model
        self.scorer = scorer

    def fit(self, X_train, y_train, **fit_kwargs) -> Dict[str, Any]:
        t0 = time.time()
        self.model.fit(X_train, y_train, **fit_kwargs)
        elapsed = time.time() - t0
        result = {"train_time": elapsed}
        if self.scorer:
            try:
                result["train_score"] = self.scorer(self.model, X_train, y_train)
            except Exception:
                result["train_score"] = None
        return result

class TrainerExtended(Trainer):
    def fit(self, X_train, y_train, X_val=None, y_val=None, output_dir: Optional[str] = None,
            model_name: str = "model", cv: Optional[int] = None, **fit_kwargs) -> Dict[str, Any]:
        logs = super().fit(X_train, y_train, **fit_kwargs)
        if X_val is not None and y_val is not None:
            try:
                preds = self.model.predict(X_val)
                from sklearn.metrics import accuracy_score
                logs["val_accuracy"] = float(accuracy_score(y_val, preds))
            except Exception:
                logs["val_accuracy"] = None

        if cv:
            try:
                cv_res = cross_validate(self.model, X_train, y_train, cv=cv, scoring=self.scorer, return_train_score=False)
            except Exception:
                base = getattr(self.model, "estimator", None)
                if base is not None:
                    cv_res = cross_validate(base, X_train, y_train, cv=cv, scoring=self.scorer, return_train_score=False)
                else:
                    cv_res = None
            if cv_res is not None:
                for k, v in cv_res.items():
                    logs[f"cv_{k}"] = np.mean(v).item() if hasattr(v, "mean") else v

        if output_dir and X_val is not None and y_val is not None:
            rep = report_module.generate_report(self.model, X_train, y_train, X_val, y_val, output_dir, model_name)
            logs["report"] = rep

        return logs
