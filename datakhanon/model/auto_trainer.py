# datakhanon/model/auto_trainer.py
from typing import Optional, Dict, Any, List
import os
import numpy as np
from datakhanon.model.selector import detect_task_type, default_candidates
from datakhanon.model.metrics import auto_metrics
from datakhanon.model.exporter import available_exporters, export
from sklearn.model_selection import cross_val_score, train_test_split
from datakhanon.model.report import generate_report
from datakhanon.model.persistence import save_model
import warnings
warnings.filterwarnings("ignore")

class AutoTrainer:
    """
    AutoTrainer: fluxo all-in-one:
     - detecta tarefa
     - se model='auto' seleciona candidatos e escolhe o melhor por CV (roc/neg_mse/accuracy)
     - treina o estimador final
     - avalia (X_val opcional; se não fornecido, faz train_test_split interno)
     - gera relatório (report.generate_report)
     - exporta (por default joblib; pode incluir onnx)
    Parâmetros principais:
     - model: 'auto' ou nome do candidato (e.g. 'rf')
     - output_dir: diretório base para salvar artefatos
     - export_on_finish: bool (padrao True)
     - export_formats: list[str] ou None -> se None usa available_exporters()
     - cv: int CV folds para seleção
    """
    def __init__(self, model: str = "auto", output_dir: str = "outputs/auto", export_on_finish: bool = True,
                 export_formats: Optional[List[str]] = None, cv: int = 3, random_state: int = 0, max_candidates:int=3):
        self.model_choice = model
        self.output_dir = output_dir
        self.export_on_finish = export_on_finish
        self.cv = cv
        self.random_state = random_state
        self.max_candidates = max_candidates
        if export_formats is None:
            self.export_formats = available_exporters()
        else:
            self.export_formats = export_formats

    def _score_for_task(self, task, estimator, X, y):
        # seleciona a pontuação de CV apropriada (maior é melhor)
        if task in ("binary", "multiclass"):
            scoring = "accuracy"
        else:
            scoring = "neg_mean_squared_error"  # maior (menos negativo) melhor
        try:
            scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=scoring)
            return float(np.mean(scores))
        except Exception:
            return -np.inf

    def fit(self, X, y, X_val=None, y_val=None, tune:bool=False, tune_params:Optional[Dict[str,Any]]=None):
        task = detect_task_type(y)
        # split val if not provided
        if X_val is None or y_val is None:
            X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.2, random_state=self.random_state, stratify=y if task!="regression" else None)
            if X_val is None: X_val = X_hold
            if y_val is None: y_val = y_hold
            X, y = X_train, y_train

        chosen = None
        chosen_name = None

        if self.model_choice == "auto":
            candidates = default_candidates(task, max_candidates=self.max_candidates, random_state=self.random_state)
            best_score = -np.inf
            for name, est in candidates:
                sc = self._score_for_task(task, est, X, y)
                if sc > best_score:
                    best_score = sc
                    chosen = est
                    chosen_name = name
        else:
            # procurar entre defaults; se não encontrado assume que model_choice é um sklearn estimator
            candidates = default_candidates(task, max_candidates=10, random_state=self.random_state)
            mapping = {n: e for n,e in candidates}
            if self.model_choice in mapping:
                chosen_name = self.model_choice
                chosen = mapping[self.model_choice]
            else:
                # se for um estimator direto
                if hasattr(self.model_choice, "fit"):
                    chosen = self.model_choice
                    chosen_name = getattr(chosen, "__class__", type(chosen)).__name__
                else:
                    raise ValueError("Modelo desconhecido e não é estimator sklearn.")

        # Fit final no conjunto inteiro de X (treino)
        chosen.fit(X, y)

        # Avaliação
        y_pred = chosen.predict(X_val)
        y_score = None
        try:
            if hasattr(chosen, "predict_proba"):
                prob = chosen.predict_proba(X_val)
                y_score = np.asarray(prob)
        except Exception:
            y_score = None

        metrics = auto_metrics(task, y_val, y_pred, y_score)

        # gerar report (usa generate_report que salva JSON + MD + imagens)
        os.makedirs(self.output_dir, exist_ok=True)
        report_paths = generate_report(chosen, X, y, X_val, y_val, self.output_dir, model_name=chosen_name or "model")

        # salvar modelo via joblib + meta (utiliza persistence.save_model)
        # cria uma spec mínima
        try:
            from datakhanon.model.model_spec import ModelSpec
            spec = ModelSpec.make(name=chosen_name or "model", framework="sklearn", metrics_summary=metrics)
        except Exception:
            spec = None
        model_path = os.path.join(self.output_dir, f"{chosen_name or 'model'}")
        os.makedirs(model_path, exist_ok=True)
        save_model(chosen, model_path, spec)

        exports = {}
        if self.export_on_finish:
            # export para todos os formatos configurados
            try:
                X_sample = X_val if X_val is not None else X[:10]
                exports = export(chosen, model_path, base_name=f"{chosen_name or 'model'}", formats=self.export_formats, X_sample=X_sample)
            except Exception as e:
                exports = {"error": str(e)}

        return {
            "model_name": chosen_name,
            "model": chosen,
            "metrics": metrics,
            "report": report_paths,
            "model_saved_at": model_path,
            "exports": exports
        }

    def predict(self, X):
        if not hasattr(self, "model") or self.model is None:
            raise ValueError("Nenhum modelo treinado. Chame fit() primeiro.")
        return self.model.predict(X)
