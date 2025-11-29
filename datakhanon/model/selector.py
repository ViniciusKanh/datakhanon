# datakhanon/model/selector.py
"""
Auto Model Selector:
- detecta tarefa (classificação x regressão)
- retorna uma lista ordenada de candidatos (estimadores sklearn)
- permite custom registry override
"""
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def detect_task_type(y) -> str:
    """
    Detecta tarefa a partir de y:
     - 'binary' (2 classes)
     - 'multiclass' (>2 categorias inteiras/strings)
     - 'regression' (float contínuo)
    Heurística:
     - se dtype numérico inteiro e poucas categorias -> classificação
     - se dtype float -> regressão
    """
    y_arr = np.asarray(y)
    if y_arr.dtype.kind in ("f",):  # float -> regressão
        return "regression"
    unique = np.unique(y_arr)
    if unique.size <= 2:
        return "binary"
    # se inteiro com <= 20 categorias -> multiclass
    if y_arr.dtype.kind in ("i", "u") and unique.size <= 50:
        return "multiclass"
    # fallback: se valores repetidos e não muitos valores, classificacao multiclass
    if unique.size < max(50, 0.1 * y_arr.size):
        return "multiclass"
    # última heurística -> regression
    return "regression"

def default_candidates(task: str, max_candidates:int=3, random_state:Optional[int]=0):
    """
    Retorna uma lista de (name, estimator) candidatos por tarefa.
    Mantemos apenas estimadores do sklearn para dependências mínimas.
    """
    if task in ("binary", "multiclass"):
        pool = [
            ("logistic", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=random_state)),
            ("gb", GradientBoostingClassifier(n_estimators=100, random_state=random_state)),
            ("dt", DecisionTreeClassifier(random_state=random_state))
        ]
    else:  # regression
        pool = [
            ("linear", LinearRegression()),
            ("rf_reg", RandomForestRegressor(n_estimators=100, random_state=random_state)),
            ("gb_reg", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
            ("dt_reg", DecisionTreeRegressor(random_state=random_state)),
            ("svr", SVR())
        ]
    return pool[:max_candidates]
