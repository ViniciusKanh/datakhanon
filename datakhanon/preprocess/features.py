# datakhanon/preprocess/features.py
"""
Engenharia e seleção de features:
 - FeatureEngineer: encapsula escalonadores, seleção por variância, SelectKBest e importância de modelos.
 - Fornece utilitários para criar ColumnTransformer simples.
"""

from typing import Optional, List, Tuple, Dict
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Classe para aplicar escalonamento e seleção básica de features.
    Parâmetros principais:
      - scaler: 'standard' | 'minmax' | 'robust' | None
      - variance_threshold: float (remover features com baixa variância)
      - select_k: int | None (seleção univariada)
      - problem_type: 'classification' | 'regression' (necessário para select_kbest e feature_importance)
    """

    def __init__(
        self,
        scaler: Optional[str] = "standard",
        variance_threshold: Optional[float] = None,
        select_k: Optional[int] = None,
        use_model_importance: bool = False,
        model_importance_k: Optional[int] = None,
        problem_type: str = "classification",
        random_state: Optional[int] = 42,
    ):
        self.scaler = scaler
        self.variance_threshold = variance_threshold
        self.select_k = select_k
        self.use_model_importance = use_model_importance
        self.model_importance_k = model_importance_k
        self.problem_type = problem_type
        self.random_state = random_state

        # objetos internos
        self.scaler_ = None
        self.variance_selector_ = None
        self.kbest_ = None
        self.importance_selected_ = None

    def _build_scaler(self):
        if self.scaler is None:
            return None
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "minmax":
            return MinMaxScaler()
        if self.scaler == "robust":
            return RobustScaler()
        raise ValueError("scaler inválido")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_num = X.select_dtypes(include=[np.number])
        # scaler
        self.scaler_ = self._build_scaler()
        if self.scaler_ is not None:
            self.scaler_.fit(X_num)

        # variance threshold
        if self.variance_threshold is not None:
            self.variance_selector_ = VarianceThreshold(self.variance_threshold)
            self.variance_selector_.fit(X_num)

        # select k best (univariado)
        if self.select_k is not None and y is not None:
            score_func = f_classif if self.problem_type == "classification" else f_regression
            self.kbest_ = SelectKBest(score_func=score_func, k=self.select_k)
            self.kbest_.fit(X_num, y)

        # feature importance (embedded)
        if self.use_model_importance and y is not None:
            if self.problem_type == "classification":
                clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            else:
                clf = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            clf.fit(X_num.fillna(0), y)
            importances = pd.Series(clf.feature_importances_, index=X_num.columns)
            k = self.model_importance_k or int(len(importances) * 0.5)
            self.importance_selected_ = importances.sort_values(ascending=False).head(k).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X_num = X.select_dtypes(include=[np.number])
        cols = X_num.columns.tolist()

        # aplicar variance threshold
        if self.variance_selector_ is not None:
            mask = self.variance_selector_.get_support()
            cols = [c for c, m in zip(cols, mask) if m]
            X_num = X_num[cols]

        # aplicar select k best
        if self.kbest_ is not None:
            mask = self.kbest_.get_support()
            cols = [c for c, m in zip(cols, mask) if m]
            X_num = X_num[cols]

        # aplicar importance selection
        if self.importance_selected_ is not None:
            cols = [c for c in cols if c in self.importance_selected_]
            X_num = X_num[cols]

        # aplicar scaler
        if self.scaler_ is not None and not X_num.empty:
            X_num[:] = self.scaler_.transform(X_num)

        # substituir as colunas numéricas no DataFrame original
        non_num = X.select_dtypes(exclude=[np.number])
        result = pd.concat([X_num, non_num], axis=1)[list(X_num.columns) + list(non_num.columns)]
        return result

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def get_selected_numeric_features(self) -> List[str]:
        """
        Retorna lista de features numéricas atualmente selecionadas (após fit).
        """
        if self.importance_selected_ is not None:
            return self.importance_selected_
        if self.kbest_ is not None:
            # quando kbest está presente, retornamos colunas suportadas
            support = self.kbest_.get_support()
            # não temos acesso direto aos nomes aqui sem contexto do X usado no fit;
            # portanto, esse método deverá ser usado após fit e acessando internal state via usuário.
            return []  # placeholder: usuário deve consultar transform result
        return []

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        return joblib.load(path)


def build_simple_column_transformer(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scaler: Optional[str] = "standard",
) -> ColumnTransformer:
    """
    Retorna um ColumnTransformer com pipelines mínimas para numéricos e categóricos.
    Útil para encaixar em pipelines sklearn.
    """
    num_scaler = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        None: "passthrough",
    }.get(scaler, StandardScaler())

    numeric_pipeline = Pipeline([("scaler", num_scaler)]) if num_scaler != "passthrough" else "passthrough"
    categorical_pipeline = Pipeline([("ohe", OneHotEncoderWrapper(columns=categorical_cols))])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
