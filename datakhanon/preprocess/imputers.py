# datakhanon/preprocess/imputers.py
"""
Imputers com API sklearn-like: fit / transform / fit_transform.
Fornece: SimpleColumnImputer (uso de SimpleImputer por colunas) e ColumnImputer
(gerencia colunas numéricas e categóricas separadamente).

Esta implementação tenta ativar o IterativeImputer (módulo experimental) quando
necessário e faz fallback para SimpleImputer com aviso quando o IterativeImputer
não estiver disponível na instalação do scikit-learn.
"""

from typing import List, Optional, Dict, Any
import warnings
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin, BaseEstimator
import joblib

# Tentar ativar IterativeImputer (em versões onde é experimental)
try:
    # ativa o experimental (em versões mais antigas do sklearn)
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer  # type: ignore
    ITERATIVE_AVAILABLE = True
except Exception:
    # tentar importar diretamente (em versões onde já não é experimental)
    try:
        from sklearn.impute import IterativeImputer  # type: ignore
        ITERATIVE_AVAILABLE = True
    except Exception:
        IterativeImputer = None  # type: ignore
        ITERATIVE_AVAILABLE = False

class SimpleColumnImputer(BaseEstimator, TransformerMixin):
    """
    Imputa colunas com estratégia simples usando sklearn.impute.SimpleImputer.
    Exemplo de uso:
        imp = SimpleColumnImputer(strategy="median", fill_value=None, columns=["a","b"])
        imp.fit(df)
        df2 = imp.transform(df)
    """

    def __init__(self, strategy: str = "mean", fill_value: Optional[Any] = None, columns: Optional[List[str]] = None):
        self.strategy = strategy
        self.fill_value = fill_value
        self.columns = columns
        self._imputer = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns if self.columns is not None else X.columns.tolist()
        self.columns_ = cols
        self._imputer = SimpleImputer(strategy=self.strategy, fill_value=self.fill_value)
        self._imputer.fit(X[cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if self._imputer is None:
            raise RuntimeError("Imputer não ajustado. Chame fit() antes.")
        X[self.columns_] = self._imputer.transform(X[self.columns_])
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class ColumnImputer(BaseEstimator, TransformerMixin):
    """
    Imputador que aplica estratégias separadas para numéricos e categóricos.
    - num_strategy: 'mean' | 'median' | 'constant' | 'iterative'
    - cat_strategy: 'most_frequent' | 'constant'
    """

    def __init__(
        self,
        num_strategy: str = "median",
        cat_strategy: str = "most_frequent",
        num_fill_value: Optional[float] = None,
        cat_fill_value: Optional[Any] = None,
        iterative_kwargs: Optional[Dict] = None,
    ):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy
        self.num_fill_value = num_fill_value
        self.cat_fill_value = cat_fill_value
        self.iterative_kwargs = iterative_kwargs or {}
        self.num_imputer_ = None
        self.cat_imputer_ = None

    def fit(self, X: pd.DataFrame, y=None):
        # detect columns
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # num imputer
        if self.num_strategy == "iterative":
            if not ITERATIVE_AVAILABLE:
                warnings.warn(
                    "IterativeImputer não disponível na sua instalação do scikit-learn. "
                    "Será utilizado SimpleImputer como fallback. Para usar IterativeImputer, "
                    "atualize o scikit-learn ou instale uma versão que suporte o experimental.",
                    UserWarning,
                )
                self.num_imputer_ = SimpleImputer(strategy="median", fill_value=self.num_fill_value)
            else:
                # type: ignore[attr-defined]
                self.num_imputer_ = IterativeImputer(**self.iterative_kwargs)  # type: ignore
        else:
            self.num_imputer_ = SimpleImputer(strategy=self.num_strategy, fill_value=self.num_fill_value)

        if num_cols:
            self.num_imputer_.fit(X[num_cols])
        self.num_columns_ = num_cols

        # cat imputer
        self.cat_imputer_ = SimpleImputer(strategy=self.cat_strategy, fill_value=self.cat_fill_value)
        if cat_cols:
            self.cat_imputer_.fit(X[cat_cols])
        self.cat_columns_ = cat_cols

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if getattr(self, "num_columns_", None):
            X[self.num_columns_] = self.num_imputer_.transform(X[self.num_columns_])
        if getattr(self, "cat_columns_", None):
            X[self.cat_columns_] = self.cat_imputer_.transform(X[self.cat_columns_])
            # preservar tipo category quando aplicável
            for c in self.cat_columns_:
                if pd.api.types.is_categorical_dtype(X[c]) is False:
                    X[c] = X[c].astype(object)
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        """Serializa o imputador para path via joblib."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "ColumnImputer":
        return joblib.load(path)
