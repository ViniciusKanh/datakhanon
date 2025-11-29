# datakhanon/preprocess/encoders.py
"""
Wrappers para codificação de variáveis categóricas.
Oferece:
 - OneHotEncoderWrapper: usa sklearn OneHotEncoder (handle_unknown='ignore').
 - OrdinalEncoderWrapper: usa sklearn OrdinalEncoder com categorias aprendidas.
 - TargetEncoderWrapper: implementação simples de encoding por média target (sem dependência extra).
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import inspect
import warnings


class OneHotEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, drop: Optional[str] = None, sparse: bool = False):
        """
        columns: lista de colunas categóricas. Se None, detecta automaticamente.
        drop: None | 'first' para evitar multicolinearidade
        sparse: se True retorna matriz esparsa; se False retorna array denso.
        Observação: o nome do argumento do sklearn mudou entre versões (sparse -> sparse_output).
        Esta implementação detecta automaticamente o argumento correto.
        """
        self.columns = columns
        self.drop = drop
        self.sparse = bool(sparse)
        self.encoder_ = None
        self.feature_names_out_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.columns_ = cols

        # Preparar argumentos compatíveis com a versão do sklearn instalada
        encoder_kwargs: Dict[str, Any] = {"handle_unknown": "ignore", "drop": self.drop}
        try:
            sig = inspect.signature(OneHotEncoder.__init__)
            params = sig.parameters
            if "sparse" in params:
                # versões antigas usam 'sparse'
                encoder_kwargs["sparse"] = self.sparse
            elif "sparse_output" in params:
                # versões novas usam 'sparse_output'
                encoder_kwargs["sparse_output"] = self.sparse
            else:
                # fallback: não passar nenhum desses e aceitar default
                warnings.warn(
                    "OneHotEncoder: parâmetro 'sparse'/'sparse_output' não encontrado na assinatura. Usando comportamento padrão."
                )
        except Exception:
            # se inspect falhar, tentar usar sparse_output (mais provável em versões recentes)
            encoder_kwargs["sparse_output"] = self.sparse

        # Criar o encoder de forma resiliente
        try:
            self.encoder_ = OneHotEncoder(**encoder_kwargs)
        except TypeError:
            # fallback: tentar sem o argumento de sparseness
            encoder_kwargs.pop("sparse", None)
            encoder_kwargs.pop("sparse_output", None)
            self.encoder_ = OneHotEncoder(**encoder_kwargs)

        # Ajustar encoder
        self.encoder_.fit(X[cols].astype(str))

        # formar nomes das features (compatível com diferentes sklearn)
        try:
            self.feature_names_out_ = self.encoder_.get_feature_names_out(self.columns_).tolist()
        except Exception:
            # fallback simples caso get_feature_names_out não esteja disponível
            categories = getattr(self.encoder_, "categories_", [])
            names = []
            for col, cats in zip(self.columns_, categories):
                for cat in cats:
                    names.append(f"{col}__{cat}")
            self.feature_names_out_ = names

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None:
            raise RuntimeError("Encoder não ajustado. Chame fit() antes.")
        arr = self.encoder_.transform(X[self.columns_].astype(str))

        # Converter para array denso caso seja matriz esparsa (mais robusto que testar self.sparse)
        if hasattr(arr, "toarray"):
            arr = arr.toarray()

        # Garantir que feature_names_out_ existe e tem o tamanho correto
        if self.feature_names_out_ is None:
            # tentar obter dinamicamente
            try:
                self.feature_names_out_ = self.encoder_.get_feature_names_out(self.columns_).tolist()
            except Exception:
                # construir nomes simples
                cats = getattr(self.encoder_, "categories_", [])
                names = []
                for col, cvals in zip(self.columns_, cats):
                    for v in cvals:
                        names.append(f"{col}__{v}")
                self.feature_names_out_ = names

        return pd.DataFrame(arr, columns=self.feature_names_out_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "OneHotEncoderWrapper":
        return joblib.load(path)


class OrdinalEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Optional[List[str]] = None, unknown_value: int = -1):
        self.columns = columns
        self.unknown_value = unknown_value
        self.encoder_ = None

    def fit(self, X: pd.DataFrame, y=None):
        cols = self.columns or X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.columns_ = cols
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=self.unknown_value)
        self.encoder_.fit(X[cols].astype(str))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.encoder_ is None:
            raise RuntimeError("Encoder não ajustado. Chame fit() antes.")
        arr = self.encoder_.transform(X[self.columns_].astype(str))
        return pd.DataFrame(arr, columns=self.columns_, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "OrdinalEncoderWrapper":
        return joblib.load(path)


class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    """
    Implementação simples de target encoding (mean encoding).
    - Requer y no fit.
    - Substitui cada categoria pela média do target (por coluna).
    - Para unseen categories usa média global do target.
    """

    def __init__(self, columns: Optional[List[str]] = None, smoothing: float = 1.0):
        self.columns = columns
        self.smoothing = float(smoothing)
        self.maps_: Dict[str, Dict[Any, float]] = {}
        self.global_mean_: float = 0.0

    def _smooth_mean(self, cat_counts, cat_mean, global_mean):
        # smoothing simples: weight = n/(n + k), k = smoothing
        k = self.smoothing
        return (cat_counts * cat_mean + k * global_mean) / (cat_counts + k)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if y is None:
            raise ValueError("TargetEncoderWrapper requer y no fit().")
        cols = self.columns or X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        self.columns_ = cols
        self.global_mean_ = float(y.mean())
        for c in cols:
            df = pd.DataFrame({c: X[c].astype(str), "target": y})
            stats = df.groupby(c)["target"].agg(["count", "mean"])
            mapping = {}
            for idx, row in stats.iterrows():
                mapping[idx] = float(self._smooth_mean(row["count"], row["mean"], self.global_mean_))
            self.maps_[c] = mapping
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not hasattr(self, "columns_"):
            raise RuntimeError("TargetEncoderWrapper não ajustado. Chame fit(X, y).")
        for c in self.columns_:
            X[c] = X[c].astype(str).map(self.maps_.get(c, {})).fillna(self.global_mean_)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "TargetEncoderWrapper":
        return joblib.load(path)
