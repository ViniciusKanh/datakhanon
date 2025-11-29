# datakhanon/preprocess/preprocessor.py
"""
Preprocessor de alto-nível para orquestrar limpeza -> imputação -> encoding -> feature engineering.
API sklearn-like: fit, transform, fit_transform. Métodos save/load para serializar o objeto.
Comentários e nomes em Português (Brasil).
"""

from typing import Optional, List, Dict, Any
import os
import joblib
import pandas as pd
import numpy as np

from .cleaning import standardize_column_names, drop_missing_threshold, convert_dtypes
from .imputers import ColumnImputer
from .encoders import OneHotEncoderWrapper, TargetEncoderWrapper, OrdinalEncoderWrapper
from .features import FeatureEngineer
from .reporter import DataHealthReport  # análise / profile do dataset


def _is_identifier(col: str) -> bool:
    """Heurística simples para detectar colunas identificadoras."""
    lower = col.lower()
    return lower == "id" or lower.endswith("_id")


class Preprocessor:
    """
    Classe orquestradora que encapsula as etapas de pré-processamento:
      - limpeza leve (normalização de nomes, remoção por missingness, conversão de tipos)
      - imputação (ColumnImputer)
      - codificação de categóricas (OneHotEncoderWrapper por padrão)
      - engenharia/seleção de features numéricas (FeatureEngineer)

    Uso típico:
        pp = Preprocessor(categorical_columns=["purpose","housing"], feature_engineer_kwargs={"select_k":10})
        X = pp.fit_transform(df, y)
        pp.save("artifacts/preprocessor.joblib")
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        drop_missing_threshold_prop: float = 0.95,
        imputers_kwargs: Optional[Dict[str, Any]] = None,
        ohe_kwargs: Optional[Dict[str, Any]] = None,
        feature_engineer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # configurações do preprocessor
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        self.drop_missing_threshold_prop = float(drop_missing_threshold_prop)
        self.imputers_kwargs = imputers_kwargs or {}
        self.ohe_kwargs = ohe_kwargs or {}
        self.feature_engineer_kwargs = feature_engineer_kwargs or {}

        # componentes que serão inicializados no fit
        self.col_imputer: Optional[ColumnImputer] = None
        self.ohe: Optional[OneHotEncoderWrapper] = None
        self.fe: Optional[FeatureEngineer] = None
        self.fitted_ = False

        # metadados do fit (usados para garantir consistência entre fit/transform)
        self.retained_columns_: Optional[List[str]] = None
        self.numeric_columns_: Optional[List[str]] = None
        self.categorical_columns_: Optional[List[str]] = None

    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Ajusta todos os componentes do preprocessor com base no DataFrame df.
        y: target opcional (necessário para certos encoders / seleção de features).
        Retorna self.
        """
        # 1) Limpeza leve
        df2 = standardize_column_names(df)
        df2 = drop_missing_threshold(df2, threshold=self.drop_missing_threshold_prop)
        df2 = convert_dtypes(df2)

        # registrar colunas retidas (ordem determinística)
        self.retained_columns_ = list(df2.columns)

        # detectar/filtrar colunas categóricas e numéricas com base no df pós-limpeza
        if self.categorical_columns is None:
            self.categorical_columns_ = df2.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        else:
            self.categorical_columns_ = [c for c in self.categorical_columns if c in df2.columns]

        if self.numeric_columns is None:
            self.numeric_columns_ = df2.select_dtypes(include=[np.number]).columns.tolist()
        else:
            self.numeric_columns_ = [c for c in self.numeric_columns if c in df2.columns]

        # remover possíveis identificadores das colunas numéricas
        self.numeric_columns_ = [c for c in self.numeric_columns_ if not _is_identifier(c)]

        # 2) Treinar imputador
        self.col_imputer = ColumnImputer(**self.imputers_kwargs)
        self.col_imputer.fit(df2)
        df_imp = self.col_imputer.transform(df2)

        # 3) Treinar encoder (OneHotEncoderWrapper por padrão)
        ohe_cols = [c for c in (self.categorical_columns_ or []) if c in df_imp.columns]
        self.ohe = OneHotEncoderWrapper(columns=ohe_cols, **self.ohe_kwargs)

        # Se o encoder for do tipo TargetEncoder (não é o padrão), necessita de y
        if isinstance(self.ohe, TargetEncoderWrapper):
            if y is None:
                raise ValueError("TargetEncoder requer y no fit(). Forneça o parâmetro y.")
            self.ohe.fit(df_imp, y)
        else:
            self.ohe.fit(df_imp)

        # 4) Treinar FeatureEngineer
        fe_kwargs = dict(self.feature_engineer_kwargs)
        n_num = len(self.numeric_columns_) if self.numeric_columns_ is not None else 0
        if "select_k" in fe_kwargs and fe_kwargs["select_k"] is not None:
            fe_kwargs["select_k"] = min(fe_kwargs["select_k"], max(1, n_num))

        self.fe = FeatureEngineer(**fe_kwargs)
        X_num_for_fit = df_imp[self.numeric_columns_] if self.numeric_columns_ else pd.DataFrame(index=df_imp.index)
        # passar y para métodos que necessitam de target
        self.fe.fit(X_num_for_fit, y)

        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica as transformações no DataFrame df com base no estado treinado (fit).
        Retorna DataFrame pronto para modelagem (numéricas transformadas + colunas codificadas).
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor não ajustado. Chame fit() primeiro.")

        # aplicar as mesmas limpezas iniciais
        df2 = standardize_column_names(df)
        df2 = drop_missing_threshold(df2, threshold=self.drop_missing_threshold_prop)
        df2 = convert_dtypes(df2)

        # reindexar para as colunas retidas no fit (colunas ausentes serão preenchidas com NaN)
        if self.retained_columns_ is not None:
            df2 = df2.reindex(columns=self.retained_columns_)

        # imputar (col_imputer já treinado)
        df_imp = self.col_imputer.transform(df2)

        # codificar categóricas
        ohe_df = self.ohe.transform(df_imp) if self.ohe is not None else pd.DataFrame(index=df_imp.index)

        # transformar numéricas
        num_cols = self.numeric_columns_ or []
        X_num = df_imp.loc[:, num_cols] if num_cols else pd.DataFrame(index=df_imp.index)
        if self.fe is not None:
            num_df = self.fe.transform(X_num)
        else:
            num_df = X_num

        # concatenar resultado (numéricas primeiro, depois categóricas codificadas)
        result = pd.concat([num_df.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
        return result

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Convenience method que executa fit e em seguida transform."""
        return self.fit(df, y).transform(df)

    def analyze(self, df: pd.DataFrame, target: Optional[pd.Series] = None, output_dir: str = "artifacts/report"):
        """
        Gera relatório de "Data Health" do dataset.
        Retorna um dicionário com o resumo (summary).
        Gera arquivos em output_dir: data_health_report.json, data_health_report.html e imagens de apoio.
        """
        reporter = DataHealthReport(df, target)
        summary = reporter.run(output_dir=output_dir, html=True, json_out=True)
        return summary

    def save(self, path: str):
        """Serializa o preprocessor completo (joblib)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Carrega um Preprocessor serializado via joblib."""
        return joblib.load(path)
