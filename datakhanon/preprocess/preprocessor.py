# datakhanon/preprocess/preprocessor.py
"""
Preprocessor de alto-nível para orquestrar limpeza -> imputação -> encoding -> feature engineering.
API sklearn-like: fit, transform, fit_transform. Métodos save/load para serializar o objeto.
Comentários e nomes em Português (Brasil).
"""

from typing import Optional, List, Dict, Any
import os
import warnings
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
      - codificação de categóricas (OneHotEncoderWrapper por padrão, mas suporta 'target' e 'ordinal')
      - engenharia/seleção de features numéricas (FeatureEngineer)

    Parâmetros principais:
    - categorical_columns: lista de colunas categóricas (opcional)
    - numeric_columns: lista de colunas numéricas (opcional)
    - drop_missing_threshold_prop: proporção (0..1) para remover colunas com too many missings
    - imputers_kwargs: dict passado para ColumnImputer
    - imputer_config: alias compatível (deprecated) para imputers_kwargs
    - ohe_kwargs: dict com configurações do encoder; aceita chave 'encoder' com valores 'onehot'|'target'|'ordinal'
    - feature_engineer_kwargs: dict passado para FeatureEngineer

    Uso típico:
        pp = Preprocessor(categorical_columns=["purpose","housing"],
                          feature_engineer_kwargs={"select_k":10})
        X = pp.fit_transform(df, y)
        pp.save("artifacts/preprocessor.joblib")
    """

    def __init__(
        self,
        categorical_columns: Optional[List[str]] = None,
        numeric_columns: Optional[List[str]] = None,
        drop_missing_threshold_prop: float = 0.95,
        imputers_kwargs: Optional[Dict[str, Any]] = None,
        imputer_config: Optional[Dict[str, Any]] = None,
        ohe_kwargs: Optional[Dict[str, Any]] = None,
        feature_engineer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Compatibilidade: imputer_config -> imputers_kwargs
        if imputers_kwargs is None and imputer_config is not None:
            warnings.warn(
                "Argumento `imputer_config` está obsoleto. Use `imputers_kwargs` na inicialização do Preprocessor.",
                DeprecationWarning,
            )
            imputers_kwargs = imputer_config

        # configurações do preprocessor
        self.categorical_columns = categorical_columns
        self.numeric_columns = numeric_columns
        try:
            self.drop_missing_threshold_prop = float(drop_missing_threshold_prop)
        except Exception:
            raise ValueError("drop_missing_threshold_prop deve ser numérico (0..1).")

        self.imputers_kwargs = dict(imputers_kwargs or {})
        self.ohe_kwargs = dict(ohe_kwargs or {})
        self.feature_engineer_kwargs = dict(feature_engineer_kwargs or {})

        # componentes que serão inicializados no fit
        self.col_imputer: Optional[ColumnImputer] = None
        self.ohe: Optional[Any] = None  # OneHotEncoderWrapper | TargetEncoderWrapper | OrdinalEncoderWrapper
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
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df deve ser um pandas.DataFrame")

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
            # filtrar apenas as colunas presentes
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

        # 3) Instanciar e treinar encoder (suporta onehot / target / ordinal)
        # O tipo de encoder pode ser definido em ohe_kwargs["encoder"]
        encoder_type = str(self.ohe_kwargs.pop("encoder", "onehot")).lower()
        ohe_cols = [c for c in (self.categorical_columns_ or []) if c in df_imp.columns]

        if encoder_type == "target":
            # TargetEncoderWrapper normalmente exige y
            self.ohe = TargetEncoderWrapper(columns=ohe_cols, **self.ohe_kwargs)
            if y is None:
                raise ValueError("TargetEncoder requer parâmetro y em fit(). Forneça y.")
            self.ohe.fit(df_imp, y)
        elif encoder_type == "ordinal":
            self.ohe = OrdinalEncoderWrapper(columns=ohe_cols, **self.ohe_kwargs)
            # ordinal encoder não necessita y por padrão
            self.ohe.fit(df_imp)
        else:
            # default: one-hot
            self.ohe = OneHotEncoderWrapper(columns=ohe_cols, **self.ohe_kwargs)
            self.ohe.fit(df_imp)

        # 4) Treinar FeatureEngineer
        fe_kwargs = dict(self.feature_engineer_kwargs)
        # proteger select_k para não exceder número de variáveis numéricas
        n_num = len(self.numeric_columns_) if self.numeric_columns_ is not None else 0
        if "select_k" in fe_kwargs and fe_kwargs["select_k"] is not None:
            try:
                select_k = int(fe_kwargs["select_k"])
            except Exception:
                select_k = None
            if select_k is not None:
                fe_kwargs["select_k"] = min(select_k, max(1, n_num))

        self.fe = FeatureEngineer(**fe_kwargs)
        X_num_for_fit = df_imp[self.numeric_columns_] if (self.numeric_columns_ and len(self.numeric_columns_) > 0) else pd.DataFrame(index=df_imp.index)
        # passar y para métodos que necessitam de target (FeatureEngineer pode usar y)
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

        if not isinstance(df, pd.DataFrame):
            raise TypeError("df deve ser um pandas.DataFrame")

        # aplicar as mesmas limpezas iniciais
        df2 = standardize_column_names(df)
        df2 = drop_missing_threshold(df2, threshold=self.drop_missing_threshold_prop)
        df2 = convert_dtypes(df2)

        # reindexar para as colunas retidas no fit (colunas ausentes serão criadas com NaN)
        if self.retained_columns_ is not None:
            # Garantir colunas extras do novo df sejam descartadas; manter exatamente o schema treinado
            df2 = df2.reindex(columns=self.retained_columns_)

        # imputar (col_imputer já treinado)
        df_imp = self.col_imputer.transform(df2)

        # codificar categóricas
        ohe_df = self.ohe.transform(df_imp) if self.ohe is not None else pd.DataFrame(index=df_imp.index)

        # transformar numéricas
        num_cols = self.numeric_columns_ or []
        X_num = df_imp.loc[:, num_cols] if (num_cols and len(num_cols) > 0) else pd.DataFrame(index=df_imp.index)
        if self.fe is not None:
            num_df = self.fe.transform(X_num)
        else:
            num_df = X_num

        # assegurar que num_df e ohe_df tenham índices compatíveis
        num_df = num_df.reset_index(drop=True)
        ohe_df = ohe_df.reset_index(drop=True)

        # concatenar resultado (numéricas primeiro, depois categóricas codificadas)
        result = pd.concat([num_df, ohe_df], axis=1)

        # Se desejar garantir ordem de colunas determinística, você pode armazenar e aplicar um esquema adicional.
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
        if not path:
            raise ValueError("Forneça um caminho válido para salvar o preprocessor.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "Preprocessor":
        """Carrega um Preprocessor serializado via joblib."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return joblib.load(path)

    def __repr__(self) -> str:
        return (
            f"<Preprocessor fitted={self.fitted_} "
            f"num_cols={len(self.numeric_columns_ or [])} "
            f"cat_cols={len(self.categorical_columns_ or [])}>"
        )
