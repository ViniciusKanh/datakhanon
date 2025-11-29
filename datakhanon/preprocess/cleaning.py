# datakhanon/preprocess/cleaning.py
"""
Funções utilitárias para limpeza de DataFrames.
Objetivo: operações comuns de pré-processamento que não dependem de estado (stateless).
"""

from typing import Optional, Dict
import pandas as pd
import numpy as np
import re


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes de colunas: minúsculas, underscores, remove espaços e caracteres não-alfanuméricos.
    Retorna um novo DataFrame (cópia).
    """
    df = df.copy()
    new_cols = []
    for c in df.columns:
        # substitui espaços por underscore, remove caracteres especiais, passa pra minúsculas
        nc = re.sub(r"[^\w\s]", "", c)  # remove caracteres especiais
        nc = re.sub(r"\s+", "_", nc)    # espaços -> underscore
        nc = nc.strip().lower()
        new_cols.append(nc)
    df.columns = new_cols
    return df


def drop_duplicates(df: pd.DataFrame, subset: Optional[list] = None, keep: str = "first") -> pd.DataFrame:
    """
    Remove duplicatas. subset -> colunas para considerar (None = todas).
    keep -> 'first' | 'last' | False
    """
    return df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)


def drop_missing_threshold(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Remove colunas cujo percentual de valores faltantes é >= threshold.
    threshold: proporção entre 0 e 1 (ex: 0.5 remove colunas com >=50% NA).
    Retorna novo DataFrame com colunas filtradas.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold deve estar entre 0 e 1")
    na_frac = df.isna().mean()
    keep_cols = na_frac[na_frac < threshold].index.tolist()
    return df.loc[:, keep_cols].copy()


def convert_dtypes(df: pd.DataFrame, dtype_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Converte tipos explicitamente com base em dtype_map {col: dtype}.
    Atenção: falhas de conversão viram NaN.
    Se dtype_map for None, tenta inferir tipos numéricos automaticamente (secure cast).
    """
    df = df.copy()
    if dtype_map:
        for col, dt in dtype_map.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dt)
                except Exception:
                    # tentativa segura: para numérico, usar to_numeric com coerce
                    if dt in ("int", "float", "numeric"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(str)
    else:
        # infer numeric columns conservatively
        for col in df.columns:
            # se tem maioria numérica (strings que podem ser float), converte
            sample = df[col].dropna().head(200)
            if len(sample) == 0:
                continue
            # checking if can be coerced to numeric
            coerced = pd.to_numeric(sample, errors="coerce")
            if coerced.notna().sum() / len(sample) > 0.8:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
