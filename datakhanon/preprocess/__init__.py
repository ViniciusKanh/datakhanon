# datakhanon/preprocess/__init__.py
"""
Interface pública do pacote datakhanon.preprocess.
Exporta utilitários principais para limpeza, imputação, codificação e engenharia de features.
"""
from .cleaning import (
    standardize_column_names,
    drop_duplicates,
    drop_missing_threshold,
    convert_dtypes,
)
from .imputers import ColumnImputer, SimpleColumnImputer
from .encoders import OneHotEncoderWrapper, OrdinalEncoderWrapper, TargetEncoderWrapper
from .features import FeatureEngineer

__all__ = [
    "standardize_column_names",
    "drop_duplicates",
    "drop_missing_threshold",
    "convert_dtypes",
    "ColumnImputer",
    "SimpleColumnImputer",
    "OneHotEncoderWrapper",
    "OrdinalEncoderWrapper",
    "TargetEncoderWrapper",
    "FeatureEngineer",
]
