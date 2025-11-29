# datakhanon/preprocess/utils.py
"""
Utilitários de verificação e limpeza rápida para usar com DataKhanon.
Fornece:
 - data_quality_check(df, target=None) -> dict
 - treat_data(df, ..., save_path=None) -> (cleaned_df, log)
Comentários e nomes em Português.
"""

from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
import os
from .imputers import ColumnImputer

def data_quality_check(df: pd.DataFrame, target: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Roda verificações rápidas: nulos, duplicatas, outliers (IQR), cardinalidade categórica.
    Retorna dicionário com resumo e máscaras básicas.
    """
    summary: Dict[str, Any] = {}
    n_rows, n_cols = df.shape
    summary["shape"] = {"n_rows": int(n_rows), "n_cols": int(n_cols)}

    # nulos
    na_counts = df.isna().sum()
    na_pct = (na_counts / max(1, n_rows) * 100).round(4)
    summary["missing_count"] = na_counts.to_dict()
    summary["missing_percent"] = na_pct.to_dict()

    # duplicatas
    n_dup = int(df.duplicated().sum())
    summary["n_duplicates"] = n_dup
    if n_dup > 0:
        summary["duplicate_example_index"] = df[df.duplicated(keep=False)].index.tolist()[:10]

    # categorical cardinality
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    cat_card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
    summary["categorical_cardinality"] = cat_card

    # outliers (IQR) -> proporção por coluna numérica
    num = df.select_dtypes(include=[np.number])
    outlier_info = {}
    outlier_mask = pd.DataFrame(index=df.index)
    for col in num.columns:
        s = num[col].dropna()
        if s.empty:
            outlier_info[col] = {"n_outliers": 0, "pct_outliers": 0.0}
            outlier_mask[col + "_outlier"] = False
            continue
        q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
        if iqr == 0:
            mask = pd.Series(False, index=df.index)
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (df[col] < lower) | (df[col] > upper)
        n_out = int(mask.sum())
        pct = float(round(n_out / max(1, n_rows) * 100, 4))
        outlier_info[col] = {"n_outliers": n_out, "pct_outliers": pct}
        outlier_mask[col + "_outlier"] = mask.fillna(False)
    summary["outliers_iqr"] = outlier_info
    # incluir amostra de linhas que possuem ao menos 1 outlier
    if not outlier_mask.empty:
        rows_with_outlier = outlier_mask.any(axis=1)
        summary["rows_with_outlier_count"] = int(rows_with_outlier.sum())
        summary["rows_with_outlier_index_example"] = df[rows_with_outlier].index.tolist()[:10]
    else:
        summary["rows_with_outlier_count"] = 0
        summary["rows_with_outlier_index_example"] = []

    # target balance (se fornecido)
    if target is not None and len(target) == n_rows:
        try:
            t = pd.Series(target).dropna()
            vc = t.value_counts(normalize=True).round(4).to_dict()
            summary["target_balance"] = vc
        except Exception:
            summary["target_balance"] = None

    # return masks for programmatic use if needed
    summary["example_masks"] = {
        "outlier_mask": outlier_mask,   # DataFrame booleano (col_outlier columns)
        "duplicated_mask": df.duplicated(keep=False)
    }
    return summary


def treat_data(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    drop_rows_missing_fraction_gt: Optional[float] = 0.5,
    drop_cols_missing_fraction_gt: Optional[float] = None,
    impute_num_strategy: str = "median",
    impute_cat_strategy: str = "most_frequent",
    outlier_method: str = "iqr",         # only 'iqr' currently supported
    outlier_action: str = "cap",         # 'cap'|'remove'|'mark'
    outlier_cap_quantile: float = 0.01,  # when capping, trim tails by quantiles as safety
    save_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aplica tratamento no DataFrame e retorna (cleaned_df, log).
    - drop_duplicates: remove linhas duplicadas (mantém primeira).
    - drop_rows_missing_fraction_gt: remove linhas com fracao de missing > threshold.
    - drop_cols_missing_fraction_gt: remove colunas com fracao missing > threshold.
    - imputação: usa ColumnImputer do datakhanon (fit/transform).
    - outlier: por IQR; ação pode ser 'cap' (winsorize aos limites IQR), 'remove' (apagar linhas que possuem outliers),
      ou 'mark' (adicionar colunas booleanas indicando outlier).
    - save_path: se fornecido, salva CSV do dataframe tratado.
    """
    log: Dict[str, Any] = {}
    df_work = df.copy()
    n0 = len(df_work)

    # 1) duplicatas
    if drop_duplicates:
        before = len(df_work)
        df_work = df_work.drop_duplicates(keep="first")
        log["duplicates_removed"] = before - len(df_work)

    # 2) drop colunas por missing %
    n_rows = len(df_work)
    if drop_cols_missing_fraction_gt is not None:
        miss_frac_cols = df_work.isna().mean()
        cols_to_drop = miss_frac_cols[miss_frac_cols > drop_cols_missing_fraction_gt].index.tolist()
        df_work = df_work.drop(columns=cols_to_drop)
        log["cols_dropped_for_missing"] = cols_to_drop

    # 3) drop linhas por missing % (se solicitado)
    if drop_rows_missing_fraction_gt is not None:
        row_frac = df_work.isna().sum(axis=1) / max(1, df_work.shape[1])
        idx_drop = row_frac[row_frac > drop_rows_missing_fraction_gt].index
        log["rows_removed_for_missing_count"] = int(len(idx_drop))
        df_work = df_work.drop(index=idx_drop)

    # 4) outliers (apenas colunas numéricas)
    nums = df_work.select_dtypes(include=[np.number]).columns.tolist()
    outlier_rows_index = set()
    outlier_marks = {}
    if outlier_method == "iqr" and nums:
        for col in nums:
            s = df_work[col].dropna()
            if s.empty:
                continue
            q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr; upper = q3 + 1.5 * iqr
            mask = (df_work[col] < lower) | (df_work[col] > upper)
            if mask.any():
                outlier_marks[col + "_outlier"] = mask.fillna(False)
                outlier_idx = set(df_work[mask].index.tolist())
                outlier_rows_index.update(outlier_idx)
                if outlier_action == "cap":
                    # winsorize: cap to quantile boundaries to avoid extreme clamping
                    low_cap = df_work[col].quantile(outlier_cap_quantile)
                    high_cap = df_work[col].quantile(1.0 - outlier_cap_quantile)
                    df_work.loc[df_work[col] < lower, col] = max(lower, low_cap)
                    df_work.loc[df_work[col] > upper, col] = min(upper, high_cap)
                # remove will be applied after loop
    if outlier_action == "remove" and outlier_rows_index:
        before = len(df_work)
        df_work = df_work.drop(index=list(outlier_rows_index))
        log["rows_removed_for_outliers_count"] = before - len(df_work)
    else:
        log["outlier_marks_cols"] = list(outlier_marks.keys())

    # 5) imputação (num + cat) usando ColumnImputer do datakhanon
    imp = ColumnImputer(num_strategy=impute_num_strategy, cat_strategy=impute_cat_strategy)
    df_work = imp.fit_transform(df_work)
    log["imputer_used"] = {"num_strategy": impute_num_strategy, "cat_strategy": impute_cat_strategy}

    # 6) se outlier_action == 'mark', adicionar colunas booleanas
    if outlier_action == "mark" and outlier_marks:
        for cname, mask in outlier_marks.items():
            # reindex mask para o df_work final (mask foi construído antes de possíveis drops)
            mask_full = pd.Series(False, index=df.index)
            mask_full.loc[mask.index] = mask
            # manter alinhamento com df_work; reindex e preencher False para linhas removidas
            mask_aligned = mask_full.reindex(df_work.index).fillna(False).astype(bool)
            df_work[cname] = mask_aligned.values
        log["outlier_mark_columns_added"] = list(outlier_marks.keys())

    # 7) salvar (opcional)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df_work.to_csv(save_path, index=False)
        log["saved_to"] = save_path

    log["initial_rows"] = n0
    log["final_rows"] = len(df_work)
    log["final_cols"] = df_work.shape[1]
    return df_work, log
