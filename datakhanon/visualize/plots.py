# datakhanon/visualize/plots.py
"""
Funções utilitárias de plotagem para EDA.
Cada função retorna um matplotlib.figure.Figure e tem argumento opcional save_path para persistir em disco.
Comentários e nomes em Português-BR.
"""

from typing import Optional, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# backend para ambientes headless
plt.switch_backend("Agg")

# tentar importar seaborn (melhora visual)
try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False


def _ensure_dir(path: str):
    if path:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def plot_missing_heatmap(df: pd.DataFrame, save_path: Optional[str] = None, top_n: int = 50) -> plt.Figure:
    """
    Plota barra/heatmap de missingness por coluna.
    - top_n: quantas colunas mais missing exibir (ordem decrescente)
    - retorna matplotlib.figure.Figure
    """
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    fig, ax = plt.subplots(figsize=(min(14, 0.18 * max(10, len(miss))), 3))
    if miss.empty:
        ax.text(0.5, 0.5, "Sem missingness relevante", ha="center", va="center")
        ax.axis("off")
    else:
        ax.bar(miss.index, miss.values)
        ax.set_ylabel("% missing")
        ax.set_xticklabels(miss.index, rotation=45, ha="right", fontsize=8)
        ax.set_title("Missing (%) por coluna — top {}".format(len(miss)))
        fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_corr_heatmap(df: pd.DataFrame, save_path: Optional[str] = None, max_vars: int = 25) -> plt.Figure:
    """
    Plota mapa de correlação (pearson) para colunas numéricas.
    - max_vars: limita a quantidade de variáveis (seleciona por variância)
    """
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] <= 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "Insuficientes colunas numéricas para correlação", ha="center", va="center")
        ax.axis("off")
        return fig

    # limitar por variância
    if nums.shape[1] > max_vars:
        cols = nums.var().sort_values(ascending=False).head(max_vars).index.tolist()
        nums = nums[cols]

    corr = nums.corr()
    fig, ax = plt.subplots(figsize=(min(12, 0.35 * corr.shape[0] + 3), min(10, 0.35 * corr.shape[0] + 3)))
    cax = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Mapa de correlação (Pearson)")
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_histograms(df: pd.DataFrame, cols: Optional[List[str]] = None, save_dir: Optional[str] = None, max_per_row: int = 3) -> List[plt.Figure]:
    """
    Gera histogramas para colunas numéricas.
    - cols: lista de colunas; se None, usa as primeiras 12 colunas numéricas
    - save_dir: se fornecido, salva cada figura como hist_<col>.png
    - retorna lista de Figures
    """
    nums = df.select_dtypes(include=[np.number])
    if cols is None:
        cols = nums.columns.tolist()[:12]
    figs = []
    for c in cols:
        fig, ax = plt.subplots(figsize=(5, 3))
        data = df[c].dropna()
        if data.empty:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            ax.axis("off")
        else:
            if _HAS_SEABORN:
                sns.histplot(data, kde=True, ax=ax)
            else:
                ax.hist(data, bins=30)
            ax.set_title(c)
        fig.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"hist_{c}.png")
            _ensure_dir(path)
            fig.savefig(path, dpi=120)
        figs.append(fig)
    return figs


def plot_boxplot(df: pd.DataFrame, column: str, by: Optional[str] = None, save_path: Optional[str] = None) -> plt.Figure:
    """
    Boxplot para uma coluna numérica; opcionalmente segmentado por 'by' (categórica).
    - retorna Figure
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    data = df[[column]].copy()
    if by is None:
        if data[column].dropna().empty:
            ax.text(0.5, 0.5, "Sem dados", ha="center", va="center")
            ax.axis("off")
        else:
            ax.boxplot(data[column].dropna())
            ax.set_title(column)
    else:
        if _HAS_SEABORN:
            sns.boxplot(x=by, y=column, data=df, ax=ax)
        else:
            # desenhar vários boxplots por grupo simples
            groups = df.groupby(by)[column].apply(lambda s: s.dropna().values)
            ax.boxplot(list(groups), labels=list(groups.index))
            ax.set_xticklabels(list(groups.index), rotation=45, ha="right", fontsize=8)
            ax.set_title(f"{column} por {by}")
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_categorical_counts(df: pd.DataFrame, column: str, save_path: Optional[str] = None, top_k: int = 20) -> plt.Figure:
    """
    Plota barras de contagem para coluna categórica (exibe top_k categorias).
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    vc = df[column].astype(str).value_counts(dropna=False).head(top_k)
    vc.plot(kind="bar", ax=ax)
    ax.set_title(f"Contagem — {column}")
    ax.set_ylabel("count")
    ax.set_xticklabels(vc.index, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=150)
    return fig


def plot_pairplot_reduced(df: pd.DataFrame, cols: Optional[List[str]] = None, save_path: Optional[str] = None, max_vars: int = 6) -> plt.Figure:
    """
    Pairplot reduzido (mantém até max_vars). Usa seaborn.pairplot se disponível; caso contrário, cria grid simples.
    Retorna Figure (se seaborn, a figura do pairplot).
    """
    nums = df.select_dtypes(include=[np.number])
    if cols is None:
        cols = nums.columns.tolist()[:max_vars]
    else:
        cols = [c for c in cols if c in df.columns][:max_vars]
    if len(cols) < 2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "Poucas variáveis para pairplot", ha="center", va="center")
        ax.axis("off")
        return fig

    if _HAS_SEABORN:
        g = sns.pairplot(df[cols].dropna(), diag_kind="kde", plot_kws={"s": 10})
        # seaborn retorna PairGrid; converter para fig via plt.gcf()
        fig = plt.gcf()
    else:
        # grid simples de scatterplots
        n = len(cols)
        fig, axes = plt.subplots(n, n, figsize=(2 * n, 2 * n))
        for i, xi in enumerate(cols):
            for j, yj in enumerate(cols):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df[xi].dropna(), bins=20)
                else:
                    ax.scatter(df[yj], df[xi], s=6)
                if i == n - 1:
                    ax.set_xlabel(yj, fontsize=8, rotation=45)
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(xi, fontsize=8)
                else:
                    ax.set_yticklabels([])
        fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=140)
    return fig


def plot_feature_importance(importances: List[float], feature_names: List[str], save_path: Optional[str] = None, top_k: int = 20) -> plt.Figure:
    """
    Plota barras de importância de features (recebe listas do modelo).
    - importances: valores numéricos
    - feature_names: nomes correspondentes
    """
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * len(df_imp))))
    ax.barh(df_imp["feature"][::-1], df_imp["importance"][::-1])
    ax.set_title("Feature Importances — top {}".format(len(df_imp)))
    fig.tight_layout()
    if save_path:
        _ensure_dir(save_path)
        fig.savefig(save_path, dpi=140)
    return fig
