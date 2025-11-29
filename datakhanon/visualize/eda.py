# datakhanon/visualize/eda.py
"""
Funções de EDA de alto nível que orquestram os plots e geram um resumo JSON.
- quick_eda: função única que gera as visualizações principais e resume em JSON.
- eda_summary: retorna dicionário com estatísticas e caminhos das imagens geradas.
"""

from typing import Optional, Dict, Any
import os
import json
import pandas as pd
from .plots import (
    plot_missing_heatmap,
    plot_corr_heatmap,
    plot_histograms,
    plot_pairplot_reduced,
    plot_categorical_counts,
)
# opção de usar reporter se preferir relatório HTML completo
try:
    from datakhanon.preprocess.reporter import generate_report  # type: ignore
    _HAS_REPORTER = True
except Exception:
    _HAS_REPORTER = False


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def eda_summary(df: pd.DataFrame, output_dir: str = "artifacts/eda", target: Optional[pd.Series] = None) -> Dict[str, Any]:
    """
    Gera um conjunto de plots e um resumo JSON.
    - output_dir: diretório onde imagens e resumo JSON serão salvos.
    - target: série opcional para produzir distribuição do target (não usada por default aqui).
    Retorna dicionário com chaves: images (dict), numeric_summary (DataFrame -> dict), categorical_summary (dict).
    """
    _ensure_dir(output_dir)
    summary: Dict[str, Any] = {}
    # estatísticas numéricas e categóricas
    numeric = df.select_dtypes(include=["number"]).describe().round(4).to_dict()
    categorical = {}
    for c in df.select_dtypes(include=["object", "category", "bool"]).columns:
        vc = df[c].astype(str).value_counts(dropna=False).head(10).to_dict()
        categorical[c] = {"top": vc, "n_unique": int(df[c].nunique(dropna=True)), "n_missing": int(df[c].isna().sum())}

    summary["numeric_summary"] = numeric
    summary["categorical_summary"] = categorical

    # plots
    images = {}
    images["missing"] = None
    images["corr"] = None
    images["histograms"] = []
    images["pairplot"] = None

    # missing
    missing_path = os.path.join(output_dir, "missing.png")
    fig = plot_missing_heatmap(df, save_path=missing_path)
    images["missing"] = missing_path

    # correlation
    corr_path = os.path.join(output_dir, "corr.png")
    fig = plot_corr_heatmap(df, save_path=corr_path)
    images["corr"] = corr_path

    # histograms
    hist_dir = os.path.join(output_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)
    hist_figs = plot_histograms(df, save_dir=hist_dir)
    images["histograms"] = [os.path.join(hist_dir, f"hist_{i}.png") for i in range(len(hist_figs))]

    # pairplot reduzido
    pair_path = os.path.join(output_dir, "pairplot.png")
    fig = plot_pairplot_reduced(df, save_path=pair_path)
    images["pairplot"] = pair_path

    # categorical counts (gera até 4 primeiras categóricas)
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()[:4]
    for c in cat_cols:
        p = os.path.join(output_dir, f"count_{c}.png")
        fig = plot_categorical_counts(df, column=c, save_path=p)
        images.setdefault("categorical_counts", []).append(p)

    summary["images"] = images

    # salvar JSON resumo
    json_path = os.path.join(output_dir, "eda_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    summary["json_path"] = json_path
    return summary


def quick_eda(df: pd.DataFrame, output_dir: str = "artifacts/eda", target: Optional[pd.Series] = None, use_reporter: bool = False) -> Dict[str, Any]:
    """
    Função de alto-nível que executa EDA:
    - Se use_reporter=True e o reporter estiver disponível, delega para generate_report (HTML completo).
    - Caso contrário, usa eda_summary para salvar plots e resumo JSON.
    Retorna dicionário com paths e resumo.
    """
    _ensure_dir(output_dir)
    if use_reporter and _HAS_REPORTER:
        # reporter gera HTML auto-contido e json
        out = generate_report(df, target=target, output_dir=output_dir)
        # normalize keys para compatibilidade
        return {
            "kind": "reporter_html",
            "html": out.get("html_report"),
            "json": out.get("json_summary"),
            "alerts": out.get("alerts"),
        }
    else:
        return eda_summary(df, output_dir=output_dir, target=target)
