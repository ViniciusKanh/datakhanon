# datakhanon/preprocess/reporter.py
"""
DataKhanon — Interface de Perfilamento (Relatório HTML autônomo)
Gera um HTML auto-contido com visão de "data health" para qualquer DataFrame pandas.
- Depende apenas de: pandas, numpy, matplotlib.
- Imagens são embutidas em base64 no HTML (relatório portátil).
- Gera JSON resumido e o HTML. Fornece seção de alertas com recomendações.
- Comentários e strings em Português (Brasil).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import base64
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuração mínima do matplotlib para ambientes sem display
plt.switch_backend("Agg")


# ============================================================
# Funções auxiliares
# ============================================================

def _png_base64_from_fig(fig) -> str:
    """Converte figura matplotlib para string base64 (data:image/png;base64,...)"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_head_html(df: pd.DataFrame, n: int = 7) -> str:
    """Gera HTML de uma amostra reduzida (head) com classes para estilo."""
    if df.empty:
        return "<div class='empty'>DataFrame vazio</div>"
    return df.head(n).to_html(
        classes="table table-sample",
        border=0,
        index=False,
        justify="left",
        escape=False,
    )


def _summary_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo estatístico para colunas numéricas + missing + outliers (IQR)."""
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()

    desc = num.describe().T
    desc["median"] = num.median()
    desc["n_missing"] = num.isna().sum()

    def _n_outliers(s: pd.Series) -> int:
        s = s.dropna()
        if s.empty:
            return 0
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            return 0
        mask = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))
        return int(mask.sum())

    desc["n_outliers_iqr"] = num.apply(_n_outliers)
    return desc.round(4)


def _summary_categorical(df: pd.DataFrame, topk: int = 7) -> Dict[str, Any]:
    """Resumo para colunas categóricas: n_unique, top valores, missing."""
    cats = df.select_dtypes(include=["object", "category", "bool"])
    out: Dict[str, Any] = {}
    for col in cats.columns:
        vc = cats[col].astype(str).value_counts(dropna=False)
        top = vc.head(topk).to_dict()
        out[col] = {
            "n_unique": int(cats[col].nunique(dropna=True)),
            "top": top,
            "n_missing": int(cats[col].isna().sum()),
        }
    return out


def _get_subset_data_base64(df: pd.DataFrame) -> Dict[str, str]:
    """
    Extrai subsets de dados (duplicatas, linhas com alto missing, amostra)
    e retorna em base64 para download em CSV.
    """
    subsets: Dict[str, str] = {}

    # Duplicatas
    duplicates = df[df.duplicated(keep=False)]
    if not duplicates.empty:
        subsets["duplicates"] = base64.b64encode(
            duplicates.to_csv(index=False).encode("utf-8")
        ).decode("utf-8")

    # Linhas com >= 50% de missing
    n_cols = df.shape[1]
    if n_cols > 0:
        missing_rows = df[df.isna().sum(axis=1) / n_cols >= 0.5]
        if not missing_rows.empty:
            subsets["high_missing_rows"] = base64.b64encode(
                missing_rows.to_csv(index=False).encode("utf-8")
            ).decode("utf-8")

    # Amostra de até 100 linhas
    if len(df) > 0:
        sample = df.sample(min(100, len(df)), random_state=42)
        subsets["sample"] = base64.b64encode(
            sample.to_csv(index=False).encode("utf-8")
        ).decode("utf-8")

    return subsets


def _compute_alerts(
    df: pd.DataFrame, target: Optional[pd.Series] = None
) -> List[Dict[str, Any]]:
    """
    Detecta problemas/alertas simples:
    - colunas com >30% missing
    - média de missing por linha alta
    - percent duplicates >1%
    - categorias com alta cardinalidade (>50% distintos)
    - target imbalance (min class < 5%)
    - muitas outliers em variáveis numéricas (proporção >5%)
    """
    alerts: List[Dict[str, Any]] = []
    n_rows, n_cols = df.shape

    # Missing por coluna
    miss_pct = (df.isna().mean() * 100).round(3)
    high_miss = miss_pct[miss_pct > 30.0]
    for col, pct in high_miss.items():
        alerts.append(
            {
                "type": "high_missing_column",
                "column": col,
                "missing_percent": float(pct),
                "message": (
                    f"Coluna '{col}' possui {pct:.2f}% de valores ausentes "
                    "(sugestão: considerar imputação ou remoção)."
                ),
            }
        )

    # Média de missing por linha
    if n_cols > 0:
        row_missing_frac = (df.isna().sum(axis=1) / n_cols).mean() * 100
        if row_missing_frac > 50.0:
            alerts.append(
                {
                    "type": "high_missing_rows",
                    "percent_rows_avg_missing": float(row_missing_frac),
                    "message": (
                        "Média de missing por linha é alta "
                        f"({row_missing_frac:.1f}%). Verificar qualidade dos registros."
                    ),
                }
            )

    # Duplicatas
    dup_frac = df.duplicated().mean() * 100
    if dup_frac > 1.0:
        alerts.append(
            {
                "type": "duplicates",
                "percent_duplicates": float(round(dup_frac, 3)),
                "message": (
                    f"Dataset tem {dup_frac:.3f}% de linhas duplicadas. "
                    "Considere deduplicar."
                ),
            }
        )

    # Alta cardinalidade em categóricas
    cats = df.select_dtypes(include=["object", "category", "bool"])
    for col in cats.columns:
        n_unique = cats[col].nunique(dropna=True)
        ratio = n_unique / max(1, n_rows)
        if ratio > 0.5:
            alerts.append(
                {
                    "type": "high_cardinality",
                    "column": col,
                    "n_unique": int(n_unique),
                    "ratio": float(round(ratio, 3)),
                    "message": (
                        f"Coluna '{col}' tem alta cardinalidade "
                        f"({n_unique} valores únicos, {ratio:.2f} da base). "
                        "Pode precisar de hashing ou target encoding."
                    ),
                }
            )

    # Desbalanceamento do target
    if target is not None and len(target) == n_rows:
        try:
            ser = pd.Series(target).dropna()
            if not ser.empty:
                vc = ser.value_counts(normalize=True)
                min_frac = float(vc.min())
                if min_frac < 0.05:
                    alerts.append(
                        {
                            "type": "target_imbalanced",
                            "min_class_fraction": min_frac,
                            "message": (
                                "Target desbalanceado: menor classe tem "
                                f"{min_frac*100:.2f}% dos exemplos."
                            ),
                        }
                    )
        except Exception:
            # Em caso de erro, apenas não adiciona alerta
            pass

    # Proporção de outliers numéricos
    nums = df.select_dtypes(include=[np.number])
    for col in nums.columns:
        s = nums[col].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (s < (q1 - 1.5 * iqr)) | (s > (q3 + 1.5 * iqr))
        prop = mask.mean()
        if prop > 0.05:
            alerts.append(
                {
                    "type": "many_outliers",
                    "column": col,
                    "outlier_fraction": float(round(prop, 4)),
                    "message": (
                        f"Coluna '{col}' tem {prop*100:.2f}% de outliers (IQR). "
                        "Verificar tratamento (winsorização, transformação, etc.)."
                    ),
                }
            )

    return alerts


def _plot_missing_bar(df: pd.DataFrame, top_n: int = 30):
    """Barplot de % missing por coluna."""
    fig, ax = plt.subplots(figsize=(8, 3))
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0].head(top_n)
    if miss.empty:
        ax.text(0.5, 0.5, "Sem missingness relevante", ha="center", va="center")
    else:
        ax.bar(miss.index, miss.values, color="#4c51bf")
        ax.set_ylabel("% missing")
        ax.set_xticklabels(miss.index, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    return fig


def _plot_corr_heatmap(df: pd.DataFrame, max_vars: int = 20):
    """Mapa de calor da correlação numérica (limitando nº de colunas)."""
    nums = df.select_dtypes(include=[np.number])
    if nums.shape[1] <= 1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5,
            0.5,
            "Insuficientes colunas numéricas para correlação",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        return fig

    # limitar número de variáveis para evitar imagens enormes
    if nums.shape[1] > max_vars:
        variances = nums.var().sort_values(ascending=False).head(max_vars).index.tolist()
        nums = nums[variances]

    corr = nums.corr()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def _plot_histograms(df: pd.DataFrame, max_plots: int = 8):
    """Histogramas para algumas colunas numéricas."""
    nums = df.select_dtypes(include=[np.number])
    figs: List[Any] = []
    cols = nums.columns.tolist()[:max_plots]
    for c in cols:
        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.hist(nums[c].dropna(), bins=30, color="#4c51bf", alpha=0.7)
        ax.set_title(c, fontsize=10)
        ax.tick_params(axis="both", which="major", labelsize=8)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ============================================================
# Geração de relatório
# ============================================================

def generate_report(
    df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    output_dir: str = "artifacts/report",
    filename: str = "data_health_report.html",
    save_json: bool = True,
) -> Dict[str, Any]:
    """
    Gera relatório HTML auto-contido e resumo JSON.

    Retorna dicionário `summary_out` com metadados de alto nível:
    - html_report: caminho para o HTML
    - json_summary: caminho para o JSON (se save_json=True)
    - alerts: lista de alertas
    - n_rows, n_cols
    """
    _ensure_dir(output_dir)

    n_rows, n_cols = df.shape

    # ---------- resumo (para JSON) ----------
    summary: Dict[str, Any] = {}
    summary["shape"] = {"n_rows": int(n_rows), "n_cols": int(n_cols)}
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    num_summary = _summary_numeric(df)
    summary["numeric_summary"] = num_summary.fillna("").to_dict()

    cat_summary = _summary_categorical(df)
    summary["categorical_summary"] = cat_summary

    miss_pct = (df.isna().mean() * 100).round(4).to_dict()
    summary["missing_percent_by_column"] = miss_pct
    summary["n_duplicates"] = int(df.duplicated().sum())

    nums = df.select_dtypes(include=[np.number])
    if not nums.empty:
        corr = nums.corr().abs().unstack().sort_values(ascending=False)
        corr = corr[corr < 1.0].drop_duplicates().head(30)
        top_pairs = [{"pair": list(idx), "value": float(v)} for idx, v in corr.items()]
    else:
        top_pairs = []
    summary["top_correlation_pairs"] = top_pairs

    alerts = _compute_alerts(df, target)
    summary["alerts"] = alerts

    subsets_b64 = _get_subset_data_base64(df)
    summary["subsets_b64"] = subsets_b64

    # ---------- imagens ----------
    images: Dict[str, Any] = {}
    try:
        fig_miss = _plot_missing_bar(df)
        images["missing_bar"] = _png_base64_from_fig(fig_miss)
    except Exception:
        images["missing_bar"] = None

    try:
        fig_corr = _plot_corr_heatmap(df)
        images["corr_heatmap"] = _png_base64_from_fig(fig_corr)
    except Exception:
        images["corr_heatmap"] = None

    try:
        hist_figs = _plot_histograms(df, max_plots=8)
        images["histograms"] = [_png_base64_from_fig(f) for f in hist_figs]
    except Exception:
        images["histograms"] = []

    target_img = None
    if target is not None and len(target) == n_rows:
        try:
            ser = pd.Series(target).dropna()
            if not ser.empty:
                fig, ax = plt.subplots(figsize=(4, 2.5))
                ser.value_counts().plot(kind="bar", ax=ax, color="#4c51bf")
                ax.set_title("Target distribution", fontsize=10)
                ax.tick_params(axis="both", which="major", labelsize=8)
                fig.tight_layout()
                target_img = _png_base64_from_fig(fig)
        except Exception:
            target_img = None
    images["target_distribution"] = target_img

    # ---------- JSON ----------
    if save_json:
        try:
            with open(
                os.path.join(output_dir, "data_health_report.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
        except Exception:
            # em caso de falha no JSON, segue com o HTML
            pass

    # ---------- HTML ----------
    css = """
    <style>
    :root {
        --color-primary: #4c51bf;
        --color-primary-dark: #3730a3;
        --color-primary-light: #e0e7ff;
        --color-text: #1f2937;
        --color-text-muted: #6b7280;
        --color-bg: #f4f7f9;
        --color-card-bg: #ffffff;
        --color-border: #e5e7eb;
        --color-alert-bg: #fffbeb;
        --color-alert-border: #fcd34d;
        --color-success-bg: #ecfdf5;
        --color-success-border: #34d399;
    }
    body {
        font-family: 'Inter', 'Segoe UI', 'Helvetica', 'Arial', sans-serif;
        color: var(--color-text);
        background: var(--color-bg);
        margin: 0;
        padding: 0;
        line-height: 1.6;
    }
    .container {
        max-width: 1000px;
        margin: 32px auto;
        padding: 24px;
        background: var(--color-card-bg);
        border-radius: 16px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1),
                    0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    h1 {
        margin: 0 0 8px;
        font-size: 32px;
        font-weight: 800;
        color: var(--color-primary-dark);
        letter-spacing: -0.5px;
    }
    h2 {
        font-size: 20px;
        font-weight: 600;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--color-border);
    }
    .meta {
        color: var(--color-text-muted);
        font-size: 14px;
        margin-bottom: 24px;
        border-bottom: 1px solid var(--color-border);
        padding-bottom: 12px;
    }
    .alerts {
        background: var(--color-alert-bg);
        border: 1px solid var(--color-alert-border);
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 24px;
    }
    .alert-item {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 8px;
        background: #fffdfa;
        border: 1px solid #fde68a;
        font-size: 14px;
    }
    .muted {
        color: var(--color-text-muted);
        font-size: 12px;
    }
    .badge {
        display: inline-block;
        background: var(--color-primary);
        color: white;
        padding: 6px 12px;
        border-radius: 999px;
        font-size: 13px;
        margin-right: 10px;
        cursor: pointer;
        border: none;
    }
    .badge:hover {
        background: var(--color-primary-dark);
    }
    .accordion {
        border: 1px solid var(--color-border);
        border-radius: 12px;
        margin-bottom: 16px;
        overflow: hidden;
    }
    .accordion-item {
        border-bottom: 1px solid var(--color-border);
    }
    .accordion-item:last-child {
        border-bottom: none;
    }
    .accordion-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        cursor: pointer;
        background: var(--color-card-bg);
        font-weight: 600;
        font-size: 16px;
    }
    .accordion-header:hover {
        background: var(--color-primary-light);
    }
    .accordion-content {
        padding: 0 20px;
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.4s ease, padding 0.4s ease;
        background: #fcfcfc;
    }
    .accordion-item.active .accordion-content {
        max-height: 5000px;
        padding: 16px 20px 20px 20px;
    }
    .accordion-icon {
        font-size: 18px;
        color: var(--color-primary);
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 16px;
        font-size: 13px;
        border-radius: 8px;
        overflow: hidden;
    }
    table th, table td {
        padding: 8px 10px;
        text-align: left;
        border-bottom: 1px solid var(--color-border);
    }
    table th {
        background-color: var(--color-primary-light);
        color: var(--color-primary-dark);
        font-weight: 700;
        text-transform: uppercase;
        font-size: 11px;
    }
    table tr:nth-child(even) {
        background-color: #f9fafb;
    }
    table tr:hover {
        background-color: #f3f4f6;
    }
    .plot-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-top: 10px;
        justify-content: center;
    }
    .plot {
        max-width: 100%;
        height: auto;
        border-radius: 10px;
        border: 1px solid var(--color-border);
        padding: 8px;
        background: var(--color-card-bg);
    }
    .plot-small {
        max-width: calc(33.333% - 10px);
        flex-grow: 1;
    }
    .plot-full {
        width: 100%;
    }
    input[type="text"] {
        padding: 10px;
        border: 1px solid var(--color-border);
        border-radius: 6px;
        width: 100%;
        box-sizing: border-box;
        margin-bottom: 10px;
    }
    .dark-mode {
        --color-text: #e5e7eb;
        --color-text-muted: #9ca3af;
        --color-bg: #111827;
        --color-card-bg: #1f2937;
        --color-border: #374151;
        --color-primary-light: #374151;
        --color-alert-bg: #451a03;
        --color-alert-border: #b45309;
        --color-success-bg: #064e3b;
        --color-success-border: #10b981;
    }
    .dark-mode body {
        background: var(--color-bg);
        color: var(--color-text);
    }
    .dark-mode table tr:nth-child(even) {
        background-color: #1f2937;
    }
    .dark-mode table tr:nth-child(odd) {
        background-color: var(--color-card-bg);
    }
    pre {
        white-space: pre-wrap;
        background: #f4f7f9;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid var(--color-border);
        font-size: 12px;
    }
    .dark-mode pre {
        background: #1f2937;
        border-color: #374151;
    }
    @media (max-width: 768px) {
        .container {
            margin: 16px;
            padding: 16px;
        }
        .plot-small {
            max-width: calc(50% - 8px);
        }
    }
    @media (max-width: 480px) {
        .plot-small {
            max-width: 100%;
        }
    }
    </style>
    """

    js = """
    <script>
    function toggleAccordion(header) {
        var item = header.parentNode;
        item.classList.toggle('active');
    }
    function toggle(id) {
        var e = document.getElementById(id);
        if (!e) return;
        e.style.display = (e.style.display === 'none') ? 'block' : 'none';
    }
    function downloadCSV(dataBase64, filename) {
        const link = document.createElement('a');
        link.href = 'data:text/csv;charset=utf-8;base64,' + dataBase64;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
    function toggleDarkMode() {
        const isDark = document.body.classList.toggle('dark-mode');
        localStorage.setItem('dk_dark_mode', isDark ? 'on' : 'off');
        const icon = document.getElementById('dark-mode-icon');
        if (icon) {
            icon.innerHTML = isDark ? '\\u263E' : '\\u263C';
        }
    }
    function initDarkMode() {
        const pref = localStorage.getItem('dk_dark_mode');
        const prefersDark = window.matchMedia &&
            window.matchMedia('(prefers-color-scheme: dark)').matches;
        if (pref === 'on' || (pref === null && prefersDark)) {
            document.body.classList.add('dark-mode');
            const icon = document.getElementById('dark-mode-icon');
            if (icon) icon.innerHTML = '\\u263E';
        }
    }
    function filterTable(inputId, tableId) {
        const input = document.getElementById(inputId);
        const filter = input.value.toUpperCase();
        const table = document.getElementById(tableId);
        if (!table) return;
        const tr = table.getElementsByTagName('tr');
        for (let i = 1; i < tr.length; i++) {
            let td = tr[i].getElementsByTagName('td')[0];
            if (td) {
                let txtValue = td.textContent || td.innerText;
                tr[i].style.display = txtValue.toUpperCase().indexOf(filter) > -1 ? '' : 'none';
            }
        }
    }
    function copyJson() {
        try {
            const t = document.getElementById('json_block').innerText;
            navigator.clipboard.writeText(t);
            alert('JSON copiado para a área de transferência!');
        } catch (e) {
            alert('Cópia falhou. Tente selecionar e copiar manualmente.');
        }
    }
    document.addEventListener('DOMContentLoaded', function () {
        const firstAccordion = document.querySelector('.accordion-item');
        if (firstAccordion) firstAccordion.classList.add('active');
        initDarkMode();
    });
    </script>
    """

    html_parts: List[str] = []
    html_parts.append(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>DataKhanon — Data Health Report</title>"
    )
    html_parts.append(css)
    html_parts.append("</head><body>")
    html_parts.append("<div class='container'>")

    # Cabeçalho
    html_parts.append(
        "<div style='display:flex; justify-content:space-between; align-items:center;'>"
        "<h1>DataKhanon — Data Health Report</h1>"
        "<button class='badge' onclick='toggleDarkMode()' "
        "title='Alternar Modo Escuro/Claro' "
        "style='font-size: 18px; padding: 6px 10px;'>"
        "<span id='dark-mode-icon'>&#9728;</span></button>"
        "</div>"
    )

    html_parts.append(
        f"<div class='meta'><strong>Resumo do Dataset</strong> — "
        f"Linhas: <strong>{n_rows}</strong> | "
        f"Colunas: <strong>{n_cols}</strong> | "
        f"Duplicatas: <strong>{summary['n_duplicates']}</strong> | "
        f"Gerado em: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</div>"
    )

    # Alerts
    html_parts.append("<div class='alerts'><h2>Avisos / Alertas de Qualidade</h2>")
    # Botões de download
    btns: List[str] = []
    if subsets_b64.get("duplicates"):
        btns.append(
            "<button class='badge' "
            f"onclick=\"downloadCSV('{subsets_b64['duplicates']}', 'data_duplicates.csv')\">"
            "Baixar Duplicatas CSV</button>"
        )
    if subsets_b64.get("high_missing_rows"):
        btns.append(
            "<button class='badge' "
            f"onclick=\"downloadCSV('{subsets_b64['high_missing_rows']}', 'data_high_missing_rows.csv')\">"
            "Baixar Linhas com Alto Missing CSV</button>"
        )
    if subsets_b64.get("sample"):
        btns.append(
            "<button class='badge' "
            f"onclick=\"downloadCSV('{subsets_b64['sample']}', 'data_sample.csv')\">"
            "Baixar Amostra (100 linhas) CSV</button>"
        )
    if btns:
        html_parts.append("<div style='margin-bottom:10px;'>" + " ".join(btns) + "</div>")

    if not alerts:
        html_parts.append(
            "<div class='alert-item' "
            "style='background:var(--color-success-bg); "
            "border-color:var(--color-success-border);'>"
            "<strong>Nenhum alerta crítico detectado.</strong> "
            "A qualidade dos dados parece razoável.</div>"
        )
    else:
        for a in alerts:
            html_parts.append(
                "<div class='alert-item'>"
                f"<strong>{a.get('message')}</strong>"
                f"<div class='muted'>Tipo: {a.get('type')}</div>"
                "</div>"
            )
    html_parts.append("</div>")  # .alerts

    # Acordeões
    html_parts.append("<div class='accordion'>")

    # 1) Metadados e amostra
    html_parts.append("<div class='accordion-item'>")
    html_parts.append(
        "<div class='accordion-header' onclick='toggleAccordion(this)'>"
        "Metadados e Amostra <span class='accordion-icon'>&#9658;</span></div>"
    )
    html_parts.append("<div class='accordion-content'>")
    html_parts.append("<h3>Metadados Chave</h3>")
    html_parts.append("<ul>")
    html_parts.append(f"<li>Linhas (N): <strong>{n_rows}</strong></li>")
    html_parts.append(f"<li>Colunas (P): <strong>{n_cols}</strong></li>")
    html_parts.append(
        f"<li>Duplicatas: <strong>{summary['n_duplicates']}</strong> "
        f"({summary['n_duplicates']/max(1, n_rows)*100:.2f}%)</li>"
    )
    if n_cols > 0:
        mean_missing = (df.isna().sum(axis=1) / n_cols).mean() * 100
        html_parts.append(
            f"<li>Missing médio por linha: <strong>{mean_missing:.2f}%</strong></li>"
        )
    html_parts.append("</ul>")
    html_parts.append("<h3>Amostra do Dataset</h3>")
    html_parts.append(_safe_head_html(df))
    html_parts.append("</div></div>")

    # 2) Estatísticas numéricas
    html_parts.append("<div class='accordion-item'>")
    html_parts.append(
        "<div class='accordion-header' onclick='toggleAccordion(this)'>"
        "Estatísticas Numéricas <span class='accordion-icon'>&#9658;</span></div>"
    )
    html_parts.append("<div class='accordion-content'>")
    if num_summary.empty:
        html_parts.append(
            "<div class='muted'>Sem colunas numéricas para sumarizar.</div>"
        )
    else:
        html_parts.append(
            "<input type='text' id='numSearch' "
            "onkeyup=\"filterTable('numSearch','numSummaryTable')\" "
            "placeholder='Filtrar por nome da variável...'>"
        )
        html_parts.append(
            num_summary.to_html(
                border=0, classes="table", table_id="numSummaryTable"
            )
        )
    html_parts.append("</div></div>")

    # 3) Variáveis categóricas
    html_parts.append("<div class='accordion-item'>")
    html_parts.append(
        "<div class='accordion-header' onclick='toggleAccordion(this)'>"
        "Variáveis Categóricas <span class='accordion-icon'>&#9658;</span></div>"
    )
    html_parts.append("<div class='accordion-content'>")
    if not cat_summary:
        html_parts.append(
            "<div class='muted'>Sem colunas categóricas para sumarizar.</div>"
        )
    else:
        for col, info in cat_summary.items():
            html_parts.append(
                f"<h3>{col} "
                f"<span class='muted'>(únicos: {info['n_unique']}, "
                f"missing: {info['n_missing']})</span></h3>"
            )
            top_df = pd.DataFrame(
                list(info["top"].items()), columns=["Valor", "Contagem"]
            )
            html_parts.append(top_df.to_html(border=0, index=False, classes="table"))
    html_parts.append("</div></div>")

    # 4) Visualizações e correlações
    html_parts.append("<div class='accordion-item'>")
    html_parts.append(
        "<div class='accordion-header' onclick='toggleAccordion(this)'>"
        "Visualizações e Correlações <span class='accordion-icon'>&#9658;</span></div>"
    )
    html_parts.append("<div class='accordion-content'>")

    html_parts.append("<h3>Visualizações</h3>")
    html_parts.append("<div class='plot-container'>")
    if images.get("missing_bar"):
        html_parts.append("<div class='plot plot-full'>")
        html_parts.append("<h4>Missing por Coluna (%)</h4>")
        html_parts.append(
            f"<img class='plot-full' src='{images['missing_bar']}' alt='missing'>"
        )
        html_parts.append("</div>")
    if images.get("target_distribution"):
        html_parts.append("<div class='plot plot-small'>")
        html_parts.append("<h4>Distribuição do Target</h4>")
        html_parts.append(
            f"<img class='plot-full' src='{images['target_distribution']}' alt='target'>"
        )
        html_parts.append("</div>")
    if images.get("histograms"):
        html_parts.append("<div class='plot plot-full'>")
        html_parts.append("<h4>Histogramas (amostra de variáveis numéricas)</h4>")
        html_parts.append("<div class='plot-container'>")
        for p in images["histograms"]:
            html_parts.append(
                f"<img class='plot plot-small' src='{p}' alt='hist'>"
            )
        html_parts.append("</div></div>")
    if images.get("corr_heatmap"):
        html_parts.append("<div class='plot plot-full'>")
        html_parts.append("<h4>Mapa de Correlação (variáveis numéricas)</h4>")
        html_parts.append(
            f"<img class='plot-full' src='{images['corr_heatmap']}' alt='corr'>"
        )
        html_parts.append("</div>")
    html_parts.append("</div>")  # plot-container

    html_parts.append("<h3>Top Correlações</h3>")
    if top_pairs:
        html_parts.append(
            "<table><thead><tr><th>Par</th><th>|Correlação|</th></tr></thead><tbody>"
        )
        for p in top_pairs:
            pair = p["pair"]
            v = p["value"]
            html_parts.append(
                f"<tr><td>{pair[0]} ↔ {pair[1]}</td>"
                f"<td><strong>{v:.4f}</strong></td></tr>"
            )
        html_parts.append("</tbody></table>")
    else:
        html_parts.append("<div class='muted'>Sem correlações relevantes.</div>")

    html_parts.append("</div></div>")  # fim vis/corr

    # 5) JSON de resumo
    html_parts.append("<div class='accordion-item'>")
    html_parts.append(
        "<div class='accordion-header' onclick='toggleAccordion(this)'>"
        "JSON de Resumo <span class='accordion-icon'>&#9658;</span></div>"
    )
    html_parts.append("<div class='accordion-content'>")
    html_parts.append(
        "<p class='muted'>Resumo completo em JSON (mesma estrutura do arquivo "
        "<code>data_health_report.json</code>).</p>"
    )
    html_parts.append(
        "<button class='badge' onclick=\"toggle('json_block')\">"
        "Mostrar/Ocultar JSON</button> "
        "<button class='badge' onclick='copyJson()'>Copiar JSON</button>"
    )
    html_parts.append("<pre id='json_block' style='display:none;'>")
    html_parts.append(json.dumps(summary, ensure_ascii=False, indent=2))
    html_parts.append("</pre>")
    html_parts.append("</div></div>")  # JSON

    html_parts.append("</div>")  # .accordion

    # Rodapé
    html_parts.append(
        "<div style='margin-top:32px;padding-top:16px;border-top:1px solid "
        "var(--color-border);font-size:12px;color:var(--color-text-muted);"
        "text-align:center'>"
        "Relatório gerado por <strong>DataKhanon</strong></div>"
    )

    html_parts.append(js)
    html_parts.append("</div></body></html>")

    html_content = "\n".join(html_parts)
    out_path = os.path.join(output_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    summary_out: Dict[str, Any] = {
        "html_report": out_path,
        "json_summary": os.path.join(output_dir, "data_health_report.json")
        if save_json
        else None,
        "alerts": alerts,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
    }
    return summary_out


# ============================================================
# Helpers externos
# ============================================================

def profile_and_open(
    df: pd.DataFrame,
    target: Optional[pd.Series] = None,
    output_dir: str = "artifacts/report",
) -> Dict[str, Any]:
    """
    Utilitário simples para uso direto em notebooks:
    gera relatório e retorna o dicionário de metadados (summary_out).
    """
    return generate_report(df, target=target, output_dir=output_dir)


@dataclass
class DataHealthReport:
    """
    Wrapper orientado a objeto para o gerador de relatório de qualidade dos dados.

    Compatível com o uso no Preprocessor:

        reporter = DataHealthReport(df, target)
        summary = reporter.run(output_dir=..., html=True, json_out=True)

    Internamente delega para `generate_report`.
    """

    df: pd.DataFrame
    target: Optional[pd.Series] = None

    def run(
        self,
        output_dir: str = "artifacts/report",
        html: bool = True,   # mantido por compatibilidade
        json_out: bool = True,
        filename: str = "data_health_report.html",
    ) -> Dict[str, Any]:
        """
        Gera o relatório de *data health* e retorna o dicionário de resumo.

        Parâmetros
        ----------
        output_dir : str
            Diretório onde serão gravados os artefatos (HTML e JSON).
        html : bool
            Ignorado (HTML é sempre gerado).
        json_out : bool
            Se True, grava também o JSON de resumo.
        filename : str
            Nome do arquivo HTML a ser gerado.
        """
        return generate_report(
            self.df,
            target=self.target,
            output_dir=output_dir,
            filename=filename,
            save_json=bool(json_out),
        )
