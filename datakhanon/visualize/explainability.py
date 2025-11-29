# datakhanon/visualize/explainability.py
"""
Integração com SHAP para gerar relatórios explicáveis.
Funções:
- shap_summary_html(model, X, output_path)
- shap_force_html(model, X_row, output_path)
Graceful fallback se 'shap' não estiver instalado.
"""

from typing import Optional, Any
import os
import json
import base64
import io
import pandas as pd

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

import matplotlib.pyplot as plt
plt.switch_backend("Agg")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def shap_summary_html(model: Any, X: pd.DataFrame, output_path: str, nsample: int = 200):
    """
    Gera SHAP summary plot e salva HTML auto-contido em output_path.
    nsample: número de amostras para acelerar (se X muito grande).
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap não instalado. Instale com `pip install shap` para usar explainability.")
    Xs = X.sample(n=min(nsample, len(X)), random_state=1) if len(X) > nsample else X
    explainer = shap.Explainer(model, Xs, silent=True)
    shap_values = explainer(Xs)
    # gerar plot summary (matplotlib) e salvar como PNG embutido em HTML
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, Xs, show=False)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    html = f"<html><body><h2>SHAP summary</h2><img src='data:image/png;base64,{img_b64}'/></body></html>"
    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def shap_force_html(model: Any, X_row: pd.DataFrame, output_path: str):
    """
    Gera SHAP force plot para uma linha (X_row: DataFrame com 1 linha) e salva HTML.
    Observação: dependendo do explainer e modelo, o force_plot pode gerar HTML separado.
    """
    if not _HAS_SHAP:
        raise RuntimeError("shap não instalado. Instale com `pip install shap` para usar explainability.")
    # criar explainer com um sample pequeno (necessário para alguns explainers)
    explainer = shap.Explainer(model, X_row, silent=True)
    shap_values = explainer(X_row)
    # try force_plot (HTML)
    try:
        html_str = shap.plots.force(shap_values[0], matplotlib=False, show=False)
        # shap.plots.force may return an HTML component; fallback to shap.plots._force
        # If above returns widget, save using shap.save_html (if available)
        from shap import save_html  # type: ignore
        _ensure_dir(output_path)
        save_html(output_path, html_str)
        return output_path
    except Exception:
        # fallback: create a static waterfall plot
        plt.figure(figsize=(6, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close()
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("ascii")
        html = f"<html><body><h2>SHAP (waterfall fallback)</h2><img src='data:image/png;base64,{img_b64}'/></body></html>"
        _ensure_dir(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        return output_path
