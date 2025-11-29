# datakhanon/visualize/interactive.py
"""
Plotagens interativas com Plotly / Bokeh (fallback para matplotlib).
Exporta HTML auto-contido. Docstrings e mensagens em Português (BR).
"""

from typing import Optional, List, Any, Dict
import os
import io
import base64
import pandas as pd

# tentar importar plotly e bokeh; se ausentes, usar matplotlib
try:
    import plotly.express as px  # type: ignore
    import plotly.graph_objects as go  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

try:
    from bokeh.plotting import figure, output_file, save as bokeh_save  # type: ignore
    from bokeh.embed import file_html  # type: ignore
    from bokeh.resources import CDN  # type: ignore
    _HAS_BOKEH = True
except Exception:
    _HAS_BOKEH = False

import matplotlib.pyplot as plt  # usado como fallback
plt.switch_backend("Agg")


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _fig_to_html_base64(fig_bytes: bytes) -> str:
    """Retorna data-uri para embedding em HTML."""
    data = base64.b64encode(fig_bytes).decode("ascii")
    return f"data:text/html;base64,{data}"


def histogram_interactive(
    df: pd.DataFrame,
    column: str,
    nbins: int = 50,
    title: Optional[str] = None,
    output_html: Optional[str] = None,
    interactive_engine: str = "plotly",
    **px_kwargs
) -> Dict[str, Any]:
    """
    Gera histograma interativo para 'column'. Retorna dicionário com keys:
      - 'html' (str) se output_html não fornecido: HTML string auto-contido
      - 'path' (str) se output_html fornecido: caminho salvo
      - 'engine' (str) engine usada
    interactive_engine: 'plotly' or 'bokeh' (se disponível)
    """
    title = title or f"Distribuição — {column}"
    if interactive_engine == "plotly" and _HAS_PLOTLY:
        fig = px.histogram(df, x=column, nbins=nbins, title=title, **px_kwargs)
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        if output_html:
            _ensure_dir(output_html)
            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html)
            return {"path": output_html, "engine": "plotly"}
        return {"html": html, "engine": "plotly"}
    if interactive_engine == "bokeh" and _HAS_BOKEH:
        p = figure(title=title, tools="pan,wheel_zoom,box_zoom,reset,save")
        vals = df[column].dropna().values
        hist, edges = np.histogram(vals, bins=nbins)
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.7)
        html = file_html(p, CDN, title)
        if output_html:
            _ensure_dir(output_html)
            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html)
            return {"path": output_html, "engine": "bokeh"}
        return {"html": html, "engine": "bokeh"}
    # fallback matplotlib -> salva PNG embutido em HTML
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df[column].dropna(), bins=nbins)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    html = f"<html><body><h3>{title}</h3><img src='data:image/png;base64,{img_b64}'/></body></html>"
    if output_html:
        _ensure_dir(output_html)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
        return {"path": output_html, "engine": "matplotlib-fallback"}
    return {"html": html, "engine": "matplotlib-fallback"}


def scatter_interactive(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    title: Optional[str] = None,
    output_html: Optional[str] = None,
    interactive_engine: str = "plotly",
    sample_frac: Optional[float] = 0.2,
    max_points: int = 20000,
    **px_kwargs
) -> Dict[str, Any]:
    """
    Gera scatter interativo (plotly/bokeh); faz amostragem se grande.
    """
    title = title or f"{y} vs {x}"
    n = len(df)
    if sample_frac is not None and n > max_points:
        df_plot = df.sample(frac=sample_frac, random_state=1)
    else:
        df_plot = df
    if interactive_engine == "plotly" and _HAS_PLOTLY:
        fig = px.scatter(df_plot, x=x, y=y, color=color, title=title, hover_data=df_plot.columns.tolist(), **px_kwargs)
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        if output_html:
            _ensure_dir(output_html)
            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html)
            return {"path": output_html, "engine": "plotly"}
        return {"html": html, "engine": "plotly"}
    # fallback: matplotlib static PNG embedded
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df_plot[x], df_plot[y], s=6, alpha=0.6)
    ax.set_title(title)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("ascii")
    html = f"<html><body><h3>{title}</h3><img src='data:image/png;base64,{img_b64}'/></body></html>"
    if output_html:
        _ensure_dir(output_html)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(html)
        return {"path": output_html, "engine": "matplotlib-fallback"}
    return {"html": html, "engine": "matplotlib-fallback"}
