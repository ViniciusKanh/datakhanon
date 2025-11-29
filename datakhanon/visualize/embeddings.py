# datakhanon/visualize/embeddings.py
"""
Projeções de embeddings: UMAP e t-SNE, com saída Plotly interativa (quando disponível).
"""

from typing import Optional, Any
import os
import pandas as pd
import numpy as np

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

from datakhanon.visualize.interactive import scatter_interactive as _scatter_interactive
try:
    import plotly.express as px  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False


def umap_projection_to_html(X: pd.DataFrame, n_components: int = 2, output_html: Optional[str] = None, label: Optional[pd.Series] = None):
    if not _HAS_UMAP:
        raise RuntimeError("umap não instalado. `pip install umap-learn`")
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    proj = reducer.fit_transform(X)
    dfp = pd.DataFrame(proj, columns=[f"dim_{i+1}" for i in range(n_components)])
    if label is not None:
        dfp["_label"] = label.values
    if _HAS_PLOTLY:
        fig = px.scatter(dfp, x="dim_1", y="dim_2", color="_label" if "_label" in dfp.columns else None, title="UMAP projection")
        html = fig.to_html(full_html=True, include_plotlyjs="cdn")
        if output_html:
            os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
            with open(output_html, "w", encoding="utf-8") as f:
                f.write(html)
            return {"path": output_html}
        return {"html": html}
    else:
        # fallback: use scatter_interactive (will generate static PNG embedded)
        dfp_full = dfp.copy()
        return _scatter_interactive(dfp_full, x="dim_1", y="dim_2", output_html=output_html)
