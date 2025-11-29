# examples/visualize_streamlit_app.py
"""
Dashboard Streamlit mínimo para DataKhanon visualize:
- Carrega CSV (ex: examples/credit_dataset_2000.csv)
- Exibe quick EDA (usar visualize.quick_eda or reporter)
- Mostra plots interativos (histogram, scatter)
- Se model.joblib fornecido, exibe SHAP summary
Execute: streamlit run examples/visualize_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import os

from datakhanon.visualize.interactive import histogram_interactive, scatter_interactive
from datakhanon.visualize.explainability import shap_summary_html
from datakhanon.preprocess.reporter import generate_report

st.set_page_config(page_title="DataKhanon — Visualize Dashboard", layout="wide")

st.title("DataKhanon — Visualize (Dashboard)")

uploaded = st.file_uploader("Carregue um CSV", type=["csv"])
if uploaded is None:
    default = os.path.join("examples", "credit_dataset_2000.csv")
    if os.path.exists(default):
        st.info(f"Nenhum arquivo carregado — usando exemplo {default}")
        df = pd.read_csv(default)
    else:
        st.warning("Carregue um CSV ou coloque `examples/credit_dataset_2000.csv` no repositório.")
        st.stop()
else:
    df = pd.read_csv(uploaded)

st.write("Dimensões:", df.shape)
if st.checkbox("Mostrar amostra"):
    st.dataframe(df.head(100))

# quick EDA via reporter (gera HTML e mostra link)
if st.button("Gerar relatório HTML (reporter)"):
    out = generate_report(df, output_dir="artifacts/streamlit_report")
    st.success("Relatório gerado")
    st.write("HTML:", out.get("html_report"))
    st.write("JSON:", out.get("json_summary"))

# plots interativos
st.header("Plots interativos")
cols = df.select_dtypes(include=["number"]).columns.tolist()
if cols:
    col = st.selectbox("Escolha coluna para histograma", cols, index=0)
    res = histogram_interactive(df, column=col, output_html=None)
    if "html" in res:
        st.components.v1.html(res["html"], height=420, scrolling=True)
    else:
        st.write("Relatório salvo em:", res.get("path"))
else:
    st.info("Sem colunas numéricas para histogramas.")

st.subheader("Scatter interativo")
cols_xy = df.select_dtypes(include=["number"]).columns.tolist()
if len(cols_xy) >= 2:
    x = st.selectbox("x", cols_xy, index=0)
    y = st.selectbox("y", cols_xy, index=1)
    color_options = [None] + df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    color = st.selectbox("Color (opcional)", color_options)
    res = scatter_interactive(df, x=x, y=y, color=color, output_html=None)
    if "html" in res:
        st.components.v1.html(res["html"], height=600, scrolling=True)
else:
    st.info("Precisamos de pelo menos 2 colunas numéricas para scatter.")

# SHAP (opcional)
st.header("Explainability (SHAP) — opcional")
model_path = st.text_input("Caminho do modelo joblib (opcional)", value="")
if model_path:
    if os.path.exists(model_path):
        st.info("Modelo encontrado — gerando SHAP summary (pode demorar)")
        try:
            import joblib
            model = joblib.load(model_path)
            # selecionar até 200 amostras para explicar
            X = df.select_dtypes(include=["number"]).dropna().head(500)
            outpath = os.path.join("artifacts", "streamlit_shap_summary.html")
            shap_summary_html(model, X, output_path=outpath)
            with open(outpath, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Erro ao gerar SHAP: {e}")
    else:
        st.warning("Caminho do modelo inválido.")
