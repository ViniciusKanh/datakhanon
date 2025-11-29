# DataKhanon — Ferramentas para Pré-Processamento, Visualização e Modelagem de Dados

[![PyPI version](https://img.shields.io/pypi/v/datakhanon.svg)](https://pypi.org/project/datakhanon/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/seu-usuario/datakhanon/actions)

<!-- Tecnologias / “botões” -->
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.5%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.8%2B-11557C?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![plotly](https://img.shields.io/badge/Plotly-opcional-3D58FF?logo=plotly&logoColor=white)](https://plotly.com/)
[![shap](https://img.shields.io/badge/SHAP-opcional-FB5E00?logo=shap&logoColor=white)](https://github.com/slundberg/shap)
[![umap-learn](https://img.shields.io/badge/UMAP-opcional-6B21A8?logo=python&logoColor=white)](https://umap-learn.readthedocs.io/)

---

## Descrição

**DataKhanon** é uma biblioteca Python para construção de pipelines reprodutíveis de ciência de dados. Integra funcionalidades para **pré-processamento** (limpeza, imputação, codificação, engenharia/seleção de features), **visualização / EDA / explainability** (plotagem estática e interativa, relatórios HTML, integração SHAP/UMAP) e **modelagem / experimentação** (wrappers, treinadores, AutoTrainer, persistência de artefatos). O objetivo é facilitar a transição entre protótipo, CI e produção mantendo artefatos auditáveis (`preprocessor.joblib`, `schema.json`, `data_health_report.html`, `model/*.joblib`, `summary.json`).

---

## Recursos principais:

* Normalização de schema (nomes e tipos) e remoção baseada em missingness.  
* Imputação por tipo (numérico / categórico) com opções simples e iterativas.  
* Codificações: One-Hot, Ordinal e Target (com mapeamento persistente).  
* Engenharia de features: escalonadores, `VarianceThreshold`, `SelectKBest`, seleção por importância de modelo.  
* Orquestrador `datakhanon.preprocess.Preprocessor` — interface `fit/transform/save/load`.  
* Geração de relatórios de saúde de dados (`data_health_report.html`) e sumários JSON.  
* Visualizações estáticas (matplotlib/seaborn) e interativas (Plotly/Bokeh) com fallback.  
* Explainability com SHAP e projeções UMAP (quando instalados).  
* `datakhanon.model.AutoTrainer` — CV sobre candidatos, seleção do melhor modelo e export de artefatos.  
* `quick_experiment_from_csv` — atalho end-to-end: CSV → EDA → Preprocess → Treino → Artefatos + `summary.json`.

---

## Instalação

Requisitos mínimos: Python ≥ 3.9.

Instalação básica:

```bash
pip install datakhanon
````

Instalação com extras (EDA / interactive / explainability):

```bash
pip install "datakhanon[viz,interactive,explainer]"
```

Recomenda-se utilizar ambiente virtual (venv / conda).

---

## Quickstart — 3 passos (exemplo mínimo)

```python
import pandas as pd
from datakhanon.preprocess import Preprocessor
from datakhanon.visualize import quick_eda
from datakhanon.model import AutoTrainer

# 1. carregar dados
df = pd.read_csv("examples/credit_dataset_2000.csv")
y = (df["loan_status"] == "Default").astype(int)

# 2. inspeção rápida
quick_eda(df, output_dir="artifacts/eda", target=y)

# 3. preprocess + treino
pp = Preprocessor(categorical_columns=["purpose","housing"],
                  imputer_config={"num_strategy":"median","cat_strategy":"most_frequent"},
                  encoder_config={"ohe":{"drop":"first"}},
                  feature_engineer_config={"scaler":"standard","select_k":20})
X = pp.fit_transform(df, y=y)
pp.save("artifacts/preprocessor.joblib")

trainer = AutoTrainer(output_dir="artifacts/model", cv=3, candidates=["rf","lr","xgb"])
res = trainer.fit(X, y)
print(res["metrics"])
```

---

## `quick_experiment_from_csv` — documentação completa (copy-paste)

**Assinatura (exemplo):**

```python
quick_experiment_from_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    out_dir: str = "outputs/quick_experiment",
    preprocess_config: Optional[dict] = None,
    trainer_config: Optional[dict] = None,
    run_eda: bool = True,
    sample_predictions: int = 10,
    random_state: int = 42,
    overwrite: bool = False
) -> dict
```

**Descrição:** atalho que executa, de forma repetível, o pipeline completo: leitura do CSV, (opcional) EDA via `datakhanon.visualize.quick_eda`, pré-processamento com `datakhanon.preprocess.Preprocessor` (fit e persistência), treinamento e seleção por `datakhanon.model.AutoTrainer`, export de artefatos e construção de `summary.json`.

**Parâmetros importantes:**

* `csv_path`: caminho para o arquivo CSV de entrada.
* `target_col`: nome da coluna alvo (se `None`, tentativa de autodetecção).
* `out_dir`: diretório de saída para todos os artefatos.
* `preprocess_config`: dicionário com parâmetros para `Preprocessor`.
* `trainer_config`: dicionário com parâmetros para `AutoTrainer` (candidates, cv, scoring, etc.).
* `run_eda`: se `True`, gera relatório EDA.
* `sample_predictions`: número de linhas de predição exemplificativa a salvar.

**Fluxo executado internamente (resumido):**

1. valida `csv_path` e carrega pandas DataFrame.
2. detecta `target_col` ou usa o fornecido.
3. gera EDA (se `run_eda=True`).
4. inicializa e executa `datakhanon.preprocess.Preprocessor.fit_transform`; salva `preprocessor.joblib` e `schema.json`.
5. inicializa `datakhanon.model.AutoTrainer` com `trainer_config`, executa CV e seleciona melhor candidato; salva `best_model.joblib`, `best_model_spec.json`, `candidates_cv_results.csv` e `metrics_aggregated.json`.
6. salva `predictions/sample_predictions.csv` com `id, y_true, y_pred, y_score`.
7. monta e salva `summary.json` (retornado também como `dict` em memória).

**Saída em disco (padrão):**

```
out_dir/
├─ preprocessor/preprocessor.joblib
├─ schema.json
├─ eda/data_health_report.html
├─ eda/eda_summary.json
├─ model/best_model.joblib
├─ model/best_model_spec.json
├─ model/candidates_cv_results.csv
├─ model/metrics_aggregated.json
├─ predictions/sample_predictions.csv
├─ logs/run.log
└─ summary.json
```

**Exemplo de uso copy-paste:**

```python
from datakhanon.model.experiment import quick_experiment_from_csv

summary = quick_experiment_from_csv(
    csv_path="examples/credit_dataset_2000.csv",
    target_col="loan_status",
    out_dir="outputs/credit_exp1",
    preprocess_config={
        "categorical_columns":["purpose","housing"],
        "imputer_config":{"num_strategy":"median","cat_strategy":"most_frequent"},
        "encoder_config":{"ohe":{"drop":"first"}},
        "feature_engineer_config":{"scaler":"standard","select_k":20}
    },
    trainer_config={
        "cv": 3,
        "candidates": ["rf","xgb","lr"],
        "scoring": "f1"
    },
    run_eda=True,
    sample_predictions=10,
    random_state=42,
    overwrite=True
)

# 'summary' é um dict Python equivalente ao summary.json salvo.
print(summary["training"]["best_model_name"])
print("EDA salvo em:", summary["paths"]["eda_report"])
```

**Formato de `summary.json` (exemplo resumido):**

```json
{
  "config": { "csv_path": "examples/credit_dataset_2000.csv", "target_col": "loan_status", "trainer_config": {...} },
  "data": { "n_rows": 2000, "n_cols": 45, "class_balance": {"NoDefault": 0.84, "Default": 0.16} },
  "paths": { "preprocessor": "preprocessor/preprocessor.joblib", "schema": "schema.json", "eda_report": "eda/data_health_report.html", "model_dir": "model/" },
  "training": { "best_model_name": "RandomForest", "best_model_path": "model/best_model.joblib", "metrics": {"f1": {"mean":0.701,"std":0.028}, "roc_auc": {"mean":0.812,"std":0.014}} },
  "predictions": { "sample_predictions_path": "predictions/sample_predictions.csv" },
  "run_metadata": { "run_id": "credit_exp1_20251129T150312", "created_at": "2025-11-29T15:03:12Z" }
}
```

**Exemplo de linhas esperadas em `candidates_cv_results.csv`:**

```
candidate,fold,metric_name,metric_value,train_time_s,params
RandomForest,0,f1,0.694,12.3,"{'n_estimators':200}"
RandomForest,1,f1,0.702,11.8,"{'n_estimators':200}"
XGBoost,0,f1,0.681,14.5,"{'max_depth':6}"
...
```

---

## Artefatos gerados (resumo rápido e como usá-los)

* `preprocessor.joblib` — carregar em produção com `datakhanon.preprocess.Preprocessor.load(...)` e aplicar `transform(df_new)`.
* `schema.json` — verificar `retained_columns` e dtypes antes de scoring.
* `data_health_report.html` — relatório auto-contido para auditoria.
* `best_model.joblib` + `best_model_spec.json` — carregar com `datakhanon.model.load_model(...)`.
* `summary.json` — entrada canónica para integração com CI e dashboards.

---

## Exemplos avançados

*Uso em produção — inferência:*

```python
from datakhanon.preprocess import Preprocessor
from datakhanon.model.persistence import load_model
import pandas as pd

pp = Preprocessor.load("outputs/credit_exp1/preprocessor/preprocessor.joblib")
model, spec = load_model("outputs/credit_exp1/model/best_model.joblib")

df_new = pd.read_csv("incoming/new_batch.csv")
pp.validate_input(df_new)     # checar schema
X_new = pp.transform(df_new)
preds = model.predict(X_new)
```

---

# `examples/quickstart.py`

```python
#!/usr/bin/env python3
"""
examples/quickstart.py
Exemplo mínimo de uso do DataKhanon:
- gera EDA
- treina Preprocessor
- executa AutoTrainer
- imprime resumo (summary)

Uso:
    python examples/quickstart.py
"""

import json
from pathlib import Path
import pandas as pd

# Import (API de alto nível)
from datakhanon.preprocess import Preprocessor
from datakhanon.visualize import quick_eda
from datakhanon.model.experiment import quick_experiment_from_csv
from datakhanon.model import AutoTrainer  # opcional, uso direto

# Ajuste: caminhos
ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CSV = ROOT / "examples" / "credit_dataset_2000.csv"
OUT_DIR = ROOT / "outputs" / "quickstart_example"

def run_quick_example():
    # 1) Carregar dados
    if not EXAMPLE_CSV.exists():
        raise FileNotFoundError(f"Arquivo de exemplo não encontrado: {EXAMPLE_CSV}")
    df = pd.read_csv(EXAMPLE_CSV)
    # Exemplo: converter label para binário
    if "loan_status" not in df.columns:
        raise KeyError("Coluna 'loan_status' esperada no dataset de exemplo.")
    y = (df["loan_status"] == "Default").astype(int)

    # 2) Quick EDA (gera artifacts/eda)
    print("Gerando EDA rápido...")
    quick_eda(df, output_dir=str(OUT_DIR / "eda"), target=y, use_reporter=True)

    # 3) Executar quick_experiment_from_csv (end-to-end)
    print("Executando quick_experiment_from_csv (treino completo)...")
    summary = quick_experiment_from_csv(
        csv_path=str(EXAMPLE_CSV),
        target_col="loan_status",
        out_dir=str(OUT_DIR),
        preprocess_config={
            "categorical_columns": ["purpose", "housing"],
            "imputer_config": {"num_strategy": "median", "cat_strategy": "most_frequent"},
            "encoder_config": {"ohe": {"drop": "first"}},
            "feature_engineer_config": {"scaler": "standard", "select_k": 20}
        },
        trainer_config={
            "cv": 3,
            "candidates": ["rf", "xgb", "lr"],
            "scoring": "f1"
        },
        run_eda=False,            # já rodamos acima
        sample_predictions=10,
        random_state=42,
        overwrite=True
    )

    # 4) Salvar e imprimir resumo
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary_inspect.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Resumo do experimento salvo em:", summary_path)
    print("Melhor modelo:", summary["training"].get("best_model_name"))
    print("EDA report:", summary["paths"].get("eda_report"))
    print("Modelo salvo em:", summary["training"].get("best_model_path"))

if __name__ == "__main__":
    run_quick_example()
````
## Licença e créditos

Licença: MIT.
Autor: **Vinicius de Souza Santos** — Mestrado em Ciências da Computação (UNESP Bauru).


