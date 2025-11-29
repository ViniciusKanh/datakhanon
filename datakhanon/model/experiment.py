# datakhanon/model/experiment.py
"""
Módulo de alto nível para rodar experimentos rápidos com o DataKhanon.

Objetivo
--------
Oferecer uma API concisa para:

    * Carregar dados (CSV ou DataFrame).
    * Detectar coluna alvo (opcional).
    * Pré-processar com Preprocessor.
    * Gerar EDA com quick_eda.
    * Treinar modelo com AutoTrainer.
    * Salvar modelo e resumo em disco (JSON).

Uso típico
----------
from datakhanon.model import quick_experiment_from_csv

summary = quick_experiment_from_csv(
    csv_path="credit_dataset_2000.csv",
    target_col="loan_status",   # ou None para auto-detecção
    out_dir="outputs/exp_credit",
    model="auto",               # ou "rf", "xgb", etc., conforme AutoTrainer
    tune=False,
)

print("Métricas:", summary["metrics"])
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

import json
import numpy as np
import pandas as pd

from datakhanon.preprocess.preprocessor import Preprocessor
from datakhanon.visualize.eda import quick_eda
from datakhanon.model.auto_trainer import AutoTrainer
from datakhanon.model.persistence import load_model


# ============================================================
# Helpers internos
# ============================================================

def _nowstr() -> str:
    """Timestamp compacto para nomes de arquivos."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _detect_target_column(df: pd.DataFrame, user_target: Optional[str] = None) -> str:
    """
    Heurística simples para detectar a coluna alvo.

    Regras:
        - Se user_target for fornecido e existir no DataFrame, usa.
        - Caso contrário, tenta nomes comuns.
        - Fallback: última coluna.
    """
    if user_target and user_target in df.columns:
        return user_target

    commons = [
        "target",
        "label",
        "y",
        "classe",
        "status",
        "inadimplente",
        "loan_status",
    ]
    for c in commons:
        if c in df.columns:
            return c

    return df.columns[-1]


def _to_json_safe(obj: Any) -> Any:
    """
    Converte objetos arbitrários em algo serializável por json.dumps.

    Regras:
        - None -> None
        - numpy types -> tipos nativos Python
        - pandas Timestamp/Series/DataFrame -> representações textuais/estruturadas
        - Se tiver método to_dict(), usa.
        - Senão, tenta __dict__.
        - Senão, usa str(obj).
    """
    if obj is None:
        return None

    # numpy escalares
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # pandas básicos
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="list")

    # objetos com to_dict()
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return obj.to_dict()
        except Exception:
            pass

    # objetos simples com __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                k: _to_json_safe(v)
                for k, v in obj.__dict__.items()
                if not k.startswith("_")
            }
        except Exception:
            pass

    # fallback: string
    return str(obj)


def _pretty_print_header(title: str) -> None:
    """Imprime um cabeçalho estilizado no console."""
    line = "=" * max(40, len(title) + 10)
    print(f"\n{line}")
    print(f"[DataKhanon] {title}")
    print(f"{line}")


# ============================================================
# Estrutura de configuração
# ============================================================

@dataclass
class QuickExperimentConfig:
    """
    Configuração de alto nível para um experimento rápido com DataKhanon.

    Parâmetros principais
    ---------------------
    df : pd.DataFrame | None
        Se fornecido, o experimento usa diretamente esse DataFrame.
    csv_path : str | None
        Se df for None, os dados serão carregados de um arquivo CSV.
    target_col : str | None
        Nome da coluna alvo. Se None, a coluna é detectada automaticamente.
    out_dir : str
        Diretório base onde serão salvos EDA, modelo e resumo.

    Controle de pré-processamento
    -----------------------------
    drop_missing_threshold_prop : float
        Proporção de missing acima da qual uma coluna é descartada
        pelo Preprocessor.

    Controle do AutoTrainer
    -----------------------
    model : str
        String de modelo aceita por AutoTrainer (ex.: "auto", "rf", etc.).
    cv : int
        Número de folds de validação cruzada.
    random_state : int
        Seed para reprodutibilidade.
    max_candidates : int
        Máximo de candidatos testados (quando model="auto").
    tune : bool
        Se True, habilita tunagem adicional, se implementada pelo AutoTrainer.

    Outras opções
    -------------
    run_eda : bool
        Se True, executa quick_eda.
    export_on_finish : bool
        Se True, AutoTrainer exporta o modelo ao final.
    export_formats : list[str] | None
        Formatos a exportar (None deixa o AutoTrainer decidir).
    sample_predictions : int
        Número de linhas para gerar predições de exemplo (0 para não gerar).
    """

    # Entrada de dados
    df: Optional[pd.DataFrame] = None
    csv_path: Optional[str] = None
    target_col: Optional[str] = None

    # Saída
    out_dir: str = "outputs/datakhanon_quick_experiment"

    # Pré-processamento
    drop_missing_threshold_prop: float = 0.95

    # Treinador
    model: str = "auto"
    cv: int = 3
    random_state: int = 42
    max_candidates: int = 4
    tune: bool = False

    # EDA e export
    run_eda: bool = True
    export_on_finish: bool = True
    export_formats: Optional[list[str]] = None

    # Predições de exemplo
    sample_predictions: int = 5


# ============================================================
# Função principal de orquestração
# ============================================================

def run_quick_experiment(config: QuickExperimentConfig) -> Dict[str, Any]:
    """
    Executa o pipeline completo de experimento rápido com base em QuickExperimentConfig.

    Pipeline
    --------
    1. Carrega dados (df ou csv_path).
    2. Detecta coluna alvo.
    3. Pré-processa com Preprocessor.
    4. (Opcional) Gera EDA com quick_eda.
    5. Treina modelo com AutoTrainer.
    6. (Opcional) Gera predições de exemplo.
    7. Salva resumo em JSON.

    Retorno
    -------
    Dict[str, Any]
        Dicionário de resumo com caminhos de saída, métricas e especificações.
    """
    if config.df is None and not config.csv_path:
        raise ValueError("É necessário fornecer 'df' OU 'csv_path' em QuickExperimentConfig.")

    _pretty_print_header("Inicializando experimento rápido")

    # --------------------------------------------------------
    # Diretórios
    # --------------------------------------------------------
    out_base = Path(config.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)
    eda_dir = out_base / "eda"
    model_dir = out_base / "model"
    eda_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Carregamento de dados
    # --------------------------------------------------------
    if config.df is not None:
        df = config.df.copy()
        csv_source = None
        print("[DataKhanon] Fonte de dados: DataFrame em memória")
    else:
        df = pd.read_csv(config.csv_path)  # type: ignore[arg-type]
        csv_source = str(config.csv_path)
        print(f"[DataKhanon] Fonte de dados: CSV -> {csv_source}")

    # --------------------------------------------------------
    # Detecção da coluna alvo
    # --------------------------------------------------------
    target_col = _detect_target_column(df, config.target_col)
    y = df[target_col]
    print(f"[DataKhanon] Dataset shape: {df.shape}  |  Target: '{target_col}'")

    # --------------------------------------------------------
    # Pré-processamento
    # --------------------------------------------------------
    print("[DataKhanon] Etapa: Pré-processamento (cleaning, imputação, encoding, FE)")
    pp = Preprocessor(drop_missing_threshold_prop=config.drop_missing_threshold_prop)
    X = pp.fit_transform(df, y)
    print(f"[DataKhanon] Pós-preprocessamento: X.shape={X.shape}, y.shape={y.shape}")

    # --------------------------------------------------------
    # EDA
    # --------------------------------------------------------
    eda_res: Optional[Dict[str, Any]] = None
    if config.run_eda:
        print("[DataKhanon] Etapa: EDA / Data Health Report")
        eda_res = quick_eda(df, output_dir=str(eda_dir), target=y, use_reporter=True)
        print(f"[DataKhanon] EDA gerado em: {eda_res}")
    else:
        print("[DataKhanon] EDA desabilitado (run_eda=False)")

    # --------------------------------------------------------
    # Treino automático (AutoTrainer)
    # --------------------------------------------------------
    _pretty_print_header("Treinando modelo com AutoTrainer")

    trainer = AutoTrainer(
        model=config.model,
        output_dir=str(model_dir),
        export_on_finish=config.export_on_finish,
        export_formats=config.export_formats,
        cv=config.cv,
        random_state=config.random_state,
        max_candidates=config.max_candidates,
    )

    print(f"[DataKhanon] Iniciando treino (model='{config.model}', cv={config.cv}, tune={config.tune})...")
    res_train = trainer.fit(
        X.values if hasattr(X, "values") else X,
        y.values if hasattr(y, "values") else y,
        X_val=None,
        y_val=None,
        tune=config.tune,
    )
    print(
        f"[DataKhanon] Treino finalizado. "
        f"model_name={res_train.get('model_name')}, metrics={res_train.get('metrics')}"
    )

    # --------------------------------------------------------
    # Carregar modelo salvo e gerar predições de exemplo
    # --------------------------------------------------------
    sample_preds = None
    loaded_model_spec = None

    model_saved_at = res_train.get("model_saved_at")
    model_dir_effective = Path(model_saved_at) if model_saved_at else model_dir

    if config.sample_predictions > 0 and model_dir_effective.exists():
        print("[DataKhanon] Etapa: Carregando modelo salvo e gerando predições de exemplo")
        model_obj, spec = load_model(str(model_dir_effective))
        loaded_model_spec = spec

        # tira amostra de X
        if hasattr(X, "iloc"):
            sample_X = X.iloc[: config.sample_predictions]
        else:
            sample_X = X[: config.sample_predictions]

        try:
            preds = model_obj.predict(sample_X.values)
        except Exception:
            preds = model_obj.predict(sample_X)

        # converte para lista nativa para caber bem em JSON
        if isinstance(preds, np.ndarray):
            sample_preds = preds.tolist()
        else:
            sample_preds = list(preds)

        print(f"[DataKhanon] Predições de exemplo ({config.sample_predictions} linhas): {sample_preds}")
    else:
        print("[DataKhanon] Predições de exemplo desabilitadas ou diretório de modelo inexistente.")

    # --------------------------------------------------------
    # Resumo final (estrutura Python)
    # --------------------------------------------------------
    summary: Dict[str, Any] = {
        "config": asdict(config),
        "data": {
            "csv_source": csv_source,
            "target_col": target_col,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
        },
        "paths": {
            "output_dir": str(out_base),
            "eda_dir": str(eda_dir) if config.run_eda else None,
            "model_dir": str(model_dir_effective),
        },
        "eda": eda_res,
        "training": {
            "metrics": res_train.get("metrics"),
            "model_name": res_train.get("model_name"),
            "model_saved_at": res_train.get("model_saved_at"),
        },
        "model_spec": _to_json_safe(loaded_model_spec),
        "sample_predictions": sample_preds,
    }

    # --------------------------------------------------------
    # Salvamento em JSON (serialização segura)
    # --------------------------------------------------------
    summary_json_safe = _to_json_safe(summary)
    summary_path = out_base / f"quick_experiment_summary_{_nowstr()}.json"
    summary_path.write_text(
        json.dumps(summary_json_safe, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _pretty_print_header("Experimento concluído")
    print(f"[DataKhanon] Resumo salvo em: {summary_path}")
    print(f"[DataKhanon] Métricas principais: {res_train.get('metrics')}")

    return summary


# ============================================================
# Funções de conveniência
# ============================================================

def quick_experiment_from_csv(
    csv_path: str,
    target_col: Optional[str] = None,
    out_dir: str = "outputs/dk_quick_experiment",
    model: str = "auto",
    tune: bool = False,
    drop_missing_threshold_prop: float = 0.95,
    run_eda: bool = True,
    sample_predictions: int = 5,
) -> Dict[str, Any]:
    """
    Atalho para rodar um experimento completo a partir de um CSV.

    Parâmetros mais usados são expostos diretamente; o restante usa os defaults
    do QuickExperimentConfig.
    """
    cfg = QuickExperimentConfig(
        df=None,
        csv_path=csv_path,
        target_col=target_col,
        out_dir=out_dir,
        drop_missing_threshold_prop=drop_missing_threshold_prop,
        model=model,
        tune=tune,
        run_eda=run_eda,
        sample_predictions=sample_predictions,
    )
    return run_quick_experiment(cfg)


def quick_experiment_from_df(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    out_dir: str = "outputs/dk_quick_experiment",
    model: str = "auto",
    tune: bool = False,
    drop_missing_threshold_prop: float = 0.95,
    run_eda: bool = True,
    sample_predictions: int = 5,
) -> Dict[str, Any]:
    """
    Atalho para rodar um experimento completo a partir de um DataFrame em memória.
    """
    cfg = QuickExperimentConfig(
        df=df,
        csv_path=None,
        target_col=target_col,
        out_dir=out_dir,
        drop_missing_threshold_prop=drop_missing_threshold_prop,
        model=model,
        tune=tune,
        run_eda=run_eda,
        sample_predictions=sample_predictions,
    )
    return run_quick_experiment(cfg)
