# datakhanon/model/__init__.py
"""
datakhanon.model
================

Submódulo responsável pela camada de modelagem do DataKhanon, incluindo:

- Abstrações de modelo e especificação (`BaseModel`, `ModelSpec`, `Serializable`);
- Registro e recuperação de modelos (`register_model`, `get_model`, `available_models`);
- Wrappers para bibliotecas externas (`SKLearnWrapper`);
- Treinadores genéricos (`Trainer`, `TrainerExtended`) e treinador automático (`AutoTrainer`);
- Geração de relatórios de treino (`generate_report`);
- Persistência de modelos (`save_model`, `load_model`, `list_models`);
- Métricas padrão para classificação e regressão (`classification_metrics`, `regression_metrics`, `auto_metrics`);
- API de alto nível para experimentos rápidos (`QuickExperimentConfig`, `run_quick_experiment`,
  `quick_experiment_from_csv`, `quick_experiment_from_df`).
"""

from .base import BaseModel, Serializable
from .model_spec import ModelSpec
from .registry import register_model, get_model, available_models
from .wrappers import SKLearnWrapper
from .trainer import Trainer, TrainerExtended
from .report import generate_report
from .persistence import save_model, load_model, list_models
from .auto_trainer import AutoTrainer
from .metrics import classification_metrics, regression_metrics, auto_metrics
from .experiment import (
    QuickExperimentConfig,
    run_quick_experiment,
    quick_experiment_from_csv,
    quick_experiment_from_df,
)

__all__ = [
    # Núcleo de modelos / especificação
    "BaseModel",
    "Serializable",
    "ModelSpec",

    # Registro de modelos
    "register_model",
    "get_model",
    "available_models",

    # Wrappers
    "SKLearnWrapper",

    # Treinadores
    "Trainer",
    "TrainerExtended",
    "AutoTrainer",

    # Relatórios de treino
    "generate_report",

    # Persistência
    "save_model",
    "load_model",
    "list_models",

    # Métricas
    "classification_metrics",
    "regression_metrics",
    "auto_metrics",

    # Experimentos de alto nível
    "QuickExperimentConfig",
    "run_quick_experiment",
    "quick_experiment_from_csv",
    "quick_experiment_from_df",
]
