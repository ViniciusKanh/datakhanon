# datakhanon/model/metrics.py
"""
Módulo de métricas padrão do DataKhanon.

Objetivo:
    - Fornecer funções de métricas robustas e genéricas para:
        * Classificação (binária e multiclasse, labels numéricos ou não).
        * Regressão.
    - Interface única via `auto_metrics(task, y_true, y_pred, y_score=None)`.

Convenções:
    - y_true: vetor 1D com rótulos verdadeiros.
    - y_pred: vetor 1D com rótulos previstos (classe).
    - y_score (opcional): scores ou probabilidades.
        * Binário: shape (n_samples,) ou (n_samples, 2).
        * Multiclasse: shape (n_samples, n_classes) com probabilidades por classe.
"""

from typing import Dict, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# CLASSIFICAÇÃO
# ============================================================

def classification_metrics(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula métricas de classificação de forma robusta a:
    - labels numéricos ou string;
    - problema binário ou multiclasse;
    - presença ou não de y_score.

    Retorna um dicionário com:
        - accuracy
        - precision
        - recall
        - f1
        - roc_auc (quando possível)
        - log_loss (quando possível)
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    # Detecta configuração de labels
    labels = np.unique(y_true_arr)
    n_classes = len(labels)
    is_binary = n_classes == 2
    # "binário clássico" se labels são {0,1}
    binary_01 = is_binary and set(labels).issubset({0, 1})

    # Estratégia de agregação:
    # - Se binário com {0,1} -> average="binary"
    # - Caso contrário -> average="weighted" (mais robusto a desbalanceamento)
    avg = "binary" if binary_01 else "weighted"

    # Métricas básicas
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=avg,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                average=avg,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average=avg,
                zero_division=0,
            )
        ),
    }

    # ------------------------
    # ROC AUC (quando possível)
    # ------------------------
    auc_value = None
    if y_score is not None:
        try:
            y_score_arr = np.asarray(y_score)

            # Caso 1: binário com labels {0,1}
            if binary_01:
                if y_score_arr.ndim == 1:
                    # y_score é probabilidade da classe positiva
                    auc_value = roc_auc_score(y_true_arr, y_score_arr)
                elif y_score_arr.ndim == 2 and y_score_arr.shape[1] >= 2:
                    # assume prob. da classe positiva na última coluna
                    auc_value = roc_auc_score(y_true_arr, y_score_arr[:, -1])

            # Caso 2: classificação geral (multi-classe ou binário com labels não numéricos)
            else:
                # Se for matriz de probabilidades (n_samples, n_classes),
                # usa AUC multi-classe (one-vs-rest ou one-vs-one).
                if y_score_arr.ndim == 2 and y_score_arr.shape[1] >= 2:
                    # Estratégia: OVO macro AUC
                    auc_value = roc_auc_score(
                        y_true_arr,
                        y_score_arr,
                        multi_class="ovo",
                        average="macro",
                    )
                # Se for 1D mas labels não são {0,1}, não arriscamos usar AUC binário
                # para evitar exceções de pos_label; nesse caso, deixamos sem AUC.
        except Exception:
            auc_value = None

    metrics["roc_auc"] = float(auc_value) if auc_value is not None else None

    # ------------------------
    # LOG-LOSS (quando possível)
    # ------------------------
    ll_value = None
    if y_score is not None:
        try:
            ll_value = log_loss(y_true_arr, y_score)
        except Exception:
            ll_value = None

    metrics["log_loss"] = float(ll_value) if ll_value is not None else None

    return metrics


# ============================================================
# REGRESSÃO
# ============================================================

def regression_metrics(
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Métricas padrão para regressão.

    Retorna:
        - mse  : mean squared error
        - rmse : raiz do MSE
        - mae  : erro absoluto médio
        - r2   : R-quadrado
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    mse = mean_squared_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mse))

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


# ============================================================
# DETECÇÃO AUTOMÁTICA DE TAREFA (fallback)
# ============================================================

def _infer_task_from_y(y_true) -> str:
    """
    Inferência simples do tipo de tarefa a partir de y_true:

        - Se dtype for float e nº de valores únicos for grande -> 'regression'
        - Se nº de valores únicos for pequeno (<= 20) -> 'classification'
        - Caso ambíguo → fallback para 'classification' (por segurança).

    Essa função é usada apenas quando o usuário passa task="auto"
    ou task inesperado em auto_metrics.
    """
    y = np.asarray(y_true)
    # nº de valores únicos (ignorando NaN)
    uniq = np.unique(y[~pd.isna(y)]) if "pandas" in str(type(y_true)) else np.unique(y)
    n_unique = len(uniq)

    # Heurística simples: muitos valores únicos -> regressão
    if y.dtype.kind in ("f", "i") and n_unique > 20:
        return "regression"

    # Poucos valores únicos -> classificação
    if n_unique <= 20:
        return "classification"

    # fallback: problema mais comum é classificação
    return "classification"


# ============================================================
# DISPATCH ÚNICO
# ============================================================

def auto_metrics(
    task: str,
    y_true,
    y_pred,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Dispatcher de métricas por tipo de tarefa.

    Parâmetros
    ----------
    task : str
        Tipo de tarefa, esperado:
            - 'binary'
            - 'multiclass'
            - 'classification'
            - 'regression'
            - 'auto'  -> detecção automática a partir de y_true
        Qualquer outro valor também cai na detecção automática.

    y_true, y_pred:
        Vetores de verdade / predição.

    y_score:
        Scores ou probabilidades (opcional) para cálculo de ROC AUC e log-loss.

    Retorno
    -------
    Dict[str, float]
        Dicionário com métricas apropriadas ao tipo de tarefa.
    """
    # Normaliza nome da tarefa
    task_norm = (task or "").lower()

    # Classificação explícita
    if task_norm in ("binary", "multiclass", "classification", "clf", "class"):
        return classification_metrics(y_true, y_pred, y_score)

    # Regressão explícita
    if task_norm in ("regression", "reg", "continuous"):
        return regression_metrics(y_true, y_pred, y_score)

    # Caso contrário, tenta inferir automaticamente
    inferred = _infer_task_from_y(y_true)

    if inferred == "regression":
        return regression_metrics(y_true, y_pred, y_score)
    else:
        return classification_metrics(y_true, y_pred, y_score)
