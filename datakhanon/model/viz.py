# datakhanon/model/viz.py
from typing import Optional, Sequence
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

def _try_import_pyplot():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None

def plot_confusion_matrix(y_true, y_pred, labels: Optional[Sequence] = None, out_path: Optional[str] = None):
    plt = _try_import_pyplot()
    if plt is None:
        return None
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title("Matriz de Confusão")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel('Verdadeiro')
    ax.set_xlabel('Previsto')

    thresh = cm.max() / 2.0 if cm.max() != 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", fontsize=10)

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig

def plot_roc_curve(y_true, y_score, out_path: Optional[str] = None, pos_label=None):
    plt = _try_import_pyplot()
    if plt is None:
        return None
    y_true_arr = np.array(y_true)
    classes = np.unique(y_true_arr)
    multiclass = len(classes) > 2

    fig, ax = plt.subplots()
    if multiclass:
        try:
            y_true_bin = label_binarize(y_true_arr, classes=classes)
            if getattr(y_score, "ndim", 1) == 1:
                raise ValueError("y_score deve ter shape (n_samples, n_classes) para multiclass")
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'ROC micro (AUC = {roc_auc:.3f})')
        except Exception:
            ax.text(0.5, 0.5, "Não foi possível calcular ROC multiclass",
                    horizontalalignment='center', verticalalignment='center')
    else:
        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")

    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig

# ---------------------------
# Funções de visualização para REGRESSÃO
# ---------------------------
def plot_regression_scatter(y_true, y_pred, out_path: Optional[str] = None):
    """
    Scatter: previsto x real (linha y=x).
    """
    plt = _try_import_pyplot()
    if plt is None:
        return None
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1, color='gray')
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")
    ax.set_title("Previsto x Real")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig

def plot_residuals(y_true, y_pred, out_path: Optional[str] = None):
    """
    Plot de resíduos (y_true - y_pred) versus previsto.
    """
    plt = _try_import_pyplot()
    if plt is None:
        return None
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    resid = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, resid, alpha=0.6, s=20)
    ax.axhline(0, linestyle='--', color='gray', linewidth=1)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Residuo (real - previsto)")
    ax.set_title("Resíduos vs Previsto")
    fig.tight_layout()
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
        return out_path
    return fig
