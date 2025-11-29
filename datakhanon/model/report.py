# datakhanon/model/report.py
from pathlib import Path
import json
from datetime import datetime
import numpy as np

from datakhanon.model import viz
from datakhanon.model.selector import detect_task_type
from datakhanon.model.metrics import auto_metrics, regression_metrics, classification_metrics

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)

def generate_report(model, X_train, y_train, X_val, y_val, output_dir: str, model_name: str = "model"):
    """
    Gera relatório inteligente:
     - Detecta tarefa (classification/regression)
     - Para classification: classification_report, confusion_matrix, ROC (se disponível)
     - Para regression: regressions metrics + scatter e residuals plots
    Retorna paths dos artefatos (metrics.json, report.md, imagens quando geradas).
    """
    out = Path(output_dir) / _safe_name(model_name)
    out.mkdir(parents=True, exist_ok=True)

    # predições
    y_pred = model.predict(X_val)
    y_score = None
    try:
        prob = model.predict_proba(X_val)
        # se retornou prob para multiclass, mantém; para binário extrai coluna 1
        if getattr(prob, "ndim", 1) == 2 and prob.shape[1] == 2:
            y_score = prob[:, 1]
        else:
            y_score = prob
    except Exception:
        y_score = None

    # detecta tarefa usando o conjunto de validação (y_val)
    task = detect_task_type(y_val)

    metrics_dict = {}
    cm_path = out / "confusion_matrix.png"
    roc_path = out / "roc.png"
    scatter_path = out / "pred_vs_true.png"
    resid_path = out / "residuals.png"

    if task in ("binary", "multiclass"):
        # classificação
        try:
            from sklearn.metrics import classification_report, accuracy_score
            report_text = classification_report(y_val, y_pred, output_dict=False)
            report_obj = classification_report(y_val, y_pred, output_dict=True)
        except Exception:
            report_text = ""
            report_obj = {}
        try:
            acc = float(accuracy_score(y_val, y_pred))
        except Exception:
            acc = None

        metrics_dict = {
            "accuracy": acc,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "classification_report": report_obj
        }
        (out / "metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

        # plots: confusion matrix + roc (se disponível)
        cm_res = viz.plot_confusion_matrix(y_val, y_pred, out_path=str(cm_path))
        roc_res = None
        if y_score is not None:
            try:
                roc_res = viz.plot_roc_curve(y_val, np.array(y_score), out_path=str(roc_path))
            except Exception:
                roc_res = None

        md = []
        md.append(f"# Relatório — {model_name}")
        md.append(f"- Data(UTC): {metrics_dict['timestamp']}")
        md.append(f"- Acurácia: {acc if acc is not None else 'N/A'}")
        md.append("")
        md.append("## Classification Report")
        md.append("```")
        md.append(report_text)
        md.append("```")
        md.append("")
        md.append("## Artefatos")
        md.append(f"- metrics.json")
        md.append(f"- confusion_matrix.png (gerada: {bool(cm_res)})")
        md.append(f"- roc.png (gerada: {bool(roc_res)})")

        (out / "report.md").write_text("\n".join(md), encoding="utf-8")
        return {"metrics": str(out / "metrics.json"),
                "confusion": str(cm_path) if cm_res else None,
                "roc": str(roc_path) if roc_res else None,
                "report_md": str(out / "report.md")}

    else:
        # regressão
        try:
            metrics_reg = regression_metrics(y_val, y_pred)
        except Exception:
            metrics_reg = {}
        metrics_dict = {"timestamp": datetime.utcnow().isoformat() + "Z", **metrics_reg}
        (out / "metrics.json").write_text(json.dumps(metrics_dict, indent=2), encoding="utf-8")

        # plots: previsto x real e resíduos
        scatter_res = viz.plot_regression_scatter(y_val, y_pred, out_path=str(scatter_path))
        resid_res = viz.plot_residuals(y_val, y_pred, out_path=str(resid_path))

        md = []
        md.append(f"# Relatório — {model_name} (Regressão)")
        md.append(f"- Data(UTC): {metrics_dict['timestamp']}")
        md.append("")
        md.append("## Métricas de Regressão")
        for k, v in metrics_reg.items():
            md.append(f"- **{k}**: {v}")
        md.append("")
        md.append("## Artefatos")
        md.append(f"- metrics.json")
        md.append(f"- pred_vs_true.png (gerada: {bool(scatter_res)})")
        md.append(f"- residuals.png (gerada: {bool(resid_res)})")

        (out / "report.md").write_text("\n".join(md), encoding="utf-8")
        return {"metrics": str(out / "metrics.json"),
                "pred_vs_true": str(scatter_path) if scatter_res else None,
                "residuals": str(resid_path) if resid_res else None,
                "report_md": str(out / "report.md")}
