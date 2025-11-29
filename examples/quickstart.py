#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
examples/quickstart_concise.py
QuickStart conciso Datakhanon:
 - Carrega CSV
 - Preprocess completo com Preprocessor (limpeza, imputação, encoding, FE)
 - Gera EDA (quick_eda)
 - Treina AutoTrainer (model="auto")
 - Salva resumo e modelo
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Datakhanon imports
from datakhanon.preprocess.preprocessor import Preprocessor  # orchestration: cleaning->impute->encode->FE. :contentReference[oaicite:3]{index=3}
from datakhanon.visualize.eda import quick_eda              # EDA + plots generator. :contentReference[oaicite:4]{index=4}
from datakhanon.model.auto_trainer import AutoTrainer        # treinador automático (model="auto")
from datakhanon.model.persistence import load_model

def nowstr():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def detect_target_column(df: pd.DataFrame, user_target: str | None = None) -> str:
    if user_target and user_target in df.columns:
        return user_target
    commons = ["target", "label", "y", "classe", "status", "inadimplente"]
    for c in commons:
        if c in df.columns:
            return c
    return df.columns[-1]

def main():
    p = argparse.ArgumentParser(description="QuickStart conciso Datakhanon")
    p.add_argument("--csv", required=True)
    p.add_argument("--target", required=False)
    p.add_argument("--out", default="outputs/quickstart_concise")
    p.add_argument("--tune", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    target_col = detect_target_column(df, args.target)
    print(f"[INFO] dataset shape: {df.shape}  target: {target_col}")

    # 1) Preprocess completo (compacto e seguro)
    pp = Preprocessor(drop_missing_threshold_prop=0.95)  # defaults sensatos
    X = pp.fit_transform(df, df[target_col])  # pp usa y quando necessário (e salva estado). :contentReference[oaicite:5]{index=5}
    y = df[target_col]
    print(f"[INFO] pós-preprocess X.shape={X.shape} y.shape={y.shape}")

    # 2) EDA (gera imagens + summary JSON)
    eda_dir = out / "eda"
    eda_dir.mkdir(exist_ok=True)
    eda_res = quick_eda(df, output_dir=str(eda_dir), target=y, use_reporter=True)
    print(f"[INFO] EDA gerado: {eda_res}")

    # 3) Treino automático (AutoTrainer)
    trainer = AutoTrainer(
        model="rf",
        output_dir=str(out / "model"),
        export_on_finish=True,
        export_formats=None,   # detecta automaticamente (joblib / onnx se disponível)
        cv=3,
        random_state=42,
        max_candidates=4,
    )
    # AutoTrainer aceita X como DataFrame ou ndarray; passamos numpy para robustez com vários estimadores
    print("[INFO] Iniciando treino automático (mode auto)...")
    res = trainer.fit(X.values, y.values, X_val=None, y_val=None, tune=args.tune)
    print(f"[INFO] Treino finalizado. model_name={res.get('model_name')}, metrics={res.get('metrics')}")

    # 4) Carregar modelo salvo e predizer (exemplo)
    model_dir = Path(res.get("model_saved_at") or (out / "model"))
    if model_dir.exists():
        model_obj, spec = load_model(str(model_dir))
        sample = X.iloc[:5] if hasattr(X, "iloc") else X[:5]
        try:
            preds = model_obj.predict(sample.values)
        except Exception:
            preds = model_obj.predict(sample)
        print(f"[INFO] Predições de exemplo (5 linhas): {preds}")

    # 5) resumo final
    summary = {
        "csv": str(args.csv),
        "target": target_col,
        "eda": str(eda_dir),
        "model_dir": str(model_dir),
        "metrics": res.get("metrics"),
    }
    (out / f"quickstart_summary_{nowstr()}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Quickstart concluído. resumo salvo em {out}")

if __name__ == "__main__":
    main()
