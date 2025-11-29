#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datakhanon.model import quick_experiment_from_csv

def main():
    summary = quick_experiment_from_csv(
        csv_path="credit_dataset_2000.csv",
        target_col="loan_status",     # ou None para autodetectar
        out_dir="outputs/credit_exp1",
        model="rf",                   # ou "auto", "xgb", etc.
        tune=False,
    )

    # Acessando as partes mais importantes do resumo
    metrics = summary["training"]["metrics"]
    model_dir = summary["paths"]["model_dir"]
    eda_dir = summary["paths"]["eda_dir"]

    print("\n================= RESUMO FINAL =================")
    print("Métricas de validação:", metrics)
    print("Diretório do modelo  :", model_dir)
    print("Diretório do EDA     :", eda_dir)
    print("Predições de exemplo :", summary["sample_predictions"])

if __name__ == "__main__":
    main()
