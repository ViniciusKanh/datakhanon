# examples/test_preprocess_pipeline.py
"""
Script de integração para testar o pacote datakhanon.preprocess
Usa: examples/credit_dataset_2000.csv
"""

import pandas as pd
from datakhanon.preprocess.cleaning import standardize_column_names, drop_missing_threshold, convert_dtypes
from datakhanon.preprocess.imputers import ColumnImputer
from datakhanon.preprocess.encoders import OneHotEncoderWrapper, TargetEncoderWrapper
from datakhanon.preprocess.features import FeatureEngineer, build_simple_column_transformer
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "credit_dataset_2000.csv")

def main():
    print("Carregando dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    print("Linhas:", len(df), "Colunas:", len(df.columns))

    # 1) Limpeza de colunas
    df = standardize_column_names(df)
    print("Colunas após standardize:", df.columns.tolist()[:10])

    # 2) Remover colunas com muitos NA (nenhuma neste CSV sintético, mas como teste)
    df = drop_missing_threshold(df, threshold=0.95)
    print("Após drop_missing_threshold:", df.shape)

    # 3) Conversão de tipos (exemplo)
    df = convert_dtypes(df)

    # 4) Imputação (num + cat)
    imp = ColumnImputer(num_strategy="median", cat_strategy="most_frequent")
    df_imp = imp.fit_transform(df)
    print("NAs por coluna após imputação (deve ser 0 para colunas num/cat detectadas):")
    print(df_imp.isna().sum().loc[lambda s: s > 0])

    # 5) Codificação: One-hot para 'purpose' e 'housing' (exemplo)
    categorical_cols = ["purpose", "housing"]
    ohe = OneHotEncoderWrapper(columns=categorical_cols, drop="first", sparse=False)
    ohe_df = ohe.fit_transform(df_imp)
    print("One-hot transform result shape:", ohe_df.shape)

    # 6) Feature engineering para variáveis numéricas
    # Escolher colunas numéricas para o exemplo:
    numeric_cols = df_imp.select_dtypes(include=["number"]).columns.tolist()
    fe = FeatureEngineer(scaler="standard", variance_threshold=0.0, select_k=20, problem_type="classification")
    # precisa de y: usar loan_status como target binário
    y = (df_imp["loan_status"].astype(str) == "Default").astype(int)
    df_num_transformed = fe.fit_transform(df_imp[numeric_cols], y)
    print("Num cols originais:", len(numeric_cols), "-> após seleção:", df_num_transformed.shape[1])

    # 7) Salvar artefatos para reuso
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(imp, "artifacts/column_imputer.joblib")
    joblib.dump(ohe, "artifacts/onehot.joblib")
    joblib.dump(fe, "artifacts/feature_engineer.joblib")
    print("Artefatos salvos em artifacts/")

    # 8) Montar DataFrame final (exemplo simple concat)
    df_final = pd.concat([df_num_transformed.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)
    print("DataFrame final pronto para modelagem:", df_final.shape)

if __name__ == "__main__":
    main()
