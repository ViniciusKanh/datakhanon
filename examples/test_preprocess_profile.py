# exemplo de uso rápido (salve em examples/run_report.py)
import os
import pandas as pd
from datakhanon.preprocess.reporter import profile_and_open

df = pd.read_csv(os.path.join("examples", "credit_dataset_2000.csv"))
y = (df["loan_status"] == "Default").astype(int)  # opcional
res = profile_and_open(df, target=y, output_dir="artifacts/report")
print("Relatório gerado em:", res["html_report"])
