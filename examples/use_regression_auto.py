# examples/use_regression_auto.py
# Demonstração em dataset de regressão (California Housing)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from datakhanon.model.auto_trainer import AutoTrainer

data = fetch_california_housing()
X, y = data.data, data.target

# AutoTrainer detecta 'regression' e usa métricas: mse, rmse, mae, r2
trainer = AutoTrainer(model="auto", output_dir="outputs/california_auto", export_on_finish=True, export_formats=["joblib"])
res = trainer.fit(X, y)  # sem X_val => AutoTrainer fará split interno

print("Modelo:", res["model_name"])
print("Métricas:", res["metrics"])  # contém mse, rmse, mae, r2
print("Relatório:", res["report"]["report_md"])
