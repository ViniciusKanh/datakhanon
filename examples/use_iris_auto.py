# examples/use_iris_auto.py
from sklearn.datasets import load_iris
from datakhanon.model.auto_trainer import AutoTrainer

X, y = load_iris(return_X_y=True)
trainer = AutoTrainer(model="auto", output_dir="outputs/iris_auto", export_on_finish=True, export_formats=["joblib"])
res = trainer.fit(X, y)
print("Modelo escolhido:", res["model_name"])
print("Métricas:", res["metrics"])
print("Relatório gerado em:", res["report"]["report_md"])
print("Modelo salvo em:", res["model_saved_at"])
