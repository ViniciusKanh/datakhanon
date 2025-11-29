# examples/use_binary_auto.py
# Demonstração com dataset sintético binário e export ONNX + joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from datakhanon.model.auto_trainer import AutoTrainer

# Dataset binário sintético
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, weights=[0.7,0.3], random_state=0)

# Treino com validação explícita e export para ONNX (requer skl2onnx instalado)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

trainer = AutoTrainer(model="auto", output_dir="outputs/binary_auto", export_on_finish=True, export_formats=["joblib","onnx"], cv=3)
res = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Impressão resumida
print("Modelo:", res["model_name"])
print("Métricas (ex):", res["metrics"])
print("Exports:", res["exports"])  # contém paths ou erro se skl2onnx ausente
