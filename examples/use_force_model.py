# examples/use_force_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from datakhanon.model.auto_trainer import AutoTrainer

X, y = load_wine(return_X_y=True)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Forçar estimador customizado (qualquer objeto sklearn)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
trainer = AutoTrainer(model=rf, output_dir="outputs/wine_rf", export_on_finish=False)  # não exporta
res = trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)

print("Modelo treinado:", type(res["model"]).__name__)
print("Métricas:", res["metrics"])
