# datakhanon/model/persistence.py
from pathlib import Path
import joblib
import json
from datakhanon.model.model_spec import ModelSpec
from typing import Tuple, Optional

def save_model(obj, path: str, spec: Optional[ModelSpec] = None):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, str(p / "model.joblib"))
    if spec is not None:
        (p / "meta.json").write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")

def load_model(path: str) -> Tuple[object, Optional[ModelSpec]]:
    p = Path(path)
    obj = joblib.load(str(p / "model.joblib"))
    spec_path = p / "meta.json"
    spec = None
    if spec_path.exists():
        spec = ModelSpec(**json.loads(spec_path.read_text(encoding="utf-8")))
    return obj, spec

def list_models(base_dir: str):
    p = Path(base_dir)
    results = []
    for child in p.iterdir():
        if (child / "model.joblib").exists():
            results.append(str(child))
    return results
