# datakhanon/model/exporter.py
from typing import List, Optional
import joblib, os
from pathlib import Path
import importlib

def available_exporters() -> List[str]:
    exporters = ["joblib"]
    try:
        importlib.import_module("skl2onnx")
        exporters.append("onnx")
    except Exception:
        pass
    return exporters

def export_joblib(model, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(p))
    return str(p)

def export_onnx(sklearn_model, X_sample, path: str, target_opset:int=13):
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        raise ImportError("Instale 'skl2onnx' para export ONNX: " + str(e))
    initial_type = [("float_input", FloatTensorType([None, X_sample.shape[1]]))]
    onx = convert_sklearn(sklearn_model, initial_types=initial_type, target_opset=target_opset)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return path

def export(model, output_dir:str, base_name:str="model", formats:Optional[List[str]]=None, X_sample=None):
    if formats is None:
        formats = ["joblib"]
    out = {}
    for fmt in formats:
        if fmt == "joblib":
            p = os.path.join(output_dir, f"{base_name}.joblib")
            out["joblib"] = export_joblib(model, p)
        elif fmt == "onnx":
            if X_sample is None:
                raise ValueError("X_sample obrigatório para export_onnx.")
            p = os.path.join(output_dir, f"{base_name}.onnx")
            out["onnx"] = export_onnx(model, X_sample, p)
        else:
            raise ValueError(f"Formato desconhecido: {fmt}")
    return out
