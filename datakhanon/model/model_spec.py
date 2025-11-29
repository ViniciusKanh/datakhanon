# datakhanon/model/model_spec.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any

@dataclass
class ModelSpec:
    name: str
    version: str
    framework: str
    created_at: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metrics_summary: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def make(name: str, version: str = "0.0.1", framework: str = "sklearn", **kwargs):
        return ModelSpec(
            name=name,
            version=version,
            framework=framework,
            created_at=datetime.utcnow().isoformat() + "Z",
            input_schema=kwargs.get("input_schema"),
            output_schema=kwargs.get("output_schema"),
            metrics_summary=kwargs.get("metrics_summary"),
            metadata=kwargs.get("metadata"),
        )

    def to_dict(self):
        return asdict(self)
