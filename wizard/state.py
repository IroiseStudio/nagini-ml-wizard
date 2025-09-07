from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import pandas as pd

@dataclass
class WizardState:
    """Central state passed across steps."""
    # Raw & cleaned data
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None

    # Schema choices
    target: Optional[str] = None
    features: List[str] = field(default_factory=list)
    task_type: Optional[str] = None  # "classification" | "regression"

    # Preprocessing settings
    num_missing: str = "mean"        # "none"|"mean"|"median"
    cat_missing: str = "mode"        # "none"|"mode"
    drop_any_na: bool = False
    scale_numeric: bool = False
    encode_categorical: bool = True

    # Train/valid split
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True

    # Fitted artifacts
    pipeline: Any = None             # sklearn Pipeline
    metrics: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None

    data_version: int = 0  # bump this on each successful data load

    # Trained model/pipeline and optional class names
    model: Optional[Any] = None                # fitted sklearn Pipeline/Estimator
    class_names: Optional[List[str]] = None    # nicer labels for classifiers

    def is_ready_for_training(self) -> bool:
        return (
            self.df_clean is not None
            and self.target is not None
            and len(self.features) > 0
            and self.task_type in {"classification", "regression"}
        )
