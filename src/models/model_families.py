from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config.model_config import LogRegConfig
from src.models.logreg import build_logreg_pipeline

MODEL_FAMILIES = ("family_c", "family_b") #"family_a",

CFG_A: dict[str, object] = {
    "C": 0.003,
    "max_iter": 6000,
    "use_scaler": False,
    "solver": "lbfgs",
    "random_state": 42,
    "class_weight": "balanced",
}

CFG_B: dict[str, object] = {
    "C": 1.0,
    "class_weight": "balanced",
    "random_state": 42,
    "calib_method": "sigmoid",
    "calib_cv": 2,          # 1 말고 2
    "max_iter": 8000,       # 20000 -> 8000 (tol과 같이)
    "tol": 1e-3,
}

CFG_C: dict[str, object] = {
    "learning_rate": 0.1,   # 0.05 -> 0.1
    "max_depth": 3,         # 6 -> 3
    "max_iter": 80,         # 300 -> 80
    "random_state": 42,
    "early_stopping": False,
}


def build_family_a_pipeline(cfg_a: dict[str, object]) -> Pipeline:
    return build_logreg_pipeline(
        LogRegConfig(
            C=float(cfg_a["C"]),
            max_iter=int(cfg_a["max_iter"]),
            use_scaler=bool(cfg_a["use_scaler"]),
            solver=str(cfg_a.get("solver", "lbfgs")),
            random_state=int(cfg_a.get("random_state", 42)),
            class_weight=(
                None if cfg_a.get("class_weight") is None else str(cfg_a.get("class_weight"))
            ),
        )
    )


def build_family_b_pipeline(cfg_b: dict[str, object]) -> Pipeline:
    base = LinearSVC(
        C=float(cfg_b["C"]),
        class_weight=None if cfg_b.get("class_weight") is None else str(cfg_b.get("class_weight")),
        random_state=int(cfg_b.get("random_state", 42)),
        max_iter=int(cfg_b.get("max_iter", 8000)),
        tol=float(cfg_b.get("tol", 1e-3)),
    )
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method=str(cfg_b.get("calib_method", "sigmoid")),
        cv=int(cfg_b.get("calib_cv", 3)),
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", calibrated),
        ]
    )


def build_family_c_pipeline(cfg_c: dict[str, object]) -> Pipeline:
    model = HistGradientBoostingClassifier(
        learning_rate=float(cfg_c.get("learning_rate", 0.05)),
        max_depth=int(cfg_c.get("max_depth", 6)),
        max_iter=int(cfg_c.get("max_iter", 300)),
        random_state=int(cfg_c.get("random_state", 42)),
        early_stopping=bool(cfg_c.get("early_stopping", False)),
    )
    return Pipeline(steps=[("model", model)])


def build_model_pipeline(model_family: str) -> Pipeline:
    if model_family == "family_a":
        return build_family_a_pipeline(CFG_A)
    if model_family == "family_b":
        return build_family_b_pipeline(CFG_B)
    if model_family == "family_c":
        return build_family_c_pipeline(CFG_C)
    raise ValueError(f"Unknown model_family={model_family}")


def build_model_pipeline_from_params(
    model_family: str,
    model_params: dict[str, object],
) -> Pipeline:
    if model_family == "family_a":
        return build_family_a_pipeline(model_params)
    if model_family == "family_b":
        return build_family_b_pipeline(model_params)
    if model_family == "family_c":
        return build_family_c_pipeline(model_params)
    raise ValueError(f"Unknown model_family={model_family}")


def get_model_params(model_family: str) -> dict[str, object]:
    if model_family == "family_a":
        return dict(CFG_A)
    if model_family == "family_b":
        return dict(CFG_B)
    if model_family == "family_c":
        return dict(CFG_C)
    raise ValueError(f"Unknown model_family={model_family}")
