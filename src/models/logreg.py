# src/models/logreg_pipeline.py
from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.config.model_config import LogRegConfig


def build_logreg_pipeline(cfg: LogRegConfig) -> Pipeline:
    """
    Logistic Regression pipeline builder (scaler optional).
    """
    steps = []
    if cfg.use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(
        ("model", LogisticRegression(
            solver=cfg.solver,
            C=cfg.C,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            class_weight=cfg.class_weight,
        ))
    )
    return Pipeline(steps)
