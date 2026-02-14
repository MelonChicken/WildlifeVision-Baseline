from dataclasses import dataclass

@dataclass(frozen=True)
class LogRegConfig:
    C: float = 1.0
    max_iter: int = 2000
    use_scaler: bool = True

    solver: str = "lbfgs"
    random_state: int = 42
    class_weight="balanced"
