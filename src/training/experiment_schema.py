from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import platform
import uuid

import sklearn


@dataclass(frozen=True)
class EnvInfo:
    python: str
    sklearn: str

    @staticmethod
    def current() -> "EnvInfo":
        return EnvInfo(
            python=platform.python_version(),
            sklearn=sklearn.__version__,
        )


@dataclass(frozen=True)
class CVInfo:
    type: str
    n_splits: int


@dataclass(frozen=True)
class LogRegParams:
    C: float
    max_iter: int
    use_scaler: bool


@dataclass(frozen=True)
class HogParams:
    pixels_per_cell: tuple[int, int]
    cells_per_block: tuple[int, int]
    orientations: int
    block_norm: str


@dataclass(frozen=True)
class Params:
    logreg: LogRegParams
    hog: HogParams


@dataclass(frozen=True)
class Metrics:
    mean_log_loss: float
    std_log_loss: float
    fold_log_loss: list[float]


@dataclass(frozen=True)
class DataSignature:
    n_samples: int
    n_sites: int
    class_counts: dict[str, int]


@dataclass(frozen=True)
class CVChecks:
    site_single_fold: bool
    no_site_overlap: bool
    random_state: int


@dataclass(frozen=True)
class ExperimentRecord:
    # meta
    run_id: str
    timestamp: str
    env: EnvInfo

    # core
    tag: str
    feature_name: str
    model_name: str
    cv: CVInfo
    params: Params
    metrics: Metrics
    data_signature: DataSignature

    # checks
    cv_checks: CVChecks
    fold_site_counts: dict[int, int]

    def to_dict(self) -> dict:
        d = asdict(self)

        # JSON 친화: tuple -> list (원하시는 로그 형태와 동일)
        hog = d["params"]["hog"]
        hog["pixels_per_cell"] = list(hog["pixels_per_cell"])
        hog["cells_per_block"] = list(hog["cells_per_block"])

        # fold_site_counts도 JSON에서는 int key가 str로 저장될 수 있으니(로거 구현에 따라)
        # 여기서 강제로 int->int 유지하고 싶으면 logger 쪽에서 처리하는 게 더 안전.
        # 일단 dict로 그대로 둠.
        return d


def new_run_meta() -> tuple[str, str, EnvInfo]:
    run_id = uuid.uuid4().hex[:10]
    ts = datetime.now(timezone.utc).isoformat()
    env = EnvInfo.current()
    return run_id, ts, env


def build_record(
    *,
    tag: str,
    feature_name: str,
    model_name: str,
    cv_type: str,
    n_splits: int,
    logreg_params: LogRegParams,
    hog_params: HogParams,
    mean_log_loss: float,
    std_log_loss: float,
    fold_log_loss: list[float],
    data_sig: DataSignature,
    cv_checks: CVChecks,
    fold_site_counts: dict[int, int],
) -> ExperimentRecord:
    run_id, ts, env = new_run_meta()
    return ExperimentRecord(
        run_id=run_id,
        timestamp=ts,
        env=env,
        tag=tag,
        feature_name=feature_name,
        model_name=model_name,
        cv=CVInfo(type=cv_type, n_splits=n_splits),
        params=Params(logreg=logreg_params, hog=hog_params),
        metrics=Metrics(
            mean_log_loss=float(mean_log_loss),
            std_log_loss=float(std_log_loss),
            fold_log_loss=[float(x) for x in fold_log_loss],
        ),
        data_signature=data_sig,
        cv_checks=cv_checks,
        fold_site_counts={int(k): int(v) for k, v in fold_site_counts.items()},
    )