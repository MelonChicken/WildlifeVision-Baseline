from data.make_table import *
from src.config.model_config import LogRegConfig
from src.data.inspect_images import *
from src.features.hog import *
from src.models.logreg import build_logreg_pipeline
from src.training.evaluate import *
from src.training.crossvalidation import *
from src.data.image_process import *
from src.training.experiment_logger import log_experiment
from src.training.experiment_schema import DataSignature, build_record, LogRegParams, HogParams, CVChecks

# 1. make single train table
df_train_table = make_train_table(autosave=True)
site_counts = df_train_table["site"].value_counts()

# print(site_counts.describe())
# print("top 10 sites:\n", site_counts.head(10))
# print(df_train_table['site'].nunique())

# 2. add 5 folds to the train table
df_train_table_w_folds = add_group_folds(df_train_table, shuffle=True, random_state=42, autosave=True)

# 3. check if the fold information added without any errors
check_site_has_single_fold(df_train_table_w_folds, site_col="site", fold_col="fold")
check_no_site_overlap_between_train_valid(df_train_table_w_folds, site_col="site", fold_col="fold")
fold_site_counts = {int(k): int(v) for k, v in df_train_table_w_folds.groupby("fold")["site"].nunique().to_dict().items()}


# 4. visualize image distribution
# df, summary = collect_image_sizes(df_train_table_w_folds['filepath'])
# plot_image_size_distributions(df_sizes=df, output_dir=ARTIFACTS_DIR / 'eda')

# 5. preprocess image & extract hog feature
# 6. build a fixed-length feature matrix (X) for scikit-learn image classification.
X, y, fold = load_or_build_train_hog_cache(
    df=df_train_table_w_folds,
    project_dir=PROJECT_ROOT,
    artifacts_dir=ARTIFACTS_DIR,
    prefix="train_hog"
)
assert fold is not None, "fold가 None입니다. add_group_folds 결과를 확인하세요."
site = df_train_table_w_folds["site"].to_numpy() if "site" in df_train_table_w_folds.columns else None

grid = [
    {"C": 0.02, "use_scaler": False, "class_weight": None},
    {"C": 0.05, "use_scaler": False, "class_weight": None},
    {"C": 0.1,  "use_scaler": False, "class_weight": None},
    {"C": 0.2,  "use_scaler": False, "class_weight": None},
    {"C": 0.5,  "use_scaler": False, "class_weight": None},

    {"C": 0.02, "use_scaler": False, "class_weight": "balanced"},
    {"C": 0.05, "use_scaler": False, "class_weight": "balanced"},
    {"C": 0.1,  "use_scaler": False, "class_weight": "balanced"},
    {"C": 0.2,  "use_scaler": False, "class_weight": "balanced"},
    {"C": 0.5,  "use_scaler": False, "class_weight": "balanced"},
]

class_counts = {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
data_sig = DataSignature(
    n_samples=int(len(y)),
    n_sites=int(df_train_table_w_folds["site"].nunique()),
    class_counts=class_counts,
)

for cfg_dict in grid:
    # 1) Config 객체 생성
    cfg = LogRegConfig(
        C=float(cfg_dict["C"]),
        max_iter=2000,
        use_scaler=bool(cfg_dict["use_scaler"]),
        solver="lbfgs",
        random_state=42,
        class_weight=cfg_dict["class_weight"],
    )

    # 2) CV 평가
    mean_log_loss, standard_log_loss, scores = evaluate_by_fold(
        X=X,
        y=y,
        fold=fold,
        build_model_fn=lambda cfg=cfg: build_logreg_pipeline(cfg)
    )

    # 3) 로그 레코드 (class_weight 포함)
    record = build_record(
        tag="logreg_hog_grid_v2",  # v2로 올리는 걸 권장(스키마/그리드 변경)
        feature_name="hog",
        model_name="logreg",
        cv_type="GroupKFold(site)",
        n_splits=5,
        logreg_params=LogRegParams(
            C=float(cfg.C),
            max_iter=int(cfg.max_iter),
            use_scaler=bool(cfg.use_scaler),
            class_weight=cfg.class_weight,   # 아래 LogRegParams에 필드 추가 필요
        ),
        hog_params=HogParams(
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            orientations=9,
            block_norm="L2-Hys",
        ),
        mean_log_loss=mean_log_loss,
        std_log_loss=standard_log_loss,
        fold_log_loss=scores,
        data_sig=data_sig,
        cv_checks=CVChecks(
            site_single_fold=True,
            no_site_overlap=True,
            random_state=42,
        ),
        fold_site_counts=fold_site_counts,
    )

    log_experiment(LOG_PATH, record.to_dict())
    print(f"[log] appended -> {LOG_PATH} | C={cfg.C}, scaler={cfg.use_scaler}, class_weight={cfg.class_weight}")