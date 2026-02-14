from data.make_table import *
from src.data.inspect_images import *
from src.features.hog import *
from src.models.logreg import build_logreg_pipeline
from src.training.evaluate import *
from src.training.crossvalidation import *
from src.data.image_process import *

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


# train model and validate

mean_log_loss, standard_log_loss, scores = evaluate_by_fold(
    X=X,
    y=y,
    fold=fold,
    build_model_fn=lambda: build_logreg_pipeline(C=1.0, max_iter=2000, use_scaler=True)
)
