from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def build_logreg_pipeline(
        C: float = 1.0,
        max_iter: int = 2000,
        use_scaler: bool = True
):
    """
    Build a logistic regression pipeline that predicts the label of each feature.
    :param C:
    :param max_iter:
    :param use_scaler:
    :return:
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", LogisticRegression(
        solver="lbfgs",
        C=C,
        max_iter=max_iter,
    )))
    return Pipeline(steps)
