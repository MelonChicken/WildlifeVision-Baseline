from typing import Callable
import numpy as np
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline

from src.config.global_variables import CLASS_COLS


def evaluate_by_fold(
        X : np.ndarray,
        y : np.ndarray,
        fold : np.ndarray,
        build_model_fn: Callable[[], Pipeline]
):
    """
    Evaluate (validate) model by precomputed fold using log loss.
    :param X:
    :param y:
    :param fold:
    :param build_model_fn: callable -> sklearn estimator (Pipeline)
    """
    scores = []
    for current_fold in np.unique(fold):
        # samples to be used in training : rows whose fold is not current fold
        train_mask = fold != current_fold
        # samples to be used in validating : rows whose fold is current fold
        validate_mask = fold == current_fold

        # instantiate classifier
        classifier = build_model_fn()
        # fit by data
        classifier.fit(X[train_mask], y[train_mask])

        # predict based on validation data
        probabilities = classifier.predict_proba(X[validate_mask])

        # sanity check whether the prediction probability is not in the wrong probability distribution
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

        # ensure label order matches probability columns
        assert "model" in classifier.named_steps
        classes = classifier.named_steps["model"].classes_

        # check if there is unknown class name in validation data
        missing = [c for c in CLASS_COLS if c not in classes]
        assert len(missing) == 0, f"Fold {current_fold}: train에 없는 클래스가 있습니다: {missing}"

        col_idx = np.array([np.where(classes == c)[0][0] for c in CLASS_COLS])
        probabilities = probabilities[:, col_idx]

        labels = np.array(CLASS_COLS, dtype=object)
        calculated_log_loss = log_loss(y[validate_mask], probabilities, labels=labels)

        scores.append(calculated_log_loss)

        print(f"[fold={current_fold}] calculated_log_loss={calculated_log_loss:.6f}")

    mean_calculated_log_loss = float(np.mean(scores))
    standard_calculated_log_loss = float(np.std(scores))

    print(f"CV mean log_loss={mean_calculated_log_loss:.6f} (+/- {standard_calculated_log_loss:.6f})")

    return mean_calculated_log_loss, standard_calculated_log_loss, scores