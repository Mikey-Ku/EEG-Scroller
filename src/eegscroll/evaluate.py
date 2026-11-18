"""Honest testing: hold out one whole session at a time."""

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, permutation_test_score


def evaluate_models(
    models: dict[str, BaseEstimator],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    """Balanced accuracy per model and per held-out session."""
    rows = []
    for name, model in models.items():
        # One fold per session: train others, test one
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            # Fresh copy, fit on training sessions only
            fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
            pred = fitted.predict(X.iloc[test_idx])

            # Held-out session may contain only one class
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = balanced_accuracy_score(y[test_idx], pred)
            rows.append({"model": name, "held_out_session": groups[test_idx][0], "balanced_accuracy": score})
    return pd.DataFrame(rows)


def permutation_p_value(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_permutations: int = 100,
) -> float:
    """How likely is this score with shuffled labels?"""
    # Labels shuffled within each session, many times
    _score, _perm_scores, p_value = permutation_test_score(
        model,
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
        scoring="balanced_accuracy",
        n_permutations=n_permutations,
        random_state=0,
        n_jobs=-1,
    )
    return float(p_value)


def print_feature_importance(
    models: dict[str, BaseEstimator], X: pd.DataFrame, y: np.ndarray, top: int = 10
) -> None:
    """Which features did each model rely on?"""
    # Logistic regression: signed coefficients, biggest first
    if "logistic_regression" in models:
        lr = clone(models["logistic_regression"]).fit(X, y)
        coef = pd.Series(lr.named_steps["clf"].coef_[0], index=X.columns)
        print(f"\nLogistic regression coefficients (+ means skip), top {top}:")
        print(coef.reindex(coef.abs().sort_values(ascending=False).index).head(top).round(3).to_string())

    # Random forest: impurity importances, biggest first
    if "random_forest" in models:
        rf = clone(models["random_forest"]).fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"\nRandom forest importances, top {top}:")
        print(imp.head(top).round(3).to_string())
