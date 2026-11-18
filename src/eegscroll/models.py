"""The three models we compare."""

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



def get_models() -> dict[str, BaseEstimator]:
    """Return {name: unfitted model}."""
    return {
        # Baseline: always guess the most common label
        "baseline": DummyClassifier(strategy="most_frequent"),
        # Linear model; scaler fit inside each fold
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)),
            ]
        ),
        # Nonlinear model, many shallow-ish trees
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=0, n_jobs=-1
        ),
    }
