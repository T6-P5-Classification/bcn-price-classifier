import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pytest

from src.preprocessing import build_preprocessor


def make_data():
    X_train = pd.DataFrame(
        {
            "accommodates": [2.0, 4.0, np.nan, 6.0],
            "room_type": ["Private room", "Entire home/apt", None, "Entire home/apt"],
        }
    )
    X_test = pd.DataFrame(
        {
            "accommodates": [np.nan, 100.0],
            "room_type": ["Hotel room", "Private room"],  # "Hotel room" unseen
        }
    )
    return X_train, X_test


def test_returns_unfitted_preprocessor():
    X_train, _ = make_data()
    with pytest.raises(NotFittedError):
        build_preprocessor().transform(X_train)


def test_imputes_nulls_and_accepts_unknown_categories():
    X_train, X_test = make_data()
    prep = build_preprocessor().set_params(cat__encoder__min_frequency=None)
    prep.fit(X_train)
    out = prep.transform(X_test)
    assert not np.isnan(out).any()
    assert out.shape[0] == len(X_test)


def test_statistics_learned_only_from_train():
    X_train, X_test = make_data()
    prep = build_preprocessor().fit(X_train)
    median_before = prep.named_transformers_["num"]["imputer"].statistics_.copy()
    prep.transform(X_test)
    assert median_before[0] == X_train["accommodates"].median()
    np.testing.assert_array_equal(
        prep.named_transformers_["num"]["imputer"].statistics_, median_before
    )


def test_works_inside_pipeline():
    X_train, X_test = make_data()
    y_train = [0, 1, 0, 1]
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LogisticRegression())])
    pipe.fit(X_train, y_train)
    assert len(pipe.predict(X_test)) == len(X_test)
