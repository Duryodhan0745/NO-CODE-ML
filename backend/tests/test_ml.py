"""Tests for ml.py: registry, factory, preprocessing, evaluation."""

import pytest
import numpy as np
import pandas as pd
from ml import (
    MODEL_REGISTRY, create_model, list_models, get_model_info,
    build_preprocessing_pipeline, compute_metrics, IQRClipper,
)


# ──────────────────────── Registry ────────────────────────

def test_registry_has_eighteen_models():
    assert len(MODEL_REGISTRY) == 18


def test_registry_eight_classifiers():
    clf = [m for m in MODEL_REGISTRY.values() if m["problem_type"] == "classification"]
    assert len(clf) == 8


def test_registry_ten_regressors():
    reg = [m for m in MODEL_REGISTRY.values() if m["problem_type"] == "regression"]
    assert len(reg) == 10


def test_every_model_has_optuna_space():
    for mid, entry in MODEL_REGISTRY.items():
        assert "optuna_space" in entry, f"{mid} missing optuna_space"
        assert len(entry["optuna_space"]) > 0, f"{mid} optuna_space is empty"


def test_every_model_has_required_keys():
    required = {"model_id", "name", "problem_type", "description", "cls", "needs_scaling", "parameters"}
    for mid, entry in MODEL_REGISTRY.items():
        assert required.issubset(entry.keys()), f"{mid} missing keys"


def test_list_models_filters_by_problem_type():
    clf = list_models("classification")
    assert all(m["problem_type"] == "classification" for m in clf)
    reg = list_models("regression")
    assert all(m["problem_type"] == "regression" for m in reg)


def test_get_model_info_excludes_cls():
    info = get_model_info("random_forest_classifier")
    assert "cls" not in info
    assert info["model_id"] == "random_forest_classifier"


# ──────────────────────── Factory ────────────────────────

def test_create_all_models_with_defaults():
    for mid in MODEL_REGISTRY:
        model = create_model(mid, {})
        assert model is not None


def test_create_random_forest_with_custom_params():
    model = create_model("random_forest_classifier", {"n_estimators": 50, "max_depth": 5})
    assert model.n_estimators == 50
    assert model.max_depth == 5


def test_reject_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("nonexistent_model", {})


def test_reject_unknown_parameter():
    with pytest.raises(ValueError, match="Unknown parameter"):
        create_model("logistic_regression", {"invalid_param": 42})


def test_reject_negative_n_estimators():
    with pytest.raises(ValueError, match="must be >="):
        create_model("random_forest_classifier", {"n_estimators": -1})


def test_reject_n_estimators_too_large():
    with pytest.raises(ValueError, match="must be <="):
        create_model("random_forest_classifier", {"n_estimators": 9999})


def test_nullable_max_depth():
    model = create_model("decision_tree_classifier", {"max_depth": None})
    assert model.max_depth is None


def test_bool_param_string_coercion():
    model = create_model("linear_regression", {"fit_intercept": "false"})
    assert model.fit_intercept is False


# ──────────────────────── Preprocessing ────────────────────────

def test_build_pipeline_numerical_only():
    pipe = build_preprocessing_pipeline(["a", "b"], [], {"missing_values": "mean"})
    assert pipe is not None
    # Fit-transform some data
    X = pd.DataFrame({"a": [1, 2, np.nan], "b": [4, np.nan, 6]})
    out = pipe.fit_transform(X)
    assert not np.isnan(out).any()


def test_build_pipeline_categorical_only():
    pipe = build_preprocessing_pipeline([], ["c"], {"missing_values": "mean"})
    assert pipe is not None
    X = pd.DataFrame({"c": ["a", "b", "a"]})
    out = pipe.fit_transform(X)
    assert out.shape[0] == 3


def test_build_pipeline_mixed():
    pipe = build_preprocessing_pipeline(["n"], ["c"], {"missing_values": "median", "scaling": "standard"})
    X = pd.DataFrame({"n": [1.0, 2.0, 3.0], "c": ["x", "y", "x"]})
    out = pipe.fit_transform(X)
    assert out.shape[0] == 3


def test_build_pipeline_none_when_empty():
    pipe = build_preprocessing_pipeline([], [], {"missing_values": "mean"})
    assert pipe is None


def test_build_pipeline_with_outlier_handling():
    pipe = build_preprocessing_pipeline(["a"], [], {"missing_values": "mean", "outlier_handling": True})
    X = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    out = pipe.fit_transform(X)
    assert out.max() < 100  # Outlier should be clipped


def test_preprocessing_no_leakage():
    """Fit on train, transform test — test must not see train stats."""
    pipe = build_preprocessing_pipeline(["x"], [], {"missing_values": "mean", "scaling": "standard"})
    X_train = pd.DataFrame({"x": [10, 20, 30]})
    X_test = pd.DataFrame({"x": [100]})  # very different from train
    pipe.fit(X_train)
    out_train = pipe.transform(X_train)
    out_test = pipe.transform(X_test)
    # Test value should be far from 0 (since it's 100 scaled by train mean~20, std~10)
    assert abs(out_test[0][0]) > 2


# ──────────────────────── IQR Clipper ────────────────────────

def test_iqr_clipper():
    clipper = IQRClipper()
    X = np.array([[1], [2], [3], [4], [5], [100]])
    clipper.fit(X)
    out = clipper.transform(X)
    assert out[-1][0] < 100  # Outlier clipped


# ──────────────────────── Evaluation ────────────────────────

def test_classification_metrics():
    y_true = [0, 1, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1]
    m = compute_metrics(y_true, y_pred, "classification")
    assert "accuracy" in m
    assert "precision" in m
    assert "recall" in m
    assert "f1" in m
    assert "confusion_matrix" in m
    assert "class_labels" in m
    assert 0 <= m["accuracy"] <= 1


def test_regression_metrics():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.1, 2.2, 2.8]
    m = compute_metrics(y_true, y_pred, "regression")
    assert "mae" in m
    assert "mse" in m
    assert "rmse" in m
    assert "r2" in m
    assert m["mae"] >= 0
    assert m["rmse"] >= 0


def test_wrong_problem_type_raises():
    with pytest.raises(ValueError, match="Unknown problem type"):
        compute_metrics([1], [1], "clustering")


def test_classification_metrics_not_used_for_regression():
    m = compute_metrics([1.0, 2.0], [1.1, 2.1], "regression")
    assert "accuracy" not in m


def test_regression_metrics_not_used_for_classification():
    m = compute_metrics([0, 1], [0, 1], "classification")
    assert "mae" not in m


def test_build_pipeline_with_pca():
    num_cols = ["age", "income", "score"]
    df = pd.DataFrame({"age": [20, 30, 40], "income": [30000, 40000, 50000], "score": [80, 90, 85]})
    pipe = build_preprocessing_pipeline(num_cols, [], {"pca": True, "pca_components": 2})
    assert pipe is not None
    res = pipe.fit_transform(df)
    assert res.shape == (3, 2)


def test_compute_metrics_with_roc_auc():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])
    m = compute_metrics(y_true, y_pred, "classification", y_proba=y_proba)
    assert "roc_auc" in m
    assert m["roc_auc"] == 1.0

