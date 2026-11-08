"""Tests for services.py: dataset ops, workspace, training, prediction."""

import pytest
import os
import tempfile
from pathlib import Path
from conftest import FIXTURES_DIR

# Override dirs before importing services
os.environ["UPLOAD_DIR"] = str(Path(tempfile.mkdtemp()) / "uploads")
os.environ["ARTIFACTS_DIR"] = str(Path(tempfile.mkdtemp()) / "artifacts")

import services


def _read_fixture(name: str) -> bytes:
    return (FIXTURES_DIR / name).read_bytes()


# ──────────────────────── Dataset ────────────────────────

def test_upload_valid_csv():
    info = services.upload_dataset("classification.csv", _read_fixture("classification.csv"))
    assert info["rows"] == 20
    assert info["num_columns"] == 4
    assert "target" in info["columns"]
    assert info["duplicate_rows"] == 0


def test_upload_regression_csv():
    info = services.upload_dataset("regression.csv", _read_fixture("regression.csv"))
    assert info["rows"] == 20
    assert "price" in info["columns"]


def test_upload_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        services.upload_dataset("empty.csv", b"")


def test_upload_rejects_non_csv():
    with pytest.raises(ValueError, match="Unsupported"):
        services.upload_dataset("data.xlsx", b"fake data")


def test_upload_rejects_malformed():
    with pytest.raises(ValueError):
        services.upload_dataset("bad.csv", b"\x00\x01\x02binary garbage")


def test_preview():
    info = services.upload_dataset("clf2.csv", _read_fixture("classification.csv"))
    preview = services.get_preview(info["id"], 5)
    assert len(preview["data"]) == 5
    assert preview["total_rows"] == 20


def test_analysis():
    info = services.upload_dataset("clf3.csv", _read_fixture("classification.csv"))
    analysis = services.get_analysis(info["id"])
    assert analysis["rows"] == 20
    assert "numerical_columns" in analysis
    assert len(analysis["column_stats"]) == 4

def test_analysis_has_correlation():
    info = services.upload_dataset("clf_corr.csv", _read_fixture("classification.csv"))
    analysis = services.get_analysis(info["id"])
    assert "correlation" in analysis
    assert analysis["correlation"] is not None
    assert len(analysis["correlation"]["columns"]) >= 2
    assert len(analysis["correlation"]["matrix"]) == len(analysis["correlation"]["columns"])

def test_analysis_has_missing_pct():
    info = services.upload_dataset("clf_pct.csv", _read_fixture("classification.csv"))
    analysis = services.get_analysis(info["id"])
    for stat in analysis["column_stats"]:
        assert "missing_pct" in stat

def test_analysis_has_dataset_size():
    info = services.upload_dataset("clf_size.csv", _read_fixture("classification.csv"))
    analysis = services.get_analysis(info["id"])
    assert "dataset_size_bytes" in analysis
    assert analysis["dataset_size_bytes"] > 0


def test_column_selection_creates_authoritative_version_and_restores_original():
    import pandas as pd

    uploaded = services.upload_dataset("selection.csv", _read_fixture("classification.csv"))
    original_path = Path(uploaded["filepath"])
    workspace = services.create_workspace(uploaded["id"])
    stats = services.get_column_statistics(uploaded["id"], target="target")
    assert next(c for c in stats["columns"] if c["name"] == "target")["suggested_role"] == "target"
    assert all("recommendation_reasons" in c for c in stats["columns"])

    selected = services.apply_column_selection(workspace["id"], ["age", "target"], target="target")
    assert selected["removed_columns"] == ["income", "education"]
    assert services.get_workspace(workspace["id"])["dataset_id"] == selected["dataset_id"]
    assert pd.read_csv(services.get_dataset(selected["dataset_id"])["filepath"]).columns.tolist() == ["age", "target"]
    assert original_path.exists()
    with pytest.raises(ValueError, match="target column must be retained"):
        services.apply_column_selection(workspace["id"], ["age"], target="target")

    restored = services.restore_column_selection(workspace["id"])
    assert restored["dataset_id"] == uploaded["id"]
    assert services.get_workspace(workspace["id"])["dataset_id"] == uploaded["id"]
    assert pd.read_csv(original_path).columns.tolist() == uploaded["columns"]

def test_correlation_endpoint():
    info = services.upload_dataset("clf_corr2.csv", _read_fixture("classification.csv"))
    corr = services.get_correlation(info["id"])
    assert "columns" in corr
    assert "matrix" in corr

def test_dataset_not_found():
    with pytest.raises(ValueError, match="not found"):
        services.get_dataset("nonexistent")


def test_profiling_report_generation():
    info = services.upload_dataset("clf_prof.csv", _read_fixture("classification.csv"))
    result = services.generate_profiling_report(info["id"])
    assert result["report_available"] is True
    assert "report_url" in result


# ──────────────────────── Workspace ────────────────────────

def _make_workspace(fixture="classification.csv"):
    ds = services.upload_dataset(f"ws_{id(fixture)}.csv", _read_fixture(fixture))
    ws = services.create_workspace(ds["id"])
    return ws, ds


def test_create_workspace():
    ws, ds = _make_workspace()
    assert ws["status"] == "created"
    assert ws["dataset_id"] == ds["id"]


def test_configure_workspace_classification():
    ws, _ = _make_workspace()
    result = services.configure_workspace(
        ws["id"], "target", "classification",
        {"missing_values": "mean"}, "random_forest_classifier", {"n_estimators": 50})
    assert result["status"] == "configured"
    assert result["model_id"] == "random_forest_classifier"


def test_configure_rejects_invalid_target():
    ws, _ = _make_workspace()
    with pytest.raises(ValueError, match="not in dataset"):
        services.configure_workspace(ws["id"], "nonexistent", "classification",
                                     {}, "logistic_regression", {})


def test_configure_rejects_wrong_problem_model():
    ws, _ = _make_workspace()
    with pytest.raises(ValueError, match="not classification"):
        services.configure_workspace(ws["id"], "target", "classification",
                                     {}, "linear_regression", {})


def test_configure_rejects_regression_on_categorical_target():
    ws, _ = _make_workspace("regression.csv")
    with pytest.raises(ValueError, match="numeric"):
        services.configure_workspace(ws["id"], "location", "regression",
                                     {}, "linear_regression", {})


# ──────────────────────── Training ────────────────────────

def test_train_classification():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"}, "random_forest_classifier",
                                 {"n_estimators": 20})
    result = services.train_workspace(ws["id"])
    assert result["status"] == "trained"
    assert "accuracy" in result["metrics"]
    assert "confusion_matrix" in result["metrics"]
    assert 0 <= result["metrics"]["accuracy"] <= 1


def test_training_reports_real_stages_and_publishes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(services, "STATE_FILE", tmp_path / "state" / "state.json")
    monkeypatch.setattr(services, "_datasets", {})
    monkeypatch.setattr(services, "_workspaces", {})
    dataset = services.upload_dataset("progress.csv", _read_fixture("classification.csv"))
    workspace = services.create_workspace(dataset["id"])
    services.configure_workspace(
        workspace["id"], "target", "classification",
        {"missing_values": "mean", "scaling": None},
        "decision_tree_classifier", {"max_depth": 3},
    )
    progress = []

    result = services.train_workspace(
        workspace["id"],
        job_id="stage-test",
        progress_callback=lambda stage, message: progress.append((stage, message)),
    )

    assert result["status"] == "trained"
    assert [stage for stage, _ in progress] == ["PREPROCESSING", "TRAINING", "EVALUATING"]
    assert Path(services.get_workspace(workspace["id"])["artifacts_dir"], "model.joblib").is_file()
    assert not list((tmp_path / "artifacts" / workspace["id"]).glob(".training-*"))


def test_train_regression():
    ws, _ = _make_workspace("regression.csv")
    services.configure_workspace(ws["id"], "price", "regression",
                                 {"missing_values": "mean", "scaling": "standard"},
                                 "random_forest_regressor", {"n_estimators": 20})
    result = services.train_workspace(ws["id"])
    assert result["status"] == "trained"
    assert "mae" in result["metrics"]
    assert "r2" in result["metrics"]


def test_train_rejects_unconfigured():
    ws, _ = _make_workspace()
    with pytest.raises(ValueError, match="Configure"):
        services.train_workspace(ws["id"])


def test_train_with_scaling():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "scaling": "standard"},
                                 "logistic_regression", {"max_iter": 200})
    result = services.train_workspace(ws["id"])
    assert result["status"] == "trained"


def test_train_with_drop_missing():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "drop"},
                                 "decision_tree_classifier", {})
    result = services.train_workspace(ws["id"])
    assert result["status"] == "trained"


# ──────────────────────── Evaluation ────────────────────────

def test_evaluation_after_training():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {}, "logistic_regression", {"max_iter": 200})
    services.train_workspace(ws["id"])
    ev = services.get_evaluation(ws["id"])
    assert ev["problem_type"] == "classification"
    assert "accuracy" in ev["metrics"]


def test_evaluation_before_training_fails():
    ws, _ = _make_workspace()
    with pytest.raises(ValueError, match="Train"):
        services.get_evaluation(ws["id"])


# ──────────────────────── Prediction ────────────────────────

def test_prediction_uses_saved_pipeline():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "scaling": "standard"},
                                 "random_forest_classifier", {"n_estimators": 20})
    services.train_workspace(ws["id"])

    # Predict using data without target column
    pred_csv = b"age,income,education\n30,45000,16\n40,60000,18\n"
    result = services.predict_workspace(ws["id"], "pred.csv", pred_csv)
    assert result["rows"] == 2
    assert len(result["predictions"]) == 2


def test_prediction_rejects_missing_features():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {}, "decision_tree_classifier", {})
    services.train_workspace(ws["id"])

    pred_csv = b"age,income\n30,45000\n"  # missing 'education'
    with pytest.raises(ValueError, match="Missing feature"):
        services.predict_workspace(ws["id"], "pred.csv", pred_csv)


def test_prediction_before_training_fails():
    ws, _ = _make_workspace()
    with pytest.raises(ValueError, match="trained"):
        services.predict_workspace(ws["id"], "pred.csv", b"a,b\n1,2\n")


def test_prediction_regression():
    ws, _ = _make_workspace("regression.csv")
    services.configure_workspace(ws["id"], "price", "regression",
                                 {"missing_values": "mean"},
                                 "linear_regression", {})
    services.train_workspace(ws["id"])

    pred_csv = b"size,bedrooms,age,location\n1500,3,10,urban\n2000,4,5,suburban\n"
    result = services.predict_workspace(ws["id"], "pred.csv", pred_csv)
    assert result["rows"] == 2
    assert all(isinstance(p, (int, float)) for p in result["predictions"])


# ──────────────────────── Artifacts ────────────────────────

def test_artifacts_after_training():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {}, "decision_tree_classifier", {})
    services.train_workspace(ws["id"])
    arts = services.get_artifacts(ws["id"])
    filenames = [a["filename"] for a in arts]
    assert "model.joblib" in filenames
    assert "metadata.joblib" in filenames
    assert "metrics.joblib" in filenames


def test_artifacts_empty_before_training():
    ws, _ = _make_workspace()
    assert services.get_artifacts(ws["id"]) == []


# ──────────────────────── Leakage Prevention ────────────────────────

def test_no_preprocessing_leakage():
    """Verify preprocessor is fit ONLY on training data (split happens first)."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "scaling": "standard"},
                                 "logistic_regression", {"max_iter": 200})
    services.train_workspace(ws["id"])

    # Load saved metadata and verify split happened
    # Load saved metadata and verify split happened
    import joblib
    metadata = joblib.load(Path(services.get_workspace(ws["id"])["artifacts_dir"]) / "metadata.joblib")
    assert metadata["train_rows"] + metadata["test_rows"] == 20
    assert metadata["has_preprocessor"] is True


def test_vif_analysis():
    info = services.upload_dataset("clf_vif.csv", _read_fixture("classification.csv"))
    result = services.get_vif(info["id"])
    assert "features" in result
    assert len(result["features"]) > 0
    for f in result["features"]:
        assert "feature" in f
        assert "vif" in f


def test_train_with_smote():
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "smote": True},
                                 "random_forest_classifier", {"n_estimators": 20})
    result = services.train_workspace(ws["id"])
    assert result["status"] == "trained"
    assert "accuracy" in result["metrics"]

def test_smote_rejected_for_regression():
    ws, _ = _make_workspace("regression.csv")
    with pytest.raises(ValueError, match="SMOTE"):
        services.configure_workspace(ws["id"], "price", "regression",
                                     {"missing_values": "mean", "smote": True},
                                     "linear_regression", {})

def test_smote_not_applied_to_test_data():
    """Verify SMOTE only affects training data, test data unchanged."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "smote": True},
                                 "random_forest_classifier", {"n_estimators": 20})
    services.train_workspace(ws["id"])
    import joblib
    metadata = joblib.load(Path(services.get_workspace(ws["id"])["artifacts_dir"]) / "metadata.joblib")
    # Test rows should be exactly 20% of original 20 = 4
    assert metadata["test_rows"] == 4


def test_training_runs_history():
    """Verify that multiple training runs are recorded in runs history."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {"C": 1.0})
    res1 = services.train_workspace(ws["id"])
    runs = services.get_workspace_runs(ws["id"])
    assert len(runs) == 1
    assert runs[0]["run_id"] == res1["run_id"]
    assert runs[0]["model_id"] == "logistic_regression"

    # Train again with decision tree
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "decision_tree_classifier", {"max_depth": 3})
    res2 = services.train_workspace(ws["id"])
    runs = services.get_workspace_runs(ws["id"])
    assert len(runs) == 2
    assert runs[1]["run_id"] == res2["run_id"]
    assert runs[1]["model_id"] == "decision_tree_classifier"


def test_config_json_artifact():
    """Verify config.json artifact is generated and listed."""
    import json
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {"C": 1.0})
    res = services.train_workspace(ws["id"])
    artifacts = services.get_artifacts(ws["id"])
    config_art = next((a for a in artifacts if a["filename"] == "config.json"), None)
    assert config_art is not None
    assert config_art["type"] == "config"

    # Verify content
    ws_data = services.get_workspace(ws["id"])
    config_path = Path(ws_data["artifacts_dir"]) / "config.json"
    assert config_path.exists()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    assert cfg["workspace_id"] == ws["id"]
    assert cfg["model_id"] == "logistic_regression"
    assert "metrics" in cfg


def test_prediction_unexpected_columns_warning():
    """Verify unexpected columns in prediction CSV produce warnings."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {"C": 1.0})
    services.train_workspace(ws["id"])

    # CSV with extra columns
    pred_csv = (
        "age,income,education,extra_col1,extra_col2\n"
        "25,50000,12,foo,bar\n"
        "30,60000,16,baz,qux\n"
    ).encode("utf-8")

    result = services.predict_workspace(ws["id"], "predict.csv", pred_csv)
    assert result["rows"] == 2
    assert "warnings" in result
    assert any("Unexpected columns ignored" in w for w in result["warnings"])


def test_prediction_invalid_numeric_type_fails():
    """Verify passing non-numeric text in numerical feature raises ValueError."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {"C": 1.0})
    services.train_workspace(ws["id"])

    # Age is numerical, pass invalid text
    pred_csv = (
        "age,income,education\n"
        "invalid_age,50000,12\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="must be numeric"):
        services.predict_workspace(ws["id"], "predict.csv", pred_csv)


def test_reconfigure_invalidates_active_results():
    """Verify reconfiguration resets status, metrics, artifacts_dir, run_id."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {"C": 1.0})
    services.train_workspace(ws["id"])
    assert services.get_workspace(ws["id"])["status"] == "trained"

    # Reconfigure
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "median"},
                                 "logistic_regression", {"C": 0.5})
    ws_updated = services.get_workspace(ws["id"])
    assert ws_updated["status"] == "configured"
    assert ws_updated["metrics"] is None
    assert ws_updated["artifacts_dir"] is None
    assert ws_updated["run_id"] is None
    # Runs history is preserved
    assert len(ws_updated["runs"]) == 1


def test_train_with_pca():
    """Verify training with PCA dimensionality reduction enabled."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "scaling": "standard", "pca": True, "pca_components": 2},
                                 "logistic_regression", {"C": 1.0})
    res = services.train_workspace(ws["id"])
    assert res["status"] == "trained"
    assert "accuracy" in res["metrics"]


def test_train_with_xgboost_and_svm():
    """Verify training with newly added models (XGBoost and SVM)."""
    ws, _ = _make_workspace()
    # XGBoost
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "xgboost_classifier", {"n_estimators": 20})
    res_xgb = services.train_workspace(ws["id"])
    assert res_xgb["status"] == "trained"

    # SVM
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean", "scaling": "standard"},
                                 "svm_classifier", {"C": 1.0, "kernel": "rbf"})
    res_svm = services.train_workspace(ws["id"])
    assert res_svm["status"] == "trained"


def test_optimize_workspace_mode_a_single_model():
    """Verify Optuna optimization on a single model."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "decision_tree_classifier", {})
    res = services.optimize_workspace(ws["id"], {
        "candidate_models": ["decision_tree_classifier"],
        "metric": "accuracy",
        "n_trials": 3,
        "cv_folds": 2,
    })
    assert res["status"] == "trained"
    assert "optimization" in res
    assert res["optimization"]["best_model_id"] == "decision_tree_classifier"
    assert len(res["optimization"]["trials"]) >= 1

    # Check optuna_results.json artifact
    artifacts = services.get_artifacts(ws["id"])
    opt_art = next((a for a in artifacts if a["filename"] == "optuna_results.json"), None)
    assert opt_art is not None
    assert opt_art["type"] == "optimization"


def test_optimize_workspace_mode_b_multi_model():
    """Verify Optuna optimization comparing multiple candidate models."""
    ws, _ = _make_workspace()
    services.configure_workspace(ws["id"], "target", "classification",
                                 {"missing_values": "mean"},
                                 "logistic_regression", {})
    res = services.optimize_workspace(ws["id"], {
        "candidate_models": ["logistic_regression", "decision_tree_classifier"],
        "metric": "f1",
        "n_trials": 4,
        "cv_folds": 2,
    })
    assert res["status"] == "trained"
    assert res["optimization"]["best_model_id"] in ("logistic_regression", "decision_tree_classifier")
    assert "f1" in res["metrics"]

