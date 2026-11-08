"""API endpoint tests using FastAPI TestClient."""

import os
import tempfile
from pathlib import Path

# Override dirs before import
os.environ["UPLOAD_DIR"] = str(Path(tempfile.mkdtemp()) / "test_uploads")
os.environ["ARTIFACTS_DIR"] = str(Path(tempfile.mkdtemp()) / "test_artifacts")

import pytest
from fastapi.testclient import TestClient
from conftest import FIXTURES_DIR

from main import app

client = TestClient(app)


def _upload(name="classification.csv"):
    data = (FIXTURES_DIR / name).read_bytes()
    resp = client.post("/api/datasets/upload", files={"file": (name, data, "text/csv")})
    assert resp.status_code == 200
    return resp.json()


# ──────────────────────── Health ────────────────────────

def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ──────────────────────── Dataset endpoints ────────────────────────

def test_upload_endpoint():
    j = _upload()
    assert j["success"] is True
    assert j["rows"] == 20


def test_upload_rejects_non_csv():
    r = client.post("/api/datasets/upload", files={"file": ("data.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_preview_endpoint():
    j = _upload()
    r = client.get(f"/api/datasets/{j['id']}/preview?rows=5")
    assert r.status_code == 200
    assert len(r.json()["data"]) == 5


def test_analysis_endpoint():
    j = _upload()
    r = client.get(f"/api/datasets/{j['id']}/analysis")
    assert r.status_code == 200
    body = r.json()
    assert body["rows"] == 20
    assert len(body["column_stats"]) == 4

def test_correlation_endpoint():
    j = _upload()
    r = client.get(f"/api/datasets/{j['id']}/correlation")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "columns" in body
    assert "matrix" in body


def test_vif_endpoint():
    j = _upload()
    r = client.get(f"/api/datasets/{j['id']}/vif")
    assert r.status_code == 200
    assert r.json()["success"] is True
    assert "features" in r.json()


def test_dataset_not_found():
    r = client.get("/api/datasets/nonexistent/preview")
    assert r.status_code == 404


def test_profiling_endpoint():
    j = _upload()
    r = client.post(f"/api/datasets/{j['id']}/profiling")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True

def test_profiling_report_not_found_before_generation():
    j = _upload()
    r = client.get(f"/api/datasets/{j['id']}/profiling/report")
    assert r.status_code == 404


# ──────────────────────── Model endpoints ────────────────────────

def test_list_all_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    assert len(r.json()["models"]) == 18


def test_list_classification_models():
    r = client.get("/api/models?problem_type=classification")
    models = r.json()["models"]
    assert len(models) == 8
    assert all(m["problem_type"] == "classification" for m in models)


def test_get_model_detail():
    r = client.get("/api/models/random_forest_classifier")
    assert r.status_code == 200
    assert r.json()["model_id"] == "random_forest_classifier"


def test_model_not_found():
    r = client.get("/api/models/nonexistent")
    assert r.status_code == 404


# ──────────────────────── Workspace + full workflow ────────────────────────

def _full_classification_workflow():
    """Helper: upload → workspace → configure → train → evaluate."""
    ds = _upload()
    r = client.post("/api/workspaces", json={"dataset_id": ds["id"]})
    assert r.status_code == 200
    ws_id = r.json()["id"]

    r = client.post(f"/api/workspaces/{ws_id}/configure", json={
        "target": "target",
        "problem_type": "classification",
        "preprocessing": {"missing_values": "mean"},
        "model_id": "random_forest_classifier",
        "model_params": {"n_estimators": 20},
    })
    assert r.status_code == 200
    assert r.json()["status"] == "configured"

    r = client.post(f"/api/workspaces/{ws_id}/train")
    assert r.status_code == 200
    assert r.json()["status"] == "trained"
    assert "accuracy" in r.json()["metrics"]

    return ws_id


def test_full_classification_workflow():
    ws_id = _full_classification_workflow()
    # Evaluation
    r = client.get(f"/api/workspaces/{ws_id}/evaluation")
    assert r.status_code == 200
    assert r.json()["problem_type"] == "classification"


def test_full_regression_workflow():
    ds = _upload("regression.csv")
    r = client.post("/api/workspaces", json={"dataset_id": ds["id"]})
    ws_id = r.json()["id"]

    r = client.post(f"/api/workspaces/{ws_id}/configure", json={
        "target": "price",
        "problem_type": "regression",
        "preprocessing": {"missing_values": "mean", "scaling": "standard"},
        "model_id": "random_forest_regressor",
        "model_params": {"n_estimators": 20},
    })
    assert r.status_code == 200

    r = client.post(f"/api/workspaces/{ws_id}/train")
    assert r.status_code == 200
    assert "r2" in r.json()["metrics"]

    r = client.get(f"/api/workspaces/{ws_id}/evaluation")
    assert r.status_code == 200
    assert r.json()["problem_type"] == "regression"


def test_predict_endpoint():
    ws_id = _full_classification_workflow()
    pred_csv = b"age,income,education\n30,45000,16\n40,60000,18\n"
    r = client.post(f"/api/workspaces/{ws_id}/predict",
                    files={"file": ("pred.csv", pred_csv, "text/csv")})
    assert r.status_code == 200
    assert r.json()["rows"] == 2
    assert len(r.json()["predictions"]) == 2


def test_artifacts_endpoint():
    ws_id = _full_classification_workflow()
    r = client.get(f"/api/workspaces/{ws_id}/artifacts")
    assert r.status_code == 200
    filenames = [a["filename"] for a in r.json()["artifacts"]]
    assert "model.joblib" in filenames


def test_artifact_download():
    ws_id = _full_classification_workflow()
    r = client.get(f"/api/workspaces/{ws_id}/artifacts/model.joblib")
    assert r.status_code == 200
    assert len(r.content) > 0


def test_configure_rejects_invalid_model():
    ds = _upload()
    r = client.post("/api/workspaces", json={"dataset_id": ds["id"]})
    ws_id = r.json()["id"]
    r = client.post(f"/api/workspaces/{ws_id}/configure", json={
        "target": "target", "problem_type": "classification",
        "model_id": "linear_regression", "model_params": {},
    })
    assert r.status_code == 400


def test_train_before_configure_fails():
    ds = _upload()
    r = client.post("/api/workspaces", json={"dataset_id": ds["id"]})
    ws_id = r.json()["id"]
    r = client.post(f"/api/workspaces/{ws_id}/train")
    assert r.status_code == 400


def test_path_traversal_blocked():
    ws_id = _full_classification_workflow()
    r = client.get(f"/api/workspaces/{ws_id}/artifacts/../../etc/passwd")
    # Should either 404 or return just the base name
    assert r.status_code == 404


def test_workspace_runs_endpoint():
    ws_id = _full_classification_workflow()
    r = client.get(f"/api/workspaces/{ws_id}/runs")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert "runs" in data
    assert len(data["runs"]) >= 1
    assert data["runs"][0]["model_id"] == "random_forest_classifier"


def test_workspace_optimize_endpoint():
    ds = _upload()
    r = client.post("/api/workspaces", json={"dataset_id": ds["id"]})
    ws_id = r.json()["id"]

    # Configure base target
    client.post(f"/api/workspaces/{ws_id}/configure", json={
        "target": "target",
        "problem_type": "classification",
        "preprocessing": {"missing_values": "mean"},
        "model_id": "logistic_regression",
        "model_params": {},
    })

    # Run Optuna optimization
    opt_r = client.post(f"/api/workspaces/{ws_id}/optimize", json={
        "candidate_models": ["logistic_regression", "decision_tree_classifier"],
        "metric": "accuracy",
        "n_trials": 3,
        "cv_folds": 2,
    })
    assert opt_r.status_code == 200
    res = opt_r.json()
    assert res["success"] is True
    assert "optimization" in res
    assert res["optimization"]["best_model_id"] in ("logistic_regression", "decision_tree_classifier")

    # Check GET optimization
    get_opt = client.get(f"/api/workspaces/{ws_id}/optimization")
    assert get_opt.status_code == 200
    assert get_opt.json()["optimization"] is not None


