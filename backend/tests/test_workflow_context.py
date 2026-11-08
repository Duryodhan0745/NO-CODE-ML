import pandas as pd

import services
from ai.workflow_context import build_workflow_context


def test_workflow_context_tracks_fresh_real_training_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(services, "STATE_FILE", tmp_path / "state" / "state.json")
    monkeypatch.setattr(services, "_save_state", lambda: None)
    monkeypatch.setattr(services, "_datasets", {})
    monkeypatch.setattr(services, "_workspaces", {})

    source = pd.DataFrame({"feature": range(100), "target": [3 * x + 2 for x in range(100)]})
    dataset = services.upload_dataset("fresh-regression.csv", source.to_csv(index=False).encode())
    workspace = services.create_workspace(dataset["id"])
    services.configure_workspace(workspace["id"], "target", "regression",
        {"missing_values": "mean", "scaling": None}, "linear_regression", {"fit_intercept": True})

    before = build_workflow_context(workspace["id"])
    assert before["stages"][5]["execution_status"] == "configured"
    assert before["stages"][6]["execution_status"] == "not_started"

    trained = services.train_workspace(workspace["id"])
    after = build_workflow_context(workspace["id"])
    training = next(stage for stage in after["stages"] if stage["stage_id"] == "training")
    evaluation = next(stage for stage in after["stages"] if stage["stage_id"] == "evaluation")

    assert trained["status"] == "trained"
    assert training["output_data"]["run_id"] == trained["run_id"]
    assert evaluation["execution_status"] == "completed"
    assert evaluation["output_data"]["metrics"] == trained["metrics"]
    assert evaluation["configuration_version"] == after["configuration_version"]
    assert evaluation["artifact_references"]


def test_reconfiguration_marks_old_evaluation_as_historical(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(services, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(services, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(services, "STATE_FILE", tmp_path / "state" / "state.json")
    monkeypatch.setattr(services, "_save_state", lambda: None)
    monkeypatch.setattr(services, "_datasets", {})
    monkeypatch.setattr(services, "_workspaces", {})
    source = pd.DataFrame({"feature": range(100), "target": [2 * x + 1 for x in range(100)]})
    dataset = services.upload_dataset("stale-regression.csv", source.to_csv(index=False).encode())
    workspace = services.create_workspace(dataset["id"])
    services.configure_workspace(workspace["id"], "target", "regression", {"missing_values": "mean", "scaling": None}, "linear_regression", {"fit_intercept": True})
    services.train_workspace(workspace["id"])
    old_run_id = services.get_workspace(workspace["id"])["run_id"]

    services.configure_workspace(workspace["id"], "target", "regression", {"missing_values": "median", "scaling": None}, "linear_regression", {"fit_intercept": True})
    context = build_workflow_context(workspace["id"])
    evaluation = next(stage for stage in context["stages"] if stage["stage_id"] == "evaluation")

    assert evaluation["execution_status"] == "not_started"
    assert context["historical_runs"][0]["run_id"] == old_run_id
    assert context["historical_runs"][0]["current"] is False

