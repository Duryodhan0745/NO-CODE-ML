from concurrent.futures import ThreadPoolExecutor

import pytest

from ai.approval_workflow import ApprovalWorkflow, ProposalError
from ai.persistence import AIStore
from ai.resource_limits import violations


@pytest.fixture
def proposal_env(monkeypatch, tmp_path):
    workspace = {"id": "ws-hitl", "dataset_id": "dataset-1", "target": None, "problem_type": None,
                 "preprocessing": {}, "model_id": None, "model_params": {}, "column_selection": None}
    dataset = {"id": "dataset-1", "rows": 20, "num_columns": 3, "version": 1, "filepath": str(tmp_path / "data.csv")}
    calls = []
    monkeypatch.setattr("ai.approval_workflow.services.get_workspace", lambda _id: workspace)
    monkeypatch.setattr("ai.approval_workflow.services.get_dataset", lambda _id: dataset)
    monkeypatch.setattr("ai.approval_workflow.services.configure_workspace", lambda workspace_id, **config: calls.append(("configure", workspace_id, config)) or workspace.update(config) or workspace)
    monkeypatch.setattr("ai.approval_workflow.services.train_workspace", lambda workspace_id: calls.append(("train", workspace_id)) or {"workspace_id": workspace_id, "status": "trained"})
    monkeypatch.setattr("ai.approval_workflow.services.optimize_workspace", lambda workspace_id, config: calls.append(("optimize", workspace_id, config)) or {"workspace_id": workspace_id, "status": "trained"})
    monkeypatch.setattr("ai.approval_workflow.services.apply_column_selection", lambda workspace_id, **config: calls.append(("columns", workspace_id, config)) or {"workspace_id": workspace_id})
    return ApprovalWorkflow(AIStore(tmp_path / "control.sqlite3")), workspace, dataset, calls


def test_yes_executes_once_and_persists_result(proposal_env):
    workflow, _, _, calls = proposal_env
    proposal = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration={"target": "y", "problem_type": "regression", "preprocessing": {}, "model_id": "linear_regression", "model_params": {}})
    completed = workflow.decide(proposal["proposal_id"], "user", "execute", proposal["configuration_fingerprint"])
    assert completed["status"] == "completed"
    assert completed["execution"]["success"] is True
    assert len(calls) == 1
    assert calls[0][0:2] == ("configure", "ws-hitl")
    with pytest.raises(ProposalError, match="already been decided|no longer actionable"):
        workflow.decide(proposal["proposal_id"], "user", "execute", proposal["configuration_fingerprint"])


def test_cancel_and_fill_only_never_execute(proposal_env):
    workflow, _, _, calls = proposal_env
    config = {"target": "y", "problem_type": "regression", "preprocessing": {}, "model_id": "linear_regression", "model_params": {}}
    cancelled = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration=config)
    filled = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration=config)
    assert workflow.decide(cancelled["proposal_id"], "user", "cancel", cancelled["configuration_fingerprint"])["status"] == "rejected"
    assert workflow.decide(filled["proposal_id"], "user", "fill_only", filled["configuration_fingerprint"])["status"] == "filled_only"
    assert calls == []


def test_stale_dataset_and_fingerprint_are_rejected(proposal_env):
    workflow, workspace, dataset, _ = proposal_env
    config = {"target": "y", "problem_type": "regression", "preprocessing": {}, "model_id": "linear_regression", "model_params": {}}
    proposal = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration=config)
    with pytest.raises(ProposalError, match="fingerprint"):
        workflow.decide(proposal["proposal_id"], "user", "execute", "0" * 64)
    proposal = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration=config)
    dataset["version"] = 2
    with pytest.raises(ProposalError, match="Dataset or workflow changed"):
        workflow.decide(proposal["proposal_id"], "user", "execute", proposal["configuration_fingerprint"])


def test_changed_workspace_configuration_invalidates_proposal(proposal_env):
    workflow, workspace, _, _ = proposal_env
    config = {"target": "y", "problem_type": "regression", "preprocessing": {}, "model_id": "linear_regression", "model_params": {}}
    proposal = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration=config)
    workspace["model_id"] = "random_forest_regressor"
    with pytest.raises(ProposalError, match="Dataset or workflow changed"):
        workflow.decide(proposal["proposal_id"], "user", "execute", proposal["configuration_fingerprint"])


def test_concurrent_yes_has_one_execution(proposal_env):
    workflow, _, _, calls = proposal_env
    proposal = workflow.create(user_id="user", workspace_id="ws-hitl", stage_id="model_configuration", configuration={"target": "y", "problem_type": "regression", "preprocessing": {}, "model_id": "linear_regression", "model_params": {}})
    def decide():
        try:
            return workflow.decide(proposal["proposal_id"], "user", "execute", proposal["configuration_fingerprint"])["status"]
        except ProposalError:
            return "blocked"
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(lambda _: decide(), range(3)))
    assert outcomes.count("completed") == 1
    assert len(calls) == 1


def test_resource_limits_report_blocking_conditions(monkeypatch):
    monkeypatch.setattr("ai.resource_limits.MAX_DATASET_ROWS", 100)
    monkeypatch.setattr("ai.resource_limits.MAX_DATASET_SIZE_MB", 1)
    monkeypatch.setattr("ai.resource_limits.MAX_MEMORY_MB", 1)
    monkeypatch.setattr("ai.resource_limits.MAX_OPTIMIZATION_TRIALS", 2)
    estimate = {"rows": 101, "dataset_bytes": 2 * 1024 * 1024, "estimated_memory_bytes": 2 * 1024 * 1024}
    problems = violations(estimate, "optimize_workspace", {"n_trials": 3})
    assert len(problems) == 4
    assert all("limit" in problem.lower() or "exceed" in problem.lower() for problem in problems)

