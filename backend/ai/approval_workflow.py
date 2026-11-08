"""Persistent three-decision HITL proposals over the existing ML services."""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import services
from .approvals import TrainingApprovalStore
from .resource_limits import enforce, limits, violations


class ProposalError(ValueError):
    pass


STAGE_TO_TOOL = {
    "model_configuration": "configure_workspace",
    "training": "train_workspace",
    "optimization": "optimize_workspace",
    "column_selection": "apply_column_selection",
}


def fingerprint(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def workflow_version(workspace: dict[str, Any]) -> str:
    state = {key: workspace.get(key) for key in ("dataset_id", "target", "problem_type", "preprocessing", "model_id", "model_params", "column_selection")}
    return fingerprint(state)[:16]


def estimate_resources(workspace: dict[str, Any], tool_name: str, configuration: dict[str, Any]) -> dict[str, Any]:
    dataset = services.get_dataset(workspace["dataset_id"])
    rows, features = dataset["rows"], max(0, dataset["num_columns"] - 1)
    size = None
    try:
        from pathlib import Path
        size = Path(dataset["filepath"]).stat().st_size
    except OSError:
        pass
    trials = configuration.get("n_trials") or configuration.get("n_iter")
    folds = configuration.get("cv_folds", 1)
    if tool_name == "optimize_workspace" and trials:
        workload = {"kind": "cross_validated_training_runs", "candidate_runs": int(trials) * int(folds), "is_estimate": True}
    elif tool_name == "train_workspace":
        workload = {"kind": "single_train_test_fit", "candidate_runs": 1, "is_estimate": True}
    else:
        workload = {"kind": "configuration_or_dataset_update", "candidate_runs": 0, "is_estimate": True}
    return {"dataset_bytes": size, "rows": rows, "features": features,
            "estimated_memory_bytes": rows * max(features, 1) * 16 if rows and features else None,
            "estimated_runtime_seconds": None, "estimate_is_approximate": True,
            "measured": {"dataset_bytes": size is not None, "rows": True, "features": True},
            "workload": workload, "limits": limits()}


class ApprovalWorkflow:
    def __init__(self, store) -> None:
        self.store = store

    def create(self, *, user_id: str, workspace_id: str, stage_id: str, configuration: dict[str, Any],
               conversation_id: str | None = None, rationale: str = "AI-proposed configuration", expected_outcome: str = "Execute the selected workflow stage", warnings: list[str] | None = None,
               expires_minutes: int = 30) -> dict[str, Any]:
        workspace = services.get_workspace(workspace_id)
        tool_name = STAGE_TO_TOOL.get(stage_id)
        if not tool_name:
            raise ProposalError(f"Unsupported approval stage: {stage_id}")
        if not isinstance(configuration, dict) or not configuration:
            raise ProposalError("A complete configuration is required")
        now = datetime.now(timezone.utc)
        resource_usage = estimate_resources(workspace, tool_name, configuration)
        limit_warnings = violations(resource_usage, tool_name, configuration)
        proposal = {"proposal_id": secrets.token_urlsafe(18), "user_id": user_id, "workspace_id": workspace_id,
                    "conversation_id": conversation_id, "stage_id": stage_id, "tool_name": tool_name,
                    "configuration": configuration, "configuration_fingerprint": fingerprint(configuration),
                    "dataset_version": str(services.get_dataset(workspace["dataset_id"]).get("version", 1)),
                    "workflow_version": workflow_version(workspace), "resource_usage": resource_usage,
                    "rationale": rationale, "expected_outcome": expected_outcome, "warnings": (warnings or []) + limit_warnings,
                    "status": "failed" if limit_warnings else "awaiting_approval", "expires_at": (now + timedelta(minutes=expires_minutes)).isoformat(),
                    "created_at": now.isoformat(), "updated_at": now.isoformat()}
        self.store.save_workflow_proposal(proposal)
        return self.public(proposal)

    def public(self, proposal: dict[str, Any]) -> dict[str, Any]:
        proposal = dict(proposal); proposal.pop("user_id", None)
        return proposal

    def get(self, proposal_id: str, user_id: str) -> dict[str, Any]:
        proposal = self.store.get_workflow_proposal(proposal_id)
        if not proposal or proposal["user_id"] != user_id:
            raise ProposalError("Approval proposal not found")
        return self.public(proposal)

    def list(self, workspace_id: str, user_id: str, actionable_only: bool = False) -> list[dict[str, Any]]:
        return [self.public(item) for item in self.store.list_workflow_proposals(workspace_id, user_id, actionable_only=actionable_only)]

    def decide(self, proposal_id: str, user_id: str, decision: str, configuration_fingerprint: str) -> dict[str, Any]:
        proposal = self.store.get_workflow_proposal(proposal_id)
        if not proposal or proposal["user_id"] != user_id:
            raise ProposalError("Approval proposal not found")
        if proposal["status"] != "awaiting_approval":
            raise ProposalError(f"Approval proposal is no longer actionable ({proposal['status']})")
        if proposal["configuration_fingerprint"] != configuration_fingerprint:
            self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "stale")
            raise ProposalError("Approval configuration fingerprint is stale")
        if datetime.fromisoformat(proposal["expires_at"]).astimezone(timezone.utc) <= datetime.now(timezone.utc):
            self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "expired")
            raise ProposalError("Approval proposal has expired")
        workspace = services.get_workspace(proposal["workspace_id"])
        current_dataset_version = str(services.get_dataset(workspace["dataset_id"]).get("version", 1))
        if current_dataset_version != proposal["dataset_version"] or workflow_version(workspace) != proposal["workflow_version"]:
            self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "stale")
            raise ProposalError("Dataset or workflow changed after this proposal was prepared")
        if decision == "cancel":
            result = self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "rejected")
            if not result: raise ProposalError("Approval proposal is no longer actionable")
            return self.public(result)
        if decision == "fill_only":
            result = self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "filled_only")
            if not result: raise ProposalError("Approval proposal is no longer actionable")
            return self.public(result)
        if decision != "execute":
            raise ProposalError("Decision must be execute, cancel, or fill_only")
        enforce(proposal["resource_usage"], proposal["tool_name"], proposal["configuration"])
        claimed = self.store.transition_workflow_proposal(proposal_id, user_id, ("awaiting_approval",), "executing")
        if not claimed: raise ProposalError("Approval proposal has already been decided")
        try:
            execution = self._execute(claimed)
            final = self.store.transition_workflow_proposal(proposal_id, user_id, ("executing",), "completed", execution=execution)
        except Exception as exc:
            final = self.store.transition_workflow_proposal(proposal_id, user_id, ("executing",), "failed", execution={"success": False, "error": str(exc)})
        return self.public(final)

    def _execute(self, proposal: dict[str, Any]) -> dict[str, Any]:
        configuration, workspace_id, tool = proposal["configuration"], proposal["workspace_id"], proposal["tool_name"]
        if tool == "configure_workspace":
            return {"success": True, "output": services.configure_workspace(workspace_id, **configuration)}
        if tool == "apply_column_selection":
            return {"success": True, "output": services.apply_column_selection(workspace_id, **configuration)}
        if tool == "train_workspace":
            # A training proposal can include the exact configuration to apply first.
            config = configuration.get("configuration")
            if config:
                services.configure_workspace(workspace_id, **config)
            return {"success": True, "output": services.train_workspace(workspace_id)}
        if tool == "optimize_workspace":
            config = configuration.get("configuration")
            if config:
                services.configure_workspace(workspace_id, **config)
            return {"success": True, "output": services.optimize_workspace(workspace_id, configuration.get("optimization", configuration))}
        raise ProposalError("Unsupported proposal tool")
