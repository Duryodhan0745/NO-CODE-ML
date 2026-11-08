"""Authoritative, bounded context over the existing ML workflow state."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import services
from ml import MODEL_REGISTRY


class WorkflowStage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str
    dataset_id: str
    workflow_id: str
    stage_id: str
    stage_name: str
    execution_status: str
    output_data: Any = None
    execution_timestamp: str | None = None
    configuration_version: str | None = None
    artifact_references: list[str] = Field(default_factory=list)
    error_information: str | None = None


def _stage(workspace_id: str, dataset_id: str, stage_id: str, name: str, status: str,
           output: Any = None, timestamp: str | None = None, version: str | None = None,
           artifacts: list[str] | None = None, error: str | None = None) -> dict:
    return WorkflowStage(workspace_id=workspace_id, dataset_id=dataset_id,
        workflow_id=workspace_id, stage_id=stage_id, stage_name=name,
        execution_status=status, output_data=output, execution_timestamp=timestamp,
        configuration_version=version, artifact_references=artifacts or [],
        error_information=error).model_dump(mode="json")


def build_workflow_context(workspace_id: str) -> dict:
    """Read current outputs from services; historical runs stay explicitly historical."""
    workspace = services.get_workspace(workspace_id)
    dataset_id = workspace["dataset_id"]
    dataset = services.get_dataset(dataset_id)
    config = {k: workspace.get(k) for k in ("target", "problem_type", "preprocessing", "model_id", "model_params")}
    version = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]
    configured = bool(workspace.get("target") and workspace.get("model_id"))
    trained = workspace.get("status") == "trained" and bool(workspace.get("metrics"))
    run = next((r for r in reversed(workspace.get("runs", [])) if r.get("run_id") == workspace.get("run_id")), None)
    artifacts = []
    art_dir = workspace.get("artifacts_dir")
    if art_dir and Path(art_dir).exists():
        artifacts = [f"/api/workspaces/{workspace_id}/artifacts/{p.name}" for p in Path(art_dir).iterdir() if p.is_file()]

    stages = [
        _stage(workspace_id, dataset_id, "dataset_ingestion", "Dataset ingestion", "completed",
               {k: dataset.get(k) for k in ("id", "filename", "rows", "num_columns", "columns", "dtypes", "missing_values", "duplicate_rows", "validation")}),
        _stage(workspace_id, dataset_id, "column_selection", "Remove unnecessary columns",
               "completed" if workspace.get("column_selection") else "available",
               workspace.get("column_selection") or {"recommendations": "Available from get_column_statistics; recommendations are metadata heuristics requiring user review"},
               version=workspace.get("column_selection", {}).get("fingerprint")),
        _stage(workspace_id, dataset_id, "eda", "Exploratory data analysis", "available",
               services.get_analysis(dataset_id)),
        _stage(workspace_id, dataset_id, "preprocessing", "Preprocessing", "completed" if trained else ("configured" if configured else "not_started"),
               {"applied_configuration": workspace.get("preprocessing", {}) if trained else None,
                "configured_options": workspace.get("preprocessing", {}),
                "training_transform_details": {k: run.get(k) for k in ("train_rows", "test_rows", "feature_columns", "numerical_columns", "categorical_columns", "smote_applied")} if run else None},
               run.get("timestamp") if run else None, version, artifacts if trained else []),
        _stage(workspace_id, dataset_id, "model_selection", "Model selection", "completed" if configured else "not_started",
               {"available_algorithms": [{"model_id": k, "problem_type": v.get("problem_type")} for k, v in MODEL_REGISTRY.items()],
                "selected_model": workspace.get("model_id"), "problem_type": workspace.get("problem_type")}, version=version),
        _stage(workspace_id, dataset_id, "model_configuration", "Model configuration", "configured" if configured else "not_started",
               config if configured else None, version=version),
        _stage(workspace_id, dataset_id, "training", "Model training", "completed" if trained else ("failed" if workspace.get("status") == "failed" else "not_started"),
               {k: run.get(k) for k in ("run_id", "status", "timestamp", "model_id", "model_params", "preprocessing", "metrics", "train_rows", "test_rows", "feature_columns", "numerical_columns", "categorical_columns", "training_duration_seconds", "smote_applied")} if run else None,
               run.get("timestamp") if run else None, version, artifacts),
        _stage(workspace_id, dataset_id, "evaluation", "Evaluation", "completed" if trained else "not_started",
               services.get_evaluation(workspace_id) if trained else None,
               run.get("timestamp") if run else None, version, artifacts),
        _stage(workspace_id, dataset_id, "optimization", "Hyperparameter optimization", "completed" if trained and workspace.get("optimization") else "not_started",
               workspace.get("optimization"), run.get("timestamp") if run and workspace.get("optimization") else None, version, artifacts),
        _stage(workspace_id, dataset_id, "final_summary", "Final workflow summary", "completed" if trained else "in_progress",
               {"workspace_status": workspace.get("status"), "selected_model": workspace.get("model_id"),
                "problem_type": workspace.get("problem_type"), "metrics": workspace.get("metrics"),
                "artifacts": artifacts, "limitations": ["Only classification and regression are currently implemented by the ML configuration service."]}),
    ]
    history = [{"run_id": r.get("run_id"), "status": r.get("status"), "model_id": r.get("model_id"),
                "metrics": r.get("metrics"), "timestamp": r.get("timestamp"),
                "current": r.get("run_id") == workspace.get("run_id")} for r in workspace.get("runs", [])]
    return {"workspace_id": workspace_id, "dataset_id": dataset_id, "workflow_id": workspace_id,
            "workspace_status": workspace.get("status"), "configuration_version": version,
            "stages": stages, "historical_runs": history,
            "dataset_preview": services.get_preview(dataset_id, rows=5)}


def get_stage_output(workspace_id: str, stage_id: str) -> dict:
    context = build_workflow_context(workspace_id)
    stage = next((item for item in context["stages"] if item["stage_id"] == stage_id), None)
    if stage is None:
        raise ValueError(f"Unknown workflow stage: {stage_id}")
    return stage


def get_workflow_status(workspace_id: str) -> dict:
    context = build_workflow_context(workspace_id)
    return {k: context[k] for k in ("workspace_id", "dataset_id", "workflow_id", "workspace_status", "configuration_version")}


def get_completed_workflow_stages(workspace_id: str) -> list[dict]:
    return [stage for stage in build_workflow_context(workspace_id)["stages"] if stage["execution_status"] == "completed"]


def get_dataset_context(dataset_id: str) -> dict:
    dataset = services.get_dataset(dataset_id)
    return {k: dataset.get(k) for k in ("id", "filename", "rows", "num_columns", "columns", "dtypes", "missing_values", "duplicate_rows", "validation", "parent_dataset_id", "version", "selection_fingerprint")}


def get_workspace_summary(workspace_id: str) -> dict:
    context = build_workflow_context(workspace_id)
    return {k: context[k] for k in ("workspace_id", "dataset_id", "workflow_id", "workspace_status", "configuration_version", "stages")}


def get_stage_by_name(workspace_id: str, stage_id: str) -> dict:
    return get_stage_output(workspace_id, stage_id)["output_data"]


def _named_stage(workspace_id: str, stage_id: str) -> dict:
    return get_stage_output(workspace_id, stage_id)


def get_preprocessing_results(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "preprocessing")


def get_model_configuration(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "model_configuration")


def get_training_results(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "training")


def get_evaluation_results(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "evaluation")


def get_optimization_results(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "optimization")


def get_final_workflow_summary(workspace_id: str) -> dict:
    return _named_stage(workspace_id, "final_summary")
