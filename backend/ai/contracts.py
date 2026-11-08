from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentKind(str, Enum):
    question = "question"
    goal = "goal"
    recommendation_request = "recommendation_request"
    configuration_change = "configuration_change"
    execution_request = "execution_request"
    approval = "approval"
    general = "general"


class IntentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: IntentKind
    objective: str | None = None
    requested_operation: str | None = None
    workspace_id: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class DatasetContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    rows: int | None = None
    columns: list[str] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing_values: dict[str, int | float] = Field(default_factory=dict)
    duplicate_rows: int | None = None
    analysis: dict[str, Any] | None = None
    correlation: dict[str, Any] | None = None
    vif: dict[str, Any] | None = None
    source_operations: list[str] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objective: str | None = None
    workspace_id: str | None = None
    operations: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    missing_decisions: list[str] = Field(default_factory=list)
    next_action: str | None = None
    requires_approval: bool = False


class RecommendationCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendation_id: str
    operation: Literal["configure_workspace", "optimize_workspace"]
    workspace_id: str
    model_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str
    limitations: list[str] = Field(default_factory=list)
    actions: list[Literal["continue", "edit", "explain", "try_another", "reject"]] = Field(
        default_factory=lambda: ["continue", "edit", "explain", "try_another", "reject"]
    )
    status: Literal["proposed", "validated", "rejected"] = "proposed"


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str = Field(min_length=8, max_length=128)
    approval_id: str | None = None


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation_id: str
    tool_name: str
    status: Literal["completed", "failed", "blocked", "duplicate"]
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: datetime | None = None


class TrainingApproval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approval_id: str
    workspace_id: str
    configuration_fingerprint: str
    approved_by: str
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: Literal["approved", "consumed", "revoked", "expired"] = "approved"


class ConversationTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant", "tool"]
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tool_name: str | None = None
    evidence: dict[str, Any] | None = None


class ContextBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    workspace_id: str | None = None
    turns: list[ConversationTurn] = Field(default_factory=list)
    memories: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class ExecutionMetrics(BaseModel):
    duration_seconds: float | None = Field(default=None, ge=0)
    rows_processed: int | None = Field(default=None, ge=0)
    features_processed: int | None = Field(default=None, ge=0)


class ResourceUsage(BaseModel):
    dataset_bytes: int | None = Field(default=None, ge=0)
    estimated_runtime_seconds: float | None = Field(default=None, ge=0)
    estimated_memory_bytes: int | None = Field(default=None, ge=0)
    estimate_is_approximate: bool = True


class WorkflowOutput(BaseModel):
    """Shared typed envelope for backend-confirmed stage output."""
    model_config = ConfigDict(extra="forbid")
    success: bool
    workflow_id: str
    workspace_id: str
    dataset_id: str
    stage_id: str
    stage_name: str
    execution_id: str | None = None
    execution_status: Literal["available", "configured", "running", "completed", "failed", "stale", "not_started"]
    input_summary: dict[str, Any] = Field(default_factory=dict)
    applied_configuration: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    output_data: Any = None
    artifact_references: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution_metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    configuration_fingerprint: str | None = None
    dataset_version: str | int | None = None
    stage_version: str | None = None


class DatasetIngestionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    metadata: dict[str, Any]
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    dataset_schema: dict[str, str] = Field(alias="schema")
    validation_results: dict[str, Any]


class ColumnSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    original_columns: list[str]
    retained_columns: list[str]
    removed_columns: list[str]
    removal_reasons: dict[str, str] = Field(default_factory=dict)
    updated_dataset_id: str
    updated_schema: dict[str, str]
    dataset_version: int = Field(ge=1)
    fingerprint: str


class DatasetExplorationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    descriptive_statistics: dict[str, Any] = Field(default_factory=dict)
    missing_values: dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = Field(ge=0)
    distributions: dict[str, Any] = Field(default_factory=dict)
    correlations: dict[str, Any] = Field(default_factory=dict)
    target_information: dict[str, Any] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)


class FeatureSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_features: list[str]
    rejected_features: list[str]
    scores: dict[str, float] = Field(default_factory=dict)
    selection_method: str
    leakage_warnings: list[str] = Field(default_factory=list)


class ModelConfigurationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hyperparameters: dict[str, Any]
    validation_strategy: dict[str, Any] = Field(default_factory=dict)
    training_configuration: dict[str, Any] = Field(default_factory=dict)
    evaluation_configuration: dict[str, Any] = Field(default_factory=dict)


class EvaluationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics: dict[str, float | int | None]
    predictions: list[Any] = Field(default_factory=list)
    confusion_matrix: list[list[int]] | None = None
    residuals: list[float] | None = None
    summary: dict[str, Any] = Field(default_factory=dict)


class OptimizationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_method: str
    search_space: dict[str, Any]
    best_parameters: dict[str, Any] = Field(default_factory=dict)
    best_score: float | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class FinalWorkflowSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    completed_stages: list[str] = Field(default_factory=list)
    failed_stages: list[str] = Field(default_factory=list)
    selected_model: str | None = None
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    actual_metrics: dict[str, float | int | None] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    recommended_next_steps: list[str] = Field(default_factory=list)


class PreprocessingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    missing_value_strategy: str | None = None
    encoding: str | None = None
    scaling: str | None = None
    outlier_treatment: str | None = None
    before_statistics: dict[str, Any] = Field(default_factory=dict)
    after_statistics: dict[str, Any] = Field(default_factory=dict)
    applied_configuration: dict[str, Any] = Field(default_factory=dict)


class ModelSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    selected_algorithm: str
    task_compatibility: Literal["classification", "regression"]
    supported_parameters: list[str] = Field(default_factory=list)
    rationale: str | None = None


class TrainingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str
    training_status: Literal["completed", "failed"]
    duration_seconds: float | None = Field(default=None, ge=0)
    train_rows: int | None = Field(default=None, ge=0)
    test_rows: int | None = Field(default=None, ge=0)
    training_logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class HITLProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: str
    workspace_id: str
    stage_id: str
    configuration: dict[str, Any]
    configuration_fingerprint: str
    dataset_version: str | int
    workflow_version: str
    reasoning: str
    expected_outcome: str
    warnings: list[str] = Field(default_factory=list)
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    status: Literal["draft", "validating", "awaiting_approval", "approved", "rejected", "cancelled", "filled_only", "executing", "completed", "failed", "expired", "stale"]
    expires_at: datetime | None = None


class HITLDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposal_id: str
    decision: Literal["execute", "cancel", "fill_only"]
    configuration_fingerprint: str

