from __future__ import annotations

import secrets
import inspect
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .approvals import ApprovalError, TrainingApprovalStore
from .contracts import ToolExecutionResult, ToolRequest


class ToolDefinition:
    def __init__(self, name: str, handler: Callable[..., Any], *, consequential: bool = False) -> None:
        self.name = name
        self.handler = handler
        self.consequential = consequential


class ToolRegistry:
    def __init__(self, approval_store: TrainingApprovalStore | None = None) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._results: dict[str, ToolExecutionResult] = {}
        self.approvals = approval_store or TrainingApprovalStore()

    def register(self, definition: ToolDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._definitions[definition.name] = definition

    def names(self) -> list[str]:
        return sorted(self._definitions)

    def inventory(self) -> list[dict[str, Any]]:
        return [{"name": name, "consequential": definition.consequential}
                for name, definition in sorted(self._definitions.items())]

    def execute(self, request: ToolRequest) -> ToolExecutionResult:
        previous = self._results.get(request.idempotency_key)
        if previous is not None:
            return previous.model_copy(update={"status": "duplicate"})
        definition = self._definitions.get(request.tool_name)
        if definition is None:
            return ToolExecutionResult(operation_id=secrets.token_urlsafe(12), tool_name=request.tool_name, status="blocked", error_code="tool_not_allowed", error_message="Tool is not allowlisted", finished_at=datetime.now(timezone.utc))
        started = datetime.now(timezone.utc)
        try:
            # Bind against the approved backend callable so unknown/missing tool
            # arguments are rejected before reaching a service. OPEN_ACCESS_MODE
            # never skips the AI tool HITL approval check.
            inspect.signature(definition.handler).bind(**request.arguments)
            if definition.name in {"train_workspace", "optimize_workspace"}:
                self._consume_training_approval(request)
            handler_arguments = dict(request.arguments)
            # The configuration is approval evidence, not an argument accepted by
            # the existing train/optimize service functions.
            handler_arguments.pop("configuration", None)
            output = definition.handler(**handler_arguments)
            result = ToolExecutionResult(operation_id=secrets.token_urlsafe(12), tool_name=request.tool_name, status="completed", output=output, started_at=started, finished_at=datetime.now(timezone.utc))
        except ApprovalError as exc:
            result = ToolExecutionResult(operation_id=secrets.token_urlsafe(12), tool_name=request.tool_name, status="blocked", error_code="approval_required", error_message=str(exc), started_at=started, finished_at=datetime.now(timezone.utc))
        except TypeError as exc:
            result = ToolExecutionResult(operation_id=secrets.token_urlsafe(12), tool_name=request.tool_name, status="blocked", error_code="invalid_tool_arguments", error_message=str(exc), started_at=started, finished_at=datetime.now(timezone.utc))
        except Exception as exc:
            if definition.name == "train_workspace":
                try:
                    from services import mark_training_failed
                    mark_training_failed(request.arguments.get("workspace_id"), str(exc))
                except Exception:
                    pass
            result = ToolExecutionResult(operation_id=secrets.token_urlsafe(12), tool_name=request.tool_name, status="failed", error_code="tool_failed", error_message=str(exc), started_at=started, finished_at=datetime.now(timezone.utc))
        self._results[request.idempotency_key] = result
        return result

    def _consume_training_approval(self, request: ToolRequest) -> None:
        approval_id = request.approval_id
        workspace_id = request.arguments.get("workspace_id")
        if not approval_id or not workspace_id:
            raise ApprovalError("Fresh training approval is required")
        configuration = request.arguments.get("configuration")
        if not isinstance(configuration, dict):
            raise ApprovalError("Training request must include the approved configuration fingerprint input")
        self.approvals.consume(approval_id, workspace_id, configuration)


def build_default_registry() -> ToolRegistry:
    from services import (
        apply_column_selection, restore_column_selection, get_column_statistics,
        create_workspace, generate_profiling_report, get_analysis,
        get_artifacts, get_correlation, get_dataset, get_evaluation, get_preview,
        get_vif, get_workspace, get_workspace_optimization, get_workspace_runs,
        predict_workspace, train_workspace, configure_workspace, optimize_workspace,
    )
    from ml import get_model_info, list_models
    from ai.workflow_context import (
        build_workflow_context, get_dataset_context, get_workspace_summary, get_workflow_status,
        get_completed_workflow_stages, get_stage_output, get_preprocessing_results,
        get_model_configuration, get_training_results, get_evaluation_results,
        get_optimization_results, get_final_workflow_summary,
    )

    registry = ToolRegistry()
    registry.register(ToolDefinition("get_column_statistics", get_column_statistics))
    registry.register(ToolDefinition("recommend_column_selection", get_column_statistics))
    registry.register(ToolDefinition("apply_column_selection", apply_column_selection, consequential=True))
    registry.register(ToolDefinition("restore_column_selection", restore_column_selection, consequential=True))
    registry.register(ToolDefinition("get_dataset", get_dataset_context))
    registry.register(ToolDefinition("get_dataset_preview", get_preview))
    registry.register(ToolDefinition("analyze_dataset", get_analysis))
    registry.register(ToolDefinition("get_analysis", get_analysis))
    registry.register(ToolDefinition("get_correlation", get_correlation))
    registry.register(ToolDefinition("get_vif", get_vif))
    registry.register(ToolDefinition("generate_profiling", generate_profiling_report, consequential=True))
    registry.register(ToolDefinition("list_models", list_models))
    registry.register(ToolDefinition("get_model", get_model_info))
    registry.register(ToolDefinition("create_workspace", create_workspace, consequential=True))
    registry.register(ToolDefinition("get_workspace", get_workspace_summary))
    registry.register(ToolDefinition("configure_workspace", configure_workspace, consequential=True))
    registry.register(ToolDefinition("train_workspace", train_workspace, consequential=True))
    registry.register(ToolDefinition("optimize_workspace", optimize_workspace, consequential=True))
    registry.register(ToolDefinition("get_optimization", get_workspace_optimization))
    registry.register(ToolDefinition("get_evaluation", get_evaluation))
    registry.register(ToolDefinition("get_runs", get_workspace_runs))
    registry.register(ToolDefinition("predict_workspace", predict_workspace, consequential=True))
    registry.register(ToolDefinition("get_artifacts", get_artifacts))
    registry.register(ToolDefinition("get_workflow_context", build_workflow_context))
    registry.register(ToolDefinition("get_workspace_context", build_workflow_context))
    registry.register(ToolDefinition("get_dataset_context", get_dataset_context))
    registry.register(ToolDefinition("get_workflow_status", get_workflow_status))
    registry.register(ToolDefinition("get_completed_workflow_stages", get_completed_workflow_stages))
    registry.register(ToolDefinition("get_stage_output", get_stage_output))
    registry.register(ToolDefinition("get_preprocessing_results", get_preprocessing_results))
    registry.register(ToolDefinition("get_model_configuration", get_model_configuration))
    registry.register(ToolDefinition("get_training_results", get_training_results))
    registry.register(ToolDefinition("get_evaluation_results", get_evaluation_results))
    registry.register(ToolDefinition("get_optimization_results", get_optimization_results))
    registry.register(ToolDefinition("get_final_workflow_summary", get_final_workflow_summary))
    return registry

