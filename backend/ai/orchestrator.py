from __future__ import annotations

import json
import re
from typing import Any

from .contracts import ContextBundle, ConversationTurn, IntentKind, IntentResult, WorkflowPlan
from .provider import OpenAICompatibleProvider
from .tools import ToolRegistry, build_default_registry


class AIOrchestrator:
    """Deterministic routing foundation; LLM calls are optional and never execute tools directly."""

    def __init__(self, registry: ToolRegistry | None = None, provider: OpenAICompatibleProvider | None = None) -> None:
        self.registry = registry or build_default_registry()
        self.provider = provider

    def understand_intent(self, message: str, context: ContextBundle | None = None) -> IntentResult:
        text = message.strip()
        if not text:
            return IntentResult(kind=IntentKind.general, confidence=1, missing_information=["message"])
        lowered = text.lower()
        kind = IntentKind.goal
        if any(word in lowered for word in ("approve", "yes, train", "start training")):
            kind = IntentKind.approval
        elif any(word in lowered for word in ("which parameter", "recommend", "suitable", "configuration")):
            kind = IntentKind.recommendation_request
        elif any(word in lowered for word in ("train", "optimize")) or ("predict" in lowered and context and context.workspace_id):
            kind = IntentKind.execution_request
        elif text.endswith("?"):
            kind = IntentKind.question
        workspace_id = context.workspace_id if context else None
        operation = None
        if "classif" in lowered:
            operation = "classification"
        elif "regress" in lowered or "price" in lowered:
            operation = "regression"
        elif "forecast" in lowered or "time series" in lowered:
            operation = "forecasting"
        return IntentResult(kind=kind, objective=text, requested_operation=operation, workspace_id=workspace_id, confidence=0.72)

    def plan(self, intent: IntentResult, workspace: dict[str, Any] | None = None) -> WorkflowPlan:
        status = (workspace or {}).get("status")
        operations: list[str] = []
        missing: list[str] = []
        if not workspace:
            missing.append("dataset and workspace")
            operations.append("upload_dataset")
        elif not workspace.get("target"):
            missing.append("target column")
            operations.extend(["get_dataset", "get_dataset_preview", "configure_workspace"])
        elif status in {"configured", "created"}:
            operations.extend(["get_workspace", "train_workspace"])
        elif status == "trained":
            operations.extend(["get_evaluation", "get_runs", "predict_workspace"])
        return WorkflowPlan(objective=intent.objective, workspace_id=intent.workspace_id, operations=operations, prerequisites=missing, missing_decisions=missing, next_action=operations[0] if operations else "clarify", requires_approval=any(op in {"configure_workspace", "train_workspace", "optimize_workspace"} for op in operations))

    def respond_without_provider(self, intent: IntentResult, plan: WorkflowPlan) -> str:
        if plan.missing_decisions:
            return "I need " + ", ".join(plan.missing_decisions) + " before I can continue."
        return f"I understood this as {intent.kind.value}. The next valid operation is {plan.next_action}."

    def provider_response(self, message: str, context: ContextBundle, *, auto_mode: bool = False, on_text_chunk=None) -> dict[str, Any]:
        if self.provider is None:
            self.provider = OpenAICompatibleProvider()
        workflow_facts = context.facts or {}
        evidence_note = (
            " Authoritative workflow evidence (JSON) follows. Treat completed/current outputs as executed facts, "
            "historical_runs as historical, configured_options as proposed until a completed training stage, "
            "and not_started outputs as unavailable. Never invent missing values: "
            + json.dumps(workflow_facts, default=str)[:24000]
            if workflow_facts else " No workflow outputs are currently attached; ask for the needed dataset/workspace or explain that the stage has not run."
        )
        mode_instruction = (
            "AUTO MODE: choose one next allowlisted operation based on the authoritative current outputs. The backend will execute it and ask you again with its actual result; do not assume future outputs. Select no tool when the task is complete or a human decision is needed. Use only implemented classification/regression operations; clustering and forecasting are unsupported. "
            + "Consequential operations must set needs_approval=true and wait for an explicit user decision in every authentication mode; Open Access does not bypass HITL."
            if auto_mode else "Select at most one read-only tool per turn. Consequential tools must set needs_approval=true."
        )
        provider_messages = [
            {"role": "system", "content": f"Return JSON with exactly these keys: response (string), intent_kind (one of question, goal, recommendation_request, configuration_change, execution_request, approval, general), requested_operation (string or null), needs_clarification (boolean), tool_name (one allowlisted name or null), tool_arguments (object), needs_approval (boolean). Allowlisted tool names are: {', '.join(self.registry.names())}. Never claim a tool ran unless tool evidence is supplied. Treat relevant_memories and conversation_summary as contextual hints, not execution evidence. Current authoritative workflow state, dataset suitability, and technical compatibility take priority over remembered preferences. Follow newer explicit user instructions over older memories. When a remembered preference materially influences a recommendation, explain that briefly; if you choose differently, explain the workflow evidence or compatibility reason. {mode_instruction}" + evidence_note},
        ]
        provider_messages.extend({"role": turn.role, "content": turn.content} for turn in context.turns[-10:] if turn.role in {"user", "assistant"})
        provider_messages.append({"role": "user", "content": message})
        payload = (
            self.provider.complete_json_stream(provider_messages, on_text_chunk)
            if on_text_chunk else self.provider.complete_json(provider_messages)
        )
        required = {"response", "intent_kind", "requested_operation", "needs_clarification", "tool_name", "tool_arguments", "needs_approval"}
        if set(payload) != required or not isinstance(payload["response"], str) or not isinstance(payload["needs_clarification"], bool) or not isinstance(payload["tool_arguments"], dict) or not isinstance(payload["needs_approval"], bool):
            raise ValueError("Provider response did not match the required structured contract")
        if payload["tool_name"] is not None and payload["tool_name"] not in self.registry.names():
            raise ValueError("Provider selected a tool that is not allowlisted")
        return payload
