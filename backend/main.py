"""FastAPI application — all routes."""

import logging
import hashlib
import json
import time
import threading
import io
import tempfile
import queue
from contextvars import ContextVar
from threading import Event, Thread
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Header, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ai.approvals import ApprovalError, TrainingApprovalStore
from ai.auth import AuthenticationError, AuthorizationError, Principal, verify_bearer_token, verify_google_id_token, issue_test_token
from ai.contracts import ContextBundle, ConversationTurn, ToolRequest, WorkflowOutput, ExecutionMetrics, ResourceUsage
from ai.orchestrator import AIOrchestrator
from ai.provider import ProviderConfigurationError, ProviderRequestError
from ai.persistence import AIStore
from ai.workflow_context import build_workflow_context
from ai.memory import extract_durable_memories
from ai.approval_workflow import ApprovalWorkflow, ProposalError

import services
from ml import list_models as ml_list_models, get_model_info
from schemas import CreateWorkspaceRequest, ConfigureRequest, OptimizeRequest
from config import UPLOAD_DIR, ARTIFACTS_DIR, STATE_DIR, AI_DB_FILE, GOOGLE_CLIENT_ID, ALLOWED_ORIGINS, AI_RATE_LIMIT_CONVERSATION, AI_RATE_LIMIT_TOOL, AI_RATE_LIMIT_UPLOAD, AI_RATE_LIMIT_WINDOW_SECONDS, ENABLE_TEST_ADMIN, ADMIN_EMAIL, ADMIN_PASSWORD, OPEN_ACCESS_MODE_ENABLED, OPEN_ACCESS_USER_ID
import data_playground as playground
from training_jobs import training_jobs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
_ai_stream_events: ContextVar[object | None] = ContextVar("ai_stream_events", default=None)


def _emit_ai_stream_event(event: dict) -> None:
    callback = _ai_stream_events.get()
    if callback:
        callback(event)


def _ai_text_chunk_callback():
    callback = _ai_stream_events.get()
    return (lambda text: callback({"type": "delta", "text": text})) if callback else None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        training_jobs.shutdown()


app = FastAPI(title="No-Code ML Platform", version="1.0.0", lifespan=lifespan)


class ColumnSelectionRequest(BaseModel):
    retained_columns: list[str] = Field(min_length=1)
    target: str | None = None

    model_config = {"extra": "forbid"}


class PlaygroundHandoffRequest(BaseModel):
    source_dataset_id: str = Field(min_length=1, max_length=100)
    source_version: int = Field(ge=1)
    ml_workspace_id: str | None = None

    model_config = {"extra": "forbid"}

_rate_hits = {}
_rate_lock = threading.Lock()

@app.middleware("http")
async def request_guard(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"detail": {"code": "origin_not_allowed", "message": "Origin is not allowed"}}, status_code=403)
        path = request.url.path
        if path.startswith("/api/ai/conversations") and request.method == "POST":
            limit = AI_RATE_LIMIT_CONVERSATION
            bucket = "conversation"
        elif path.startswith("/api/ai/tools"):
            limit = AI_RATE_LIMIT_TOOL
            bucket = "tool"
        elif path.startswith("/api/datasets/upload"):
            limit = AI_RATE_LIMIT_UPLOAD
            bucket = "upload"
        else:
            return await call_next(request)
        key = (request.client.host if request.client else "unknown", bucket)
        now = time.monotonic()
        with _rate_lock:
            recent = [stamp for stamp in _rate_hits.get(key, []) if now - stamp < AI_RATE_LIMIT_WINDOW_SECONDS]
            if len(recent) >= limit:
                return JSONResponse({"detail": {"code": "rate_limited", "message": "Too many requests"}}, status_code=429, headers={"Retry-After": str(AI_RATE_LIMIT_WINDOW_SECONDS)})
            recent.append(now)
            _rate_hits[key] = recent
        return await call_next(request)
ai_store = AIStore(AI_DB_FILE)
if ENABLE_TEST_ADMIN:
    ai_store.provision_test_admin(ADMIN_EMAIL, ADMIN_PASSWORD)
ai_orchestrator = AIOrchestrator()
ai_orchestrator.registry.approvals.attach_store(ai_store)
approval_workflow = ApprovalWorkflow(ai_store)


def _record_workflow_stage(workspace_id: str, stage_id: str, stage_name: str, status: str,
                           output: dict | None = None, error: str | None = None) -> None:
    try:
        workspace = services.get_workspace(workspace_id)
        artifact_references = [output[key] for key in ("report_url", "download_url") if isinstance(output, dict) and output.get(key)]
        ai_store.record_event(OPEN_ACCESS_USER_ID if OPEN_ACCESS_MODE_ENABLED else "workflow-system", workspace_id,
            "workflow_stage", {"workspace_id": workspace_id, "dataset_id": workspace["dataset_id"],
             "workflow_id": workspace_id, "stage_id": stage_id, "stage_name": stage_name,
             "execution_status": status, "output_data": output, "execution_timestamp": ai_store._now(),
             "configuration_version": hashlib.sha256(json.dumps({k: workspace.get(k) for k in ("target", "problem_type", "preprocessing", "model_id", "model_params")}, sort_keys=True, default=str).encode()).hexdigest()[:16],
             "artifact_references": artifact_references, "error_information": error})
    except (ValueError, OSError):
        logging.getLogger(__name__).exception("Could not persist workflow stage event")


def _workflow_output(workspace_id: str, stage_id: str, stage_name: str, status: str, output: dict | None = None,
                     configuration: dict | None = None, warnings: list[str] | None = None, errors: list[str] | None = None) -> dict:
    workspace = services.get_workspace(workspace_id)
    dataset = services.get_dataset(workspace["dataset_id"])
    config = configuration or {key: workspace.get(key) for key in ("target", "problem_type", "preprocessing", "model_id", "model_params")}
    artifacts = [value for key, value in (output or {}).items() if key in {"report_url", "download_url"} and value]
    duration = (output or {}).get("training_duration_seconds")
    return WorkflowOutput(success=status not in {"failed"}, workflow_id=workspace_id, workspace_id=workspace_id,
        dataset_id=workspace["dataset_id"], stage_id=stage_id, stage_name=stage_name, execution_id=(output or {}).get("run_id"),
        execution_status=status, applied_configuration=config, output_summary={"keys": sorted((output or {}).keys())},
        output_data=output, artifact_references=artifacts, warnings=warnings or [], errors=errors or [],
        execution_metrics=ExecutionMetrics(duration_seconds=duration),
        resource_usage=ResourceUsage(dataset_bytes=None, estimate_is_approximate=True),
        completed_at=datetime.now(timezone.utc), configuration_fingerprint=hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest(),
        dataset_version=dataset.get("version", 1), stage_version="1").model_dump(mode="json")


def _workspace_ai_facts(workspace_id: str) -> dict:
    facts = build_workflow_context(workspace_id)
    facts["recent_workflow_events"] = [
        {"stage_id": event["payload"].get("stage_id", event["event_type"]),
         "stage_name": event["payload"].get("stage_name", event["event_type"]),
         "execution_status": event["payload"].get("execution_status", event["payload"].get("status", "completed")),
         "execution_timestamp": event["created_at"],
         "output_data": event["payload"].get("output_data", event["payload"].get("output")),
         "error_information": event["payload"].get("error_information", event["payload"].get("error_message"))}
        for event in ai_store.list_events(workspace_id, limit=20)
    ]
    return facts


class AIIntentRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    workspace_id: str | None = None


class TrainingApprovalRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    configuration: dict
    approved_by: str = Field(min_length=1, max_length=200)


class WorkflowProposalRequest(BaseModel):
    workspace_id: str = Field(min_length=1)
    stage_id: Literal["model_configuration", "training", "optimization", "column_selection"]
    configuration: dict
    conversation_id: str | None = None
    rationale: str = Field(default="AI-proposed configuration", max_length=4000)
    expected_outcome: str = Field(default="Execute the selected workflow stage", max_length=2000)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class WorkflowProposalDecisionRequest(BaseModel):
    decision: Literal["execute", "cancel", "fill_only"]
    configuration_fingerprint: str = Field(min_length=64, max_length=64)


class ConversationRequest(BaseModel):
    workspace_id: str | None = None


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=12000)
    mode: Literal["manual", "auto"] = "manual"


class MemorySettingsRequest(BaseModel):
    enabled: bool


class MemoryUpdateRequest(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=1000)
    confidence: float | None = Field(default=None, ge=0, le=1)
    importance: int | None = Field(default=None, ge=1, le=5)

class AdminLoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=512)


def current_principal(authorization: str | None = Header(default=None)) -> Principal:
    # This identity is fixed by backend configuration. Client supplied tokens or
    # identity claims are deliberately ignored while local open access is active.
    if OPEN_ACCESS_MODE_ENABLED:
        return Principal(user_id=OPEN_ACCESS_USER_ID, expires_at=2**31, role="super_admin")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail={"code": "authentication_required", "message": "Bearer authentication is required"})
    try:
        token = authorization[7:].strip()
        if token.count(".") == 2:
            return verify_google_id_token(token, GOOGLE_CLIENT_ID)
        return verify_bearer_token(token)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail={"code": "invalid_authentication", "message": str(exc)}) from exc

def require_super_admin(principal: Principal = Depends(current_principal)) -> Principal:
    if principal.role != "super_admin":
        raise HTTPException(status_code=403, detail={"code": "admin_required", "message": "Super Admin role is required"})
    return principal

@app.post("/api/auth/test-admin/login")
def test_admin_login(req: AdminLoginRequest):
    if not ENABLE_TEST_ADMIN or not ai_store.verify_admin(req.email, req.password):
        raise HTTPException(status_code=401, detail={"code": "invalid_credentials", "message": "Invalid credentials"})
    return {"access_token": issue_test_token(req.email.lower(), role="super_admin"), "token_type": "bearer", "role": "super_admin"}

@app.get("/api/auth/me")
def auth_me(principal: Principal = Depends(current_principal)):
    return {"user_id": principal.user_id, "role": principal.role, "expires_at": principal.expires_at}


@app.get("/api/auth/config")
def ai_auth_config():
    return {"google_sign_in_enabled": bool(GOOGLE_CLIENT_ID), "google_client_id": GOOGLE_CLIENT_ID, "open_access_mode": OPEN_ACCESS_MODE_ENABLED}


def require_workspace_owner(workspace_id: str, principal: Principal) -> None:
    if not services.get_workspace(workspace_id):
        raise HTTPException(status_code=404, detail={"code": "workspace_not_found", "message": "Workspace not found"})
    if principal.role == "super_admin":
        return
    if OPEN_ACCESS_MODE_ENABLED:
        return
    if not ai_store.owns_workspace(workspace_id, principal.user_id):
        raise HTTPException(status_code=403, detail={"code": "workspace_forbidden", "message": "Workspace is not owned by this user"})


@app.post("/api/ai/workspaces/{workspace_id}/claim")
def claim_ai_workspace(workspace_id: str, principal: Principal = Depends(current_principal)):
    try:
        services.get_workspace(workspace_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail={"code": "workspace_not_found", "message": str(exc)}) from exc
    if not ai_store.claim_workspace(workspace_id, principal.user_id):
        raise HTTPException(status_code=403, detail={"code": "workspace_forbidden", "message": "Workspace is already owned by another user"})
    return {"workspace_id": workspace_id, "user_id": principal.user_id, "claimed": True}


@app.post("/api/ai/intent")
def ai_intent(req: AIIntentRequest, principal: Principal = Depends(current_principal)):
    context = ContextBundle(workspace_id=req.workspace_id)
    intent = ai_orchestrator.understand_intent(req.message, context)
    try:
        workspace = services.get_workspace(req.workspace_id) if req.workspace_id else None
    except ValueError as exc:
        _err("workspace_not_found", str(exc), 404)
    if req.workspace_id:
        require_workspace_owner(req.workspace_id, principal)
    plan = ai_orchestrator.plan(intent, workspace)
    return {"intent": intent.model_dump(mode="json"), "plan": plan.model_dump(mode="json"), "response": ai_orchestrator.respond_without_provider(intent, plan)}


@app.post("/api/ai/approvals/training")
def create_training_approval(req: TrainingApprovalRequest, principal: Principal = Depends(current_principal)):
    require_workspace_owner(req.workspace_id, principal)
    if req.approved_by != principal.user_id:
        raise HTTPException(status_code=403, detail={"code": "approval_forbidden", "message": "Approval identity must match the authenticated user"})
    approval = ai_orchestrator.registry.approvals.issue(req.workspace_id, req.configuration, req.approved_by)
    return approval.model_dump(mode="json")


@app.post("/api/ai/workflow-proposals")
def create_workflow_proposal(req: WorkflowProposalRequest, principal: Principal = Depends(current_principal)):
    require_workspace_owner(req.workspace_id, principal)
    try:
        proposal = approval_workflow.create(user_id=principal.user_id, workspace_id=req.workspace_id, stage_id=req.stage_id,
            configuration=req.configuration, conversation_id=req.conversation_id, rationale=req.rationale,
            expected_outcome=req.expected_outcome, warnings=req.warnings)
        ai_store.record_event(principal.user_id, req.workspace_id, "workflow_proposal", proposal)
        return {"success": True, "proposal": proposal}
    except (ProposalError, ValueError) as exc:
        _err("PROPOSAL_ERROR", str(exc))


@app.get("/api/ai/workflow-proposals/{proposal_id}")
def get_workflow_proposal(proposal_id: str, principal: Principal = Depends(current_principal)):
    try:
        return {"success": True, "proposal": approval_workflow.get(proposal_id, principal.user_id)}
    except ProposalError as exc:
        _err("PROPOSAL_NOT_FOUND", str(exc), 404)


@app.get("/api/ai/workflow-proposals")
def list_workflow_proposals(workspace_id: str, actionable_only: bool = True, principal: Principal = Depends(current_principal)):
    require_workspace_owner(workspace_id, principal)
    return {"success": True, "proposals": approval_workflow.list(workspace_id, principal.user_id, actionable_only=actionable_only)}


@app.post("/api/ai/workflow-proposals/{proposal_id}/decision")
def decide_workflow_proposal(proposal_id: str, req: WorkflowProposalDecisionRequest, principal: Principal = Depends(current_principal)):
    try:
        before = approval_workflow.get(proposal_id, principal.user_id)
        require_workspace_owner(before["workspace_id"], principal)
        proposal = approval_workflow.decide(proposal_id, principal.user_id, req.decision, req.configuration_fingerprint)
        ai_store.record_event(principal.user_id, proposal["workspace_id"], "workflow_proposal_decision", proposal)
        stage_status = "completed" if proposal["status"] == "completed" else proposal["status"]
        if req.decision == "execute":
            _record_workflow_stage(proposal["workspace_id"], proposal["stage_id"], proposal["stage_id"].replace("_", " ").title(), stage_status,
                                   (proposal.get("execution") or {}).get("output"), (proposal.get("execution") or {}).get("error"))
        return {"success": True, "proposal": proposal}
    except ProposalError as exc:
        _err("PROPOSAL_DECISION_ERROR", str(exc), 409)


@app.post("/api/ai/tools/execute")
def execute_ai_tool(req: ToolRequest, principal: Principal = Depends(current_principal)):
    workspace_id = req.arguments.get("workspace_id")
    if workspace_id:
        require_workspace_owner(workspace_id, principal)
    previous = ai_store.get_tool_execution(principal.user_id, req.idempotency_key)
    if previous:
        return previous["result"] | {"status": "duplicate"}
    result = ai_orchestrator.registry.execute(req)
    ai_store.record_tool_execution(principal.user_id, workspace_id, req.idempotency_key, req.tool_name, result.status, result.model_dump(mode="json"))
    return result.model_dump(mode="json")


@app.post("/api/ai/conversations")
def create_ai_conversation(req: ConversationRequest, principal: Principal = Depends(current_principal)):
    if req.workspace_id:
        require_workspace_owner(req.workspace_id, principal)
    return ai_store.create_conversation(principal.user_id, req.workspace_id)


@app.post("/api/ai/conversations/{conversation_id}/messages")
def add_ai_message(conversation_id: str, req: MessageRequest, principal: Principal = Depends(current_principal)):
    try:
        message = (ai_store.add_message_as_admin(conversation_id, principal.user_id, "user", req.content)
                   if principal.role == "super_admin" else ai_store.add_message(conversation_id, principal.user_id, "user", req.content))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"code": "conversation_forbidden", "message": str(exc)}) from exc
    initial_context = ai_store.context_as_admin(conversation_id) if principal.role == "super_admin" else ai_store.context(conversation_id, principal.user_id)
    memory_user_id = initial_context["conversation"]["user_id"]
    conversation_workspace_id = initial_context["conversation"]["workspace_id"]
    for candidate in extract_durable_memories(req.content):
        if candidate["scope_type"] == "project" and not conversation_workspace_id:
            continue
        ai_store.save_memory(memory_user_id, candidate["category"], candidate["memory_key"], candidate["content"],
            scope_type=candidate["scope_type"], workspace_id=conversation_workspace_id if candidate["scope_type"] == "project" else None,
            confidence=candidate["confidence"], importance=candidate["importance"], source_message_id=message["id"],
            source_conversation_id=conversation_id, review_policy=candidate["review_policy"])
    ai_store.summarize_conversation(conversation_id, memory_user_id)
    context = ai_store.context_as_admin(conversation_id) if principal.role == "super_admin" else ai_store.context(conversation_id, principal.user_id)
    try:
        workspace_id = context["conversation"]["workspace_id"]
        facts = _workspace_ai_facts(workspace_id) if workspace_id else {}
        relevant_memories = ai_store.retrieve_relevant_memories(memory_user_id, req.content,
            workspace_id=workspace_id, conversation_id=conversation_id, limit=8)
        memory_evidence = [{k: memory[k] for k in ("scope_type", "category", "content", "confidence", "importance", "relevance_score", "why_retrieved")} for memory in relevant_memories]
        facts["relevant_memories"] = memory_evidence
        if context.get("summary"):
            facts["conversation_summary"] = context["summary"]["summary"]
        conversation_summary = (context.get("summary") or {}).get("summary")
        turns = [ConversationTurn(role=entry["role"], content=entry["content"]) for entry in context["messages"][:-1] if entry["role"] in {"user", "assistant"}]
        if req.mode == "auto":
            if facts:
                facts["auto_mode"] = {"enabled": True, "policy": "bounded, one validated allowlisted tool at a time", "max_steps": 8}
            execution_history = []
            seen_actions: set[str] = set()
            response = {}
            for step in range(8):
                facts["recent_tool_results"] = execution_history[-8:]
                response = ai_orchestrator.provider_response(
                    req.content,
                    ContextBundle(workspace_id=workspace_id, turns=turns, memories=[f"[{m['scope_type']}:{m['category']}] {m['content']}" for m in relevant_memories], facts=facts),
                    auto_mode=True,
                    on_text_chunk=_ai_text_chunk_callback(),
                )
                tool_name = response.get("tool_name")
                if not tool_name:
                    break
                tool_args = dict(response.get("tool_arguments") or {})
                target_workspace = tool_args.get("workspace_id") or workspace_id
                if target_workspace:
                    require_workspace_owner(target_workspace, principal)
                    tool_args.setdefault("workspace_id", target_workspace)
                action_key = json.dumps([tool_name, tool_args], sort_keys=True, default=str)
                if action_key in seen_actions:
                    response["response"] += " I stopped because the same operation was requested again without new progress."
                    break
                seen_actions.add(action_key)
                if response.get("needs_approval"):
                    response["response"] += " This action requires approval before it can run."
                    break
                execution_id = hashlib.sha256(f"{conversation_id}:{message['id']}:{step}:{action_key}".encode()).hexdigest()
                tool_request = ToolRequest(tool_name=tool_name, arguments=tool_args, idempotency_key=execution_id)
                stage_meta = {
                    "analyze_dataset": ("eda", "Exploratory data analysis"),
                    "configure_workspace": ("model_configuration", "Model configuration"),
                    "train_workspace": ("training", "Model training"),
                    "optimize_workspace": ("optimization", "Hyperparameter optimization"),
                    "predict_workspace": ("prediction", "Prediction"),
                    "generate_profiling": ("profiling", "Dataset profiling"),
                }.get(tool_name)
                _emit_ai_stream_event({"type": "tool", "status": "started", "tool_name": tool_name})
                if target_workspace and stage_meta:
                    _record_workflow_stage(target_workspace, stage_meta[0], stage_meta[1], "running", tool_args)
                execution = ai_orchestrator.registry.execute(tool_request).model_dump(mode="json")
                _emit_ai_stream_event({"type": "tool", "status": execution["status"], "tool_name": tool_name})
                execution_history.append({"tool_name": tool_name, "arguments": tool_args, "result": execution})
                ai_store.record_tool_execution(principal.user_id, target_workspace, execution_id, tool_name, execution["status"], execution)
                if target_workspace:
                    ai_store.record_event(principal.user_id, target_workspace, "tool_execution", execution)
                    if stage_meta:
                        _record_workflow_stage(target_workspace, stage_meta[0], stage_meta[1], execution["status"], execution.get("output"), execution.get("error_message"))
                if workspace_id:
                    facts = _workspace_ai_facts(workspace_id)
                    facts["relevant_memories"] = memory_evidence
                    if conversation_summary:
                        facts["conversation_summary"] = conversation_summary
                    facts["auto_mode"] = {"enabled": True, "policy": "bounded, one validated allowlisted tool at a time", "max_steps": 8}
        else:
            response = ai_orchestrator.provider_response(
                req.content,
                ContextBundle(workspace_id=workspace_id, turns=turns, memories=[f"[{m['scope_type']}:{m['category']}] {m['content']}" for m in relevant_memories], facts=facts),
                on_text_chunk=_ai_text_chunk_callback(),
            )
    except ValueError as exc:
        raise HTTPException(status_code=502, detail={"code": "provider_contract_error", "message": str(exc)}) from exc
    except (ProviderRequestError, ProviderConfigurationError) as exc:
        raise HTTPException(status_code=502, detail={"code": "provider_unavailable", "message": str(exc)}) from exc
    tool_evidence = None
    tool_name = response.get("tool_name")
    tool_args = dict(response.get("tool_arguments") or {})
    workspace_id = tool_args.get("workspace_id") or context["conversation"]["workspace_id"]
    if req.mode != "auto" and tool_name and workspace_id:
        require_workspace_owner(workspace_id, principal)
        tool_args.setdefault("workspace_id", workspace_id)
        if not response.get("needs_approval"):
            tool_request = ToolRequest(tool_name=tool_name, arguments=tool_args, idempotency_key=hashlib.sha256(f"{conversation_id}:{message['id']}".encode()).hexdigest())
            previous = ai_store.get_tool_execution(principal.user_id, tool_request.idempotency_key)
            if previous:
                tool_evidence = previous["result"]
            else:
                _emit_ai_stream_event({"type": "tool", "status": "started", "tool_name": tool_name})
                execution = ai_orchestrator.registry.execute(tool_request)
                tool_evidence = execution.model_dump(mode="json")
                _emit_ai_stream_event({"type": "tool", "status": execution.status, "tool_name": tool_name})
                ai_store.record_tool_execution(principal.user_id, workspace_id, tool_request.idempotency_key, tool_name, execution.status, tool_evidence)
                ai_store.record_event(principal.user_id, workspace_id, "tool_execution", tool_evidence)
        else:
            response["response"] = response["response"] + " Explicit approval is required before this action can run."
            stage_id = {"configure_workspace": "model_configuration", "train_workspace": "training",
                        "optimize_workspace": "optimization", "apply_column_selection": "column_selection"}.get(tool_name)
            if stage_id:
                proposal_config = {key: value for key, value in tool_args.items() if key != "workspace_id"}
                if stage_id in {"training", "optimization"} and not proposal_config:
                    current = services.get_workspace(workspace_id)
                    proposal_config = {"configuration": {key: current.get(key) for key in ("target", "problem_type", "preprocessing", "model_id", "model_params")}}
                try:
                    response["proposal"] = approval_workflow.create(user_id=principal.user_id, workspace_id=workspace_id,
                        stage_id=stage_id, configuration=proposal_config, conversation_id=conversation_id,
                        rationale="Configuration selected by the assistant from the current workflow context",
                        expected_outcome=f"Run {stage_id.replace('_', ' ')} with the displayed configuration")
                    ai_store.record_event(principal.user_id, workspace_id, "workflow_proposal", response["proposal"])
                except (ProposalError, ValueError) as exc:
                    response["response"] += f" The proposed configuration could not be prepared: {exc}."
    if tool_evidence:
        response["tool_evidence"] = tool_evidence
    if req.mode == "auto":
        response["execution_history"] = execution_history
    assistant = (ai_store.add_message_as_admin(conversation_id, principal.user_id, "assistant", response["response"], evidence={"structured": response})
                 if principal.role == "super_admin" else ai_store.add_message(conversation_id, principal.user_id, "assistant", response["response"], evidence={"structured": response}))
    return {"message": message, "assistant": assistant, "structured": response}


@app.post("/api/ai/conversations/{conversation_id}/messages/stream")
def add_ai_message_stream(conversation_id: str, req: MessageRequest, principal: Principal = Depends(current_principal)):
    events: queue.Queue = queue.Queue()
    disconnected = Event()
    streamed_chunks: list[str] = []

    def publish(event: dict) -> None:
        if disconnected.is_set():
            raise RuntimeError("AI response stream was disconnected")
        if event.get("type") == "delta":
            streamed_chunks.append(event.get("text", ""))
        events.put(event)

    def process_message() -> None:
        token = _ai_stream_events.set(publish)
        try:
            result = add_ai_message(conversation_id, req, principal)
            events.put({
                "type": "complete",
                "assistant": result["assistant"],
                "proposal": result["structured"].get("proposal"),
            })
        except HTTPException as exc:
            detail = exc.detail
            message = detail.get("message", "The assistant request failed.") if isinstance(detail, dict) else str(detail)
            if not disconnected.is_set() and exc.status_code >= 500:
                try:
                    owner = ai_store.conversation_owner(conversation_id)
                    if owner:
                        context = ai_store.context_as_admin(conversation_id)
                        latest = context["messages"][-1] if context["messages"] else None
                        if latest and latest["role"] == "user" and latest["content"] == req.content:
                            ai_store.add_message(conversation_id, owner, "assistant", "I couldn't complete that response. Please retry.", evidence={"stream_error": True})
                except Exception:
                    logging.getLogger(__name__).exception("Could not persist failed streaming response")
            events.put({"type": "error", "message": message})
        except Exception:
            logging.getLogger(__name__).exception("Streaming AI response failed for conversation %s", conversation_id)
            if disconnected.is_set() and streamed_chunks:
                try:
                    owner = ai_store.conversation_owner(conversation_id)
                    if owner:
                        partial = "".join(streamed_chunks)
                        ai_store.add_message(conversation_id, owner, "assistant", partial, evidence={"stream_interrupted": True})
                except Exception:
                    logging.getLogger(__name__).exception("Could not persist interrupted AI response")
            elif not disconnected.is_set():
                events.put({"type": "error", "message": "The assistant could not complete this response. Please retry."})
        finally:
            _ai_stream_events.reset(token)
            events.put(None)

    Thread(target=process_message, daemon=True, name=f"ai-stream-{conversation_id[:8]}").start()

    def stream_events():
        try:
            while True:
                event = events.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            disconnected.set()

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ai/conversations/{conversation_id}/context")
def get_ai_context(conversation_id: str, principal: Principal = Depends(current_principal)):
    try:
        return ai_store.context_as_admin(conversation_id) if principal.role == "super_admin" else ai_store.context(conversation_id, principal.user_id)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail={"code": "conversation_forbidden", "message": str(exc)}) from exc


@app.get("/api/ai/memory-settings")
def get_ai_memory_settings(principal: Principal = Depends(current_principal)):
    return ai_store.memory_settings(principal.user_id)


@app.put("/api/ai/memory-settings")
def put_ai_memory_settings(req: MemorySettingsRequest, principal: Principal = Depends(current_principal)):
    return ai_store.set_memory_enabled(principal.user_id, req.enabled)


@app.get("/api/ai/memories")
def get_ai_memories(workspace_id: str | None = None, scope: str | None = None,
                    principal: Principal = Depends(current_principal)):
    if workspace_id:
        require_workspace_owner(workspace_id, principal)
        scope = "project"
    if scope not in {None, "user", "project", "conversation"}:
        raise HTTPException(status_code=422, detail={"code": "invalid_memory_scope", "message": "Unknown memory scope"})
    return {"memories": ai_store.list_memories(principal.user_id, workspace_id=workspace_id, scope_type=scope),
            "settings": ai_store.memory_settings(principal.user_id)}


@app.patch("/api/ai/memories/{memory_id}")
def patch_ai_memory(memory_id: str, req: MemoryUpdateRequest, principal: Principal = Depends(current_principal)):
    existing = ai_store.get_memory(principal.user_id, memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail={"code": "memory_not_found", "message": "Memory not found"})
    if existing["scope_type"] == "project":
        require_workspace_owner(existing["workspace_id"], principal)
    updated = ai_store.update_memory(principal.user_id, memory_id, content=req.content, confidence=req.confidence, importance=req.importance)
    return {"memory": updated}


@app.delete("/api/ai/memories/{memory_id}")
def delete_ai_memory(memory_id: str, principal: Principal = Depends(current_principal)):
    existing = ai_store.get_memory(principal.user_id, memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail={"code": "memory_not_found", "message": "Memory not found"})
    if existing["scope_type"] == "project":
        require_workspace_owner(existing["workspace_id"], principal)
    ai_store.delete_memory(principal.user_id, memory_id)
    return {"deleted": True, "memory_id": memory_id}


@app.delete("/api/ai/memories")
def clear_ai_memories(workspace_id: str | None = None, principal: Principal = Depends(current_principal)):
    if workspace_id:
        require_workspace_owner(workspace_id, principal)
        deleted = ai_store.clear_memories(principal.user_id, workspace_id=workspace_id)
        scope = "project"
    else:
        deleted = ai_store.clear_memories(principal.user_id, scope_type="user")
        scope = "user"
    return {"deleted_count": deleted, "scope": scope}


@app.delete("/api/ai/conversations/{conversation_id}/memory")
def clear_ai_conversation_memory(conversation_id: str, principal: Principal = Depends(current_principal)):
    owner_id = ai_store.conversation_owner(conversation_id)
    if owner_id is None or (owner_id != principal.user_id and principal.role != "super_admin"):
        owner_id = None
    if owner_id is None or not ai_store.clear_conversation_memory(conversation_id, owner_id):
        raise HTTPException(status_code=404, detail={"code": "conversation_not_found", "message": "Conversation not found"})
    return {"cleared": True, "conversation_id": conversation_id}


@app.get("/api/workspaces/{workspace_id}/ai-context")
def get_workspace_ai_context(workspace_id: str, principal: Principal = Depends(current_principal)):
    require_workspace_owner(workspace_id, principal)
    return _workspace_ai_facts(workspace_id)


@app.get("/api/ai/tools")
def get_ai_tool_inventory(_principal: Principal = Depends(current_principal)):
    return {"tools": ai_orchestrator.registry.inventory(), "memory_operations": [
        "retrieve_user_memory", "retrieve_project_memory", "retrieve_conversation_context",
        "retrieve_workflow_history", "save_user_memory", "update_memory", "search_memory",
        "retrieve_relevant_memories", "summarize_conversation", "invalidate_stale_memory",
    ]}


@app.get("/api/workspaces/{workspace_id}/events")
def get_workspace_events(workspace_id: str, principal: Principal = Depends(current_principal)):
    require_workspace_owner(workspace_id, principal)
    workspace = services.get_workspace(workspace_id)
    return {"workspace_status": workspace.get("status"), "events": ai_store.list_events(workspace_id)}
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Authorization", "Content-Type"], allow_credentials=True)


def _err(code: str, msg: str, status: int = 400):
    raise HTTPException(status_code=status, detail={"code": code, "message": msg})


# ──────────────────────── Health ────────────────────────

@app.post("/api/data-playground/import-ml/{ml_dataset_id}")
def playground_import_ml_dataset(ml_dataset_id: str, principal: Principal = Depends(current_principal)):
    """Import an active ML dataset into Data Playground so the user can inspect it seamlessly."""
    try:
        import pandas as pd
        ds = services.get_dataset(ml_dataset_id)
        filepath = Path(ds["filepath"])
        if not filepath.is_absolute():
            filepath = Path.cwd() / filepath
        if not filepath.exists():
            _err("DATASET_FILE_NOT_FOUND", "ML dataset file could not be found.", 404)
        frame = pd.read_csv(filepath)
        item = playground.import_dataframe(ds.get("filename", "ml_dataset.csv"), frame)
        if principal.role != "super_admin" and not OPEN_ACCESS_MODE_ENABLED:
            ai_store.set_preference(principal.user_id, f"playground_owner:{item['id']}", "claimed")
        return item
    except Exception as exc:
        _err("PLAYGROUND_IMPORT_FAILED", str(exc), 422)


@app.get("/api/health")
def health():
    return {"status": "healthy"}


def _playground_owner(dataset_id: str, principal: Principal) -> dict:
    item = playground.dataset(dataset_id)
    if principal.role != "super_admin" and not OPEN_ACCESS_MODE_ENABLED:
        # Data Playground's current metadata schema has no ownership field. Do
        # not authorize by ID alone; require an explicit server-side claim.
        if ai_store.preference(principal.user_id, f"playground_owner:{dataset_id}") != "claimed":
            raise HTTPException(status_code=403, detail={"code": "playground_dataset_forbidden", "message": "Data Playground dataset has no ownership claim for this user."})
    return item


@app.post("/api/data-playground/datasets/{dataset_id}/claim")
def claim_playground_dataset(dataset_id: str, principal: Principal = Depends(current_principal)):
    playground.dataset(dataset_id)
    if principal.role != "super_admin" and not OPEN_ACCESS_MODE_ENABLED:
        if ai_store.preference(principal.user_id, f"playground_owner:{dataset_id}") not in {None, "claimed"}:
            raise HTTPException(status_code=403, detail={"code": "playground_dataset_forbidden", "message": "Dataset is already claimed by another user."})
        ai_store.set_preference(principal.user_id, f"playground_owner:{dataset_id}", "claimed")
    return {"dataset_id": dataset_id, "claimed": True}


@app.get("/api/data-playground/registry")
def playground_registry(principal: Principal = Depends(current_principal)):
    return playground.registry()


@app.post("/api/data-playground/datasets")
async def playground_upload(file: UploadFile = File(...), principal: Principal = Depends(current_principal)):
    try:
        service = playground._load_service()
        item = await service.upload(file)
        if principal.role != "super_admin" and not OPEN_ACCESS_MODE_ENABLED:
            ai_store.set_preference(principal.user_id, f"playground_owner:{item['id']}", "claimed")
        return item
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        _err("PLAYGROUND_UPLOAD_FAILED", str(exc), 422)


@app.delete("/api/data-playground/datasets/{dataset_id}")
def playground_delete(dataset_id: str, principal: Principal = Depends(current_principal)):
    """Delete a Playground dataset through the same ownership boundary as all reads."""
    _playground_owner(dataset_id, principal)
    return playground._load_service().delete(dataset_id, confirm=True)


@app.post("/api/data-playground/datasets/{dataset_id}/analysis")
def playground_analysis(dataset_id: str, payload: dict = Body(...), principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    service = playground._load_service()
    request = service.AnalysisRequest(**payload)
    return service.analysis(dataset_id, request)


@app.post("/api/data-playground/datasets/{dataset_id}/visualizations")
def playground_visualization(dataset_id: str, payload: dict = Body(...), principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    service = playground._load_service()
    request = service.VisualizationRequest(**payload)
    return service.visualization(dataset_id, request)


@app.post("/api/data-playground/datasets/{dataset_id}/ai")
def playground_ai(dataset_id: str, payload: dict = Body(...), principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    if not payload.get("content"):
        _err("EMPTY_MESSAGE", "Message content is required.", 422)
    version = int(payload.get("version", 1))
    service = playground._load_service()
    # Build fresh, version-scoped context on every turn. Prior analyses are
    # loaded only for the active version, so switching versions cannot leak stale results.
    context = service.ai_context(dataset_id, version)
    context["recent_analysis"] = service.analysis_history(dataset_id, limit=50).get("items", [])
    context["recent_analysis"] = [item for item in context["recent_analysis"] if int(item.get("version", -1)) == version]
    context["current_page"] = str(payload.get("page", ""))
    context["dataset_profile"] = playground.schema(dataset_id, version)
    context["user_question"] = str(payload["content"])
    return playground.ai_message(dataset_id, version, str(payload["content"]), str(payload.get("page", "")), context=context)


@app.post("/api/data-playground/datasets/{dataset_id}/transformations/{action}")
def playground_transformation(dataset_id: str, action: Literal["preview", "approval", "apply"], payload: dict = Body(...), confirm: bool = False, principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    service = playground._load_service()
    request = service.TransformRequest(**payload)
    if action == "preview": return service.transformation_preview(dataset_id, request)
    if action == "approval": return service.transformation_approval(dataset_id, request)
    return service.transformation_apply(dataset_id, request, confirm=True)


@app.get("/api/data-playground/datasets/{dataset_id}/summary")
def playground_summary(dataset_id: str, version: int = Query(1, ge=1), principal: Principal = Depends(current_principal)):
    item = _playground_owner(dataset_id, principal)
    detail = playground.schema(dataset_id, version)
    history = playground.transformations(dataset_id).get("items", [])
    original = playground.schema(dataset_id, 1)
    original_names = {c["name"] for c in original["columns"]}
    current_names = {c["name"] for c in detail["columns"]}
    return {"dataset": {"id": item["id"], "name": item["name"], "created": item["created"]},
            "version": version, "version_id": next((v["version_id"] for v in item["versions"] if v["version"] == version), None),
            "row_count": detail["row_count"], "column_count": detail["column_count"],
            "duplicate_row_count": detail["duplicate_row_count"], "columns": detail["columns"],
            "versions": item["versions"], "transformations": history,
            "original": original,
            "comparison": {"row_delta": detail["row_count"] - original["row_count"],
                           "column_delta": detail["column_count"] - original["column_count"],
                           "added_columns": sorted(current_names - original_names),
                           "removed_columns": sorted(original_names - current_names),
                           "missing_values_before": sum(c["missing_count"] for c in original["columns"]),
                           "missing_values_after": sum(c["missing_count"] for c in detail["columns"]),
                           "duplicates_before": original["duplicate_row_count"],
                           "duplicates_after": detail["duplicate_row_count"]}}


@app.get("/api/data-playground/datasets/{dataset_id}/rows")
def playground_rows(dataset_id: str, version: int = Query(1, ge=1), offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), search: str | None = None, principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    return playground.rows(dataset_id, version=version, offset=offset, limit=limit, search=search)


@app.get("/api/data-playground/datasets/{dataset_id}/exports/{fmt}")
def playground_export(dataset_id: str, fmt: Literal["csv", "xlsx", "parquet"], version: int = Query(1, ge=1), principal: Principal = Depends(current_principal)):
    _playground_owner(dataset_id, principal)
    frame = playground.export_frame(dataset_id, version)
    suffix = fmt
    temp = tempfile.NamedTemporaryFile(prefix="playground-export-", suffix=f".{suffix}", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    if fmt == "csv":
        frame.to_csv(temp_path, index=False, encoding="utf-8-sig")
        media = "text/csv; charset=utf-8"
    elif fmt == "xlsx":
        frame.to_excel(temp_path, index=False, engine="openpyxl")
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    else:
        frame.to_parquet(temp_path, index=False)
        media = "application/vnd.apache.parquet"
    return FileResponse(temp_path, media_type=media, filename=f"dataset-v{version}.{suffix}", background=__import__("starlette").background.BackgroundTask(temp_path.unlink, missing_ok=True))


@app.get("/api/data-playground/datasets/{dataset_id}/report")
def playground_report(dataset_id: str, version: int = Query(1, ge=1), principal: Principal = Depends(current_principal)):
    summary = playground_summary(dataset_id, version, principal)
    return {"report_type": "data_preparation", "exported_at": datetime.now(timezone.utc).isoformat(),
            "original": playground.schema(dataset_id, 1), "final": playground.schema(dataset_id, version),
            "dataset": summary["dataset"], "version": version, "transformations": summary["transformations"]}


@app.post("/api/data-playground/handoff")
def playground_handoff(req: PlaygroundHandoffRequest, principal: Principal = Depends(current_principal)):
    item = _playground_owner(req.source_dataset_id, principal)
    version_id, frame, _ = playground.create_ml_materialization(req.source_dataset_id, req.source_version)
    handoff_id = uuid.uuid4().hex
    lineage = {"source": "data_playground", "source_dataset_id": req.source_dataset_id,
               "source_version": req.source_version, "source_version_id": version_id,
               "source_name": item["name"], "handoff_id": handoff_id,
               "transformations": playground.transformations(req.source_dataset_id).get("items", []),
               "handoff_user_id": principal.user_id, "handoff_timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        dataset = services.import_playground_dataset(item["name"], frame, lineage)
        workspace = services.create_workspace(dataset["id"])
    except ValueError as exc:
        _err("HANDOFF_FAILED", str(exc), 422)
    ai_store.claim_workspace(workspace["id"], principal.user_id)
    return {"handoff_id": handoff_id, "source_dataset_id": req.source_dataset_id,
            "source_version": req.source_version, "source_version_id": version_id,
            "ml_dataset": dataset, "ml_workspace": workspace, "status": "completed"}


# ──────────────────────── Datasets ────────────────────────

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = services.upload_dataset(file.filename or "upload.csv", content)
        return {"success": True, **result}
    except ValueError as e:
        _err("UPLOAD_ERROR", str(e))


@app.get("/api/datasets/{dataset_id}")
def get_dataset(dataset_id: str):
    try:
        return {"success": True, **services.get_dataset(dataset_id)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview(dataset_id: str, rows: int = Query(default=20, ge=1, le=200)):
    try:
        return {"success": True, **services.get_preview(dataset_id, rows)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/datasets/{dataset_id}/analysis")
def dataset_analysis(dataset_id: str):
    try:
        result = services.get_analysis(dataset_id)
        for workspace_id in services.workspaces_for_dataset(dataset_id):
            _record_workflow_stage(workspace_id, "eda", "Exploratory data analysis", "completed", result)
        return {"success": True, **result}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/datasets/{dataset_id}/columns")
def dataset_columns(dataset_id: str, target: str | None = None):
    try:
        return {"success": True, **services.get_column_statistics(dataset_id, target)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/datasets/{dataset_id}/correlation")
def dataset_correlation(dataset_id: str):
    try:
        result = services.get_correlation(dataset_id)
        for workspace_id in services.workspaces_for_dataset(dataset_id):
            _record_workflow_stage(workspace_id, "eda_correlation", "Correlation analysis", "completed", result)
        return {"success": True, **result}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.post("/api/datasets/{dataset_id}/profiling")
def generate_profiling(dataset_id: str):
    try:
        result = services.generate_profiling_report(dataset_id)
        for workspace_id in services.workspaces_for_dataset(dataset_id):
            _record_workflow_stage(workspace_id, "profiling", "Dataset profiling", "completed" if result.get("report_available") else "failed", result, result.get("error"))
        return {"success": True, **result}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)
    except Exception as e:
        _err("PROFILING_ERROR", f"Profiling failed: {e}", 500)


@app.get("/api/datasets/{dataset_id}/profiling/report")
def get_profiling_report(dataset_id: str):
    try:
        ds = services.get_dataset(dataset_id)
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)
    report_path = Path(ds["filepath"]).parent / "profiling_report.html"
    if not report_path.exists():
        _err("NOT_FOUND", "Profiling report not yet generated. Run POST /api/datasets/{id}/profiling first.", 404)
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=report_path.read_text(encoding="utf-8"))


@app.get("/api/datasets/{dataset_id}/vif")
def dataset_vif(dataset_id: str):
    try:
        result = services.get_vif(dataset_id)
        for workspace_id in services.workspaces_for_dataset(dataset_id):
            _record_workflow_stage(workspace_id, "eda_vif", "VIF analysis", "completed", result)
        return {"success": True, **result}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


# ──────────────────────── Models ────────────────────────

@app.get("/api/models")
def list_models(problem_type: str = Query(default=None)):
    return {"success": True, "models": ml_list_models(problem_type)}


@app.get("/api/models/{model_id}")
def get_model(model_id: str):
    try:
        return {"success": True, **get_model_info(model_id)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


# ──────────────────────── Workspaces ────────────────────────

@app.post("/api/workspaces")
def create_workspace(req: CreateWorkspaceRequest):
    try:
        ws = services.create_workspace(req.dataset_id)
        if OPEN_ACCESS_MODE_ENABLED:
            ai_store.claim_workspace(ws["id"], OPEN_ACCESS_USER_ID)
        dataset = services.get_dataset(req.dataset_id)
        _record_workflow_stage(ws["id"], "dataset_ingestion", "Dataset ingestion", "completed",
                               {k: dataset.get(k) for k in ("id", "filename", "rows", "num_columns", "columns", "dtypes", "missing_values", "duplicate_rows", "validation")})
        _record_workflow_stage(ws["id"], "workspace", "Workspace creation", "completed", {"status": ws["status"]})
        return {"success": True, **ws}
    except ValueError as e:
        _err("CREATE_ERROR", str(e))


@app.get("/api/workspaces/{workspace_id}")
def get_workspace(workspace_id: str):
    try:
        ws = services.get_workspace(workspace_id)
        return {"success": True, **ws}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.post("/api/workspaces/{workspace_id}/column-selection")
def select_workspace_columns(workspace_id: str, req: ColumnSelectionRequest):
    try:
        result = services.apply_column_selection(workspace_id, req.retained_columns, req.target)
        _record_workflow_stage(workspace_id, "column_selection", "Column selection", "completed", result)
        _record_workflow_stage(workspace_id, "model_configuration", "Model configuration", "stale", {"reason": "Dataset columns changed; configure the workflow again"})
        _record_workflow_stage(workspace_id, "training", "Model training", "stale", {"reason": "Dataset columns changed; retraining is required"})
        _record_workflow_stage(workspace_id, "evaluation", "Evaluation", "stale", {"reason": "Dataset columns changed; reevaluation is required"})
        return {"success": True, **result, "workflow_output": _workflow_output(workspace_id, "column_selection", "Column selection", "completed", result)}
    except ValueError as e:
        _err("COLUMN_SELECTION_ERROR", str(e))


@app.post("/api/workspaces/{workspace_id}/column-selection/restore")
def restore_workspace_columns(workspace_id: str):
    try:
        result = services.restore_column_selection(workspace_id)
        _record_workflow_stage(workspace_id, "column_selection", "Column selection", "completed", {**result, "restored_original": True})
        _record_workflow_stage(workspace_id, "training", "Model training", "stale", {"reason": "Original dataset restored; configure and train again"})
        _record_workflow_stage(workspace_id, "evaluation", "Evaluation", "stale", {"reason": "Original dataset restored; configure and evaluate again"})
        return {"success": True, **result, "workflow_output": _workflow_output(workspace_id, "column_selection", "Column selection", "completed", result)}
    except ValueError as e:
        _err("COLUMN_SELECTION_ERROR", str(e))


@app.post("/api/workspaces/{workspace_id}/configure")
def configure_workspace(workspace_id: str, req: ConfigureRequest):
    try:
        ws = services.configure_workspace(
            workspace_id, req.target, req.problem_type,
            req.preprocessing.model_dump(), req.model_id, req.model_params)
        ai_orchestrator.registry.approvals.revoke_for_workspace(workspace_id)
        _record_workflow_stage(workspace_id, "model_configuration", "Model configuration", "completed",
                               {"target": ws["target"], "problem_type": ws["problem_type"], "preprocessing": ws["preprocessing"], "model_id": ws["model_id"], "model_params": ws["model_params"]})
        _record_workflow_stage(workspace_id, "training", "Model training", "stale", {"reason": "Configuration changed; retraining is required"})
        _record_workflow_stage(workspace_id, "evaluation", "Evaluation", "stale", {"reason": "Configuration changed; reevaluation is required"})
        return {"success": True, **ws, "workflow_output": _workflow_output(workspace_id, "model_configuration", "Model configuration", "completed", ws)}
    except ValueError as e:
        _err("CONFIG_ERROR", str(e))


@app.post("/api/workspaces/{workspace_id}/train")
def train_workspace(workspace_id: str):
    try:
        _record_workflow_stage(workspace_id, "training", "Model training", "running")
        result = services.train_workspace(workspace_id)
        _record_workflow_stage(workspace_id, "training", "Model training", "completed", result)
        _record_workflow_stage(workspace_id, "evaluation", "Evaluation", "completed", services.get_evaluation(workspace_id))
        return {"success": True, **result, "workflow_output": _workflow_output(workspace_id, "training", "Model training", "completed", result)}
    except ValueError as e:
        try:
            services.mark_training_failed(workspace_id, str(e))
        except ValueError:
            pass
        _record_workflow_stage(workspace_id, "training", "Model training", "failed", error=str(e))
        _err("TRAINING_ERROR", str(e))
    except Exception as e:
        try:
            services.mark_training_failed(workspace_id, str(e))
        except ValueError:
            pass
        _record_workflow_stage(workspace_id, "training", "Model training", "failed", error=str(e))
        _err("TRAINING_ERROR", f"Training failed: {e}", 500)


@app.post("/api/workspaces/{workspace_id}/training-jobs")
def start_training_job(workspace_id: str, principal: Principal = Depends(current_principal)):
    try:
        require_workspace_owner(workspace_id, principal)
        job = training_jobs.start(workspace_id, principal.user_id)
        _record_workflow_stage(workspace_id, "training", "Model training", "running", {"job_id": job["job_id"]})
        return {"success": True, "job": job}
    except RuntimeError as exc:
        _err("TRAINING_JOB_CONFLICT", str(exc), 409)
    except ValueError as exc:
        _err("TRAINING_ERROR", str(exc), 400)


@app.get("/api/training-jobs/{job_id}")
def get_training_job(job_id: str, principal: Principal = Depends(current_principal)):
    job = training_jobs.get(job_id, principal.user_id)
    if not job:
        _err("TRAINING_JOB_NOT_FOUND", "Training job was not found.", 404)
    return {"job": job}


@app.post("/api/training-jobs/{job_id}/cancel")
def cancel_training_job(job_id: str, principal: Principal = Depends(current_principal)):
    job = training_jobs.cancel(job_id, principal.user_id)
    if not job:
        _err("TRAINING_JOB_NOT_FOUND", "Training job was not found.", 404)
    return {"job": job}


@app.post("/api/workspaces/{workspace_id}/optimize")
def optimize_workspace(workspace_id: str, req: OptimizeRequest = OptimizeRequest()):
    try:
        _record_workflow_stage(workspace_id, "optimization", "Hyperparameter optimization", "running", req.model_dump())
        result = services.optimize_workspace(workspace_id, req.model_dump())
        _record_workflow_stage(workspace_id, "optimization", "Hyperparameter optimization", "completed", result)
        _record_workflow_stage(workspace_id, "evaluation", "Evaluation", "completed", services.get_evaluation(workspace_id))
        return {"success": True, **result, "workflow_output": _workflow_output(workspace_id, "optimization", "Hyperparameter optimization", "completed", result)}
    except ValueError as e:
        _record_workflow_stage(workspace_id, "optimization", "Hyperparameter optimization", "failed", error=str(e))
        _err("OPTIMIZE_ERROR", str(e))
    except Exception as e:
        _record_workflow_stage(workspace_id, "optimization", "Hyperparameter optimization", "failed", error=str(e))
        _err("OPTIMIZE_ERROR", f"Optimization failed: {e}", 500)


@app.get("/api/workspaces/{workspace_id}/optimization")
def get_optimization(workspace_id: str):
    try:
        opt = services.get_workspace_optimization(workspace_id)
        return {"success": True, "optimization": opt}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/workspaces/{workspace_id}/evaluation")
def get_evaluation(workspace_id: str):
    try:
        return {"success": True, **services.get_evaluation(workspace_id)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/workspaces/{workspace_id}/runs")
def get_workspace_runs(workspace_id: str):
    try:
        return {"success": True, "runs": services.get_workspace_runs(workspace_id)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.post("/api/workspaces/{workspace_id}/predict")
async def predict(workspace_id: str, file: UploadFile = File(...)):
    try:
        content = await file.read()
        result = services.predict_workspace(workspace_id, file.filename or "predict.csv", content)
        _record_workflow_stage(workspace_id, "prediction", "Prediction", "completed", {"count": len(result.get("predictions", [])), "warnings": result.get("warnings", []), "download_url": result.get("download_url")})
        return {"success": True, **result}
    except ValueError as e:
        _record_workflow_stage(workspace_id, "prediction", "Prediction", "failed", error=str(e))
        _err("PREDICTION_ERROR", str(e))
    except Exception as e:
        _record_workflow_stage(workspace_id, "prediction", "Prediction", "failed", error=str(e))
        _err("PREDICTION_ERROR", f"Prediction failed: {e}", 500)


@app.get("/api/workspaces/{workspace_id}/artifacts")
def list_artifacts(workspace_id: str):
    try:
        return {"success": True, "artifacts": services.get_artifacts(workspace_id)}
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)


@app.get("/api/workspaces/{workspace_id}/artifacts/{filename}")
def download_artifact(workspace_id: str, filename: str):
    try:
        ws = services.get_workspace(workspace_id)
    except ValueError as e:
        _err("NOT_FOUND", str(e), 404)

    if not ws.get("artifacts_dir"):
        _err("NOT_FOUND", "No artifacts available", 404)

    # Prevent path traversal
    safe_name = Path(filename).name
    filepath = Path(ws["artifacts_dir"]) / safe_name
    if not filepath.exists():
        _err("NOT_FOUND", f"Artifact not found: {safe_name}", 404)

    return FileResponse(filepath, filename=safe_name)
