import pytest

from ai.approvals import ApprovalError, TrainingApprovalStore
from ai.contracts import ContextBundle, IntentKind, ToolRequest
from ai.orchestrator import AIOrchestrator
from ai.tools import ToolDefinition, ToolRegistry


def test_intent_and_plan_are_deterministic_without_provider():
    orchestrator = AIOrchestrator(registry=ToolRegistry())
    intent = orchestrator.understand_intent("Predict house prices")
    assert intent.kind == IntentKind.goal
    assert intent.requested_operation == "regression"
    plan = orchestrator.plan(intent)
    assert plan.missing_decisions == ["dataset and workspace"]
    assert plan.next_action == "upload_dataset"


def test_registry_allowlist_and_idempotency():
    registry = ToolRegistry()
    calls = []
    registry.register(ToolDefinition("read", lambda value: calls.append(value) or {"value": value}))
    request = ToolRequest(tool_name="read", arguments={"value": 3}, idempotency_key="idem-123456")
    first = registry.execute(request)
    second = registry.execute(request)
    assert first.status == "completed"
    assert second.status == "duplicate"
    assert calls == [3]
    blocked = registry.execute(ToolRequest(tool_name="shell", idempotency_key="idem-654321"))
    assert blocked.status == "blocked"
    assert blocked.error_code == "tool_not_allowed"


def test_training_approval_is_scoped_and_single_use():
    store = TrainingApprovalStore()
    config = {"target": "y", "model_id": "random_forest_classifier", "model_params": {"n_estimators": 10}}
    approval = store.issue("ws-1", config, "user-1")
    consumed = store.consume(approval.approval_id, "ws-1", config)
    assert consumed.status == "consumed"
    with pytest.raises(ApprovalError, match="consumed"):
        store.consume(approval.approval_id, "ws-1", config)


def test_training_approval_rejects_stale_configuration():
    store = TrainingApprovalStore()
    approval = store.issue("ws-1", {"target": "y"}, "user-1")
    with pytest.raises(ApprovalError, match="stale"):
        store.consume(approval.approval_id, "ws-1", {"target": "changed"})


def test_training_tool_requires_fresh_approval_even_in_open_access():
    registry = ToolRegistry()
    registry.register(ToolDefinition("train_workspace", lambda workspace_id: {"status": "trained"}, consequential=True))
    result = registry.execute(ToolRequest(tool_name="train_workspace", arguments={"workspace_id": "ws-1"}, idempotency_key="train-no-approval"))
    assert result.status == "blocked"
    assert result.error_code == "approval_required"


def test_provider_structured_response_is_validated():
    class Provider:
        def complete_json(self, messages):
            return {"response": "Ready", "intent_kind": "goal", "requested_operation": None, "needs_clarification": False, "tool_name": None, "tool_arguments": {}, "needs_approval": False}

    result = AIOrchestrator(registry=ToolRegistry(), provider=Provider()).provider_response("Build a model", ContextBundle())
    assert result["response"] == "Ready"


def test_auto_mode_receives_live_workflow_facts_and_conditional_tool_policy():
    class Provider:
        messages = None

        def complete_json(self, messages):
            self.messages = messages
            return {"response": "No more actions are needed.", "intent_kind": "question", "requested_operation": None, "needs_clarification": False, "tool_name": None, "tool_arguments": {}, "needs_approval": False}

    provider = Provider()
    orchestrator = AIOrchestrator(registry=ToolRegistry(), provider=provider)
    orchestrator.provider_response("What should happen next?", ContextBundle(workspace_id="ws-1", facts={"workspace_status": "trained", "metrics": {"r2": 0.8}}), auto_mode=True)
    system = provider.messages[0]["content"]
    assert "AUTO MODE" in system
    assert '"r2": 0.8' in system
    assert "clustering and forecasting are unsupported" in system


def test_provider_structured_response_rejects_extra_or_missing_fields():
    class Provider:
        def complete_json(self, messages):
            return {"response": "fabricated", "intent_kind": "goal"}

    with pytest.raises(ValueError, match="structured contract"):
        AIOrchestrator(registry=ToolRegistry(), provider=Provider()).provider_response("Build a model", ContextBundle())


def test_google_id_token_verification_maps_subject_to_principal(monkeypatch):
    import time
    import ai.auth as auth

    def verified_claims(token, request, audience):
        assert token == "signed-google-token"
        assert audience == "public-client-id"
        return {"sub": "google-user-123", "exp": int(time.time()) + 60}

    monkeypatch.setattr(auth.id_token, "verify_oauth2_token", verified_claims)
    principal = auth.verify_google_id_token("signed-google-token", "public-client-id")
    assert principal.user_id == "google:google-user-123"
    assert principal.expires_at > int(time.time())


def test_current_principal_routes_google_jwt_to_google_verifier(monkeypatch):
    import main
    from ai.auth import Principal

    def verified_principal(token, client_id):
        assert token == "header.payload.signature"
        assert client_id == "public-client-id"
        return Principal(user_id="google:google-user-123", expires_at=9999999999)

    monkeypatch.setattr(main, "GOOGLE_CLIENT_ID", "public-client-id")
    monkeypatch.setattr(main, "verify_google_id_token", verified_principal)
    assert main.current_principal("Bearer header.payload.signature").user_id == "google:google-user-123"

