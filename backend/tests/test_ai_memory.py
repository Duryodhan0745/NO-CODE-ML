import pytest

from ai.memory import extract_durable_memories
from ai.persistence import AIStore
from ai.contracts import ContextBundle
from ai.orchestrator import AIOrchestrator
from ai.tools import ToolRegistry


def test_memory_is_persistent_scoped_ranked_and_versioned(tmp_path):
    path = tmp_path / "ai.sqlite3"
    store = AIStore(path)
    assert store.claim_workspace("project-a", "user-a")
    assert store.claim_workspace("project-b", "user-a")
    first = store.save_memory("user-a", "preferred_model", "classification",
                              "I prefer Random Forest for classification.", importance=4)
    project = store.save_memory("user-a", "project_objective", "objective",
                                "Predict churn", scope_type="project", workspace_id="project-a")
    assert first and project
    changed = store.save_memory("user-a", "preferred_model", "classification",
                                "I prefer Logistic Regression for classification.", importance=4)
    assert changed["id"] == first["id"]
    assert changed["version"] == 2
    assert store.retrieve_relevant_memories("user-a", "classification model", workspace_id="project-a")[0]["content"].startswith("I prefer Logistic")
    assert not store.retrieve_project_memory("user-a", "project-b")
    assert not store.retrieve_relevant_memories("user-b", "classification model", workspace_id="project-a")

    restarted = AIStore(path)
    assert restarted.get_memory("user-a", first["id"])["content"] == changed["content"]
    edited = restarted.update_memory("user-a", first["id"], content="Use Logistic Regression by default.")
    assert edited["version"] == 3
    assert restarted.delete_memory("user-a", first["id"])
    assert restarted.get_memory("user-a", first["id"]) is None
    assert restarted.get_memory("user-b", project["id"]) is None


def test_memory_settings_gate_saves_and_retrieval(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    store.save_memory("user", "working_preference", "default", "Use concise explanations")
    store.set_memory_enabled("user", False)
    assert store.retrieve_relevant_memories("user", "explanations") == []
    assert store.save_memory("user", "working_preference", "second", "Use charts") is None
    assert len(store.list_memories("user")) == 1
    store.set_memory_enabled("user", True)
    assert store.retrieve_relevant_memories("user", "explanations")


def test_conversation_context_summarizes_and_can_be_cleared(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    conversation = store.create_conversation("user", None)
    for index in range(24):
        store.add_message(conversation["id"], "user", "user", f"Objective detail {index}")
    summary = store.summarize_conversation(conversation["id"], "user")
    context = store.context(conversation["id"], "user")
    assert summary and "Objective detail 4" in summary["summary"]
    assert len(context["messages"]) == 20
    assert context["summary"]["message_count"] == 12
    assert store.clear_conversation_memory(conversation["id"], "user")
    cleared = store.context(conversation["id"], "user")
    assert cleared["messages"] == []
    assert cleared["summary"] is None


def test_explicit_extraction_ignores_temporary_and_sensitive_instructions():
    preference = extract_durable_memories("I prefer Random Forest for classification.")
    assert preference and preference[0]["category"] == "preferred_model"
    assert not extract_durable_memories("For this run, use Random Forest.")
    assert not extract_durable_memories("Remember that my API key is abc123")
    assert extract_durable_memories("This project's goal is to predict churn.")[0]["scope_type"] == "project"


def test_project_workflow_history_requires_workspace_ownership(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    store.claim_workspace("project-a", "user-a")
    store.record_event("user-a", "project-a", "training", {"status": "completed"})
    assert store.retrieve_workflow_history("user-a", "project-a")[0]["event_type"] == "training"
    with pytest.raises(PermissionError):
        store.retrieve_workflow_history("user-b", "project-a")


def test_orchestrator_receives_ranked_memory_and_authority_rules():
    class Provider:
        messages = None

        def complete_json(self, messages):
            self.messages = messages
            return {"response": "Use a compatible model.", "intent_kind": "recommendation_request",
                    "requested_operation": None, "needs_clarification": False,
                    "tool_name": None, "tool_arguments": {}, "needs_approval": False}

    provider = Provider()
    orchestrator = AIOrchestrator(registry=ToolRegistry(), provider=provider)
    memory = {"scope_type": "user", "category": "preferred_model",
              "content": "I prefer Random Forest for classification.", "why_retrieved": "keyword match"}
    orchestrator.provider_response("Recommend a classifier", ContextBundle(
        memories=["[user:preferred_model] I prefer Random Forest for classification."],
        facts={"relevant_memories": [memory], "dataset_rows": 5}))
    system = provider.messages[0]["content"]
    assert "I prefer Random Forest" in system
    assert "keyword match" in system
    assert "dataset_rows" in system
    assert "dataset suitability" in system and "technical compatibility" in system



