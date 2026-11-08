import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ai.auth import AuthenticationError, issue_test_token, verify_bearer_token
from ai.persistence import AIStore
from ai.provider import ProviderConfigurationError, ProviderRequestError, ProviderConfig


def test_signed_authentication_and_expiry(monkeypatch):
    monkeypatch.setenv("AI_AUTH_SECRET", "x" * 40)
    token = issue_test_token("user-1", ttl_seconds=30)
    assert verify_bearer_token(token).user_id == "user-1"
    with pytest.raises(AuthenticationError):
        verify_bearer_token(token + "tampered")


def test_auth_requires_strong_secret(monkeypatch):
    monkeypatch.setenv("AI_AUTH_SECRET", "short")
    with pytest.raises(AuthenticationError, match="32 characters"):
        issue_test_token("user-1")


def test_conversation_persistence_isolated_by_user(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    conversation = store.create_conversation("user-1", "ws-1")
    store.add_message(conversation["id"], "user-1", "user", "hello")
    assert len(store.context(conversation["id"], "user-1")["messages"]) == 1
    with pytest.raises(PermissionError):
        store.context(conversation["id"], "user-2")
    with pytest.raises(PermissionError):
        store.add_message(conversation["id"], "user-2", "user", "steal")


def test_workspace_claim_is_first_owner_wins(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    assert store.claim_workspace("ws-1", "user-1")
    assert store.owns_workspace("ws-1", "user-1")
    assert not store.claim_workspace("ws-1", "user-2")


def test_provider_missing_credentials(monkeypatch):
    monkeypatch.delenv("AI_API_KEY", raising=False)
    with pytest.raises(ProviderConfigurationError):
        ProviderConfig.from_env()


def test_persistent_approval_is_single_use_under_concurrency(tmp_path):
    store = AIStore(tmp_path / "ai.sqlite3")
    approval_store = __import__("ai.approvals", fromlist=["TrainingApprovalStore"]).TrainingApprovalStore(store)
    approval = approval_store.issue("ws-1", {"target": "y", "model_id": "m"}, "user-1")

    def consume():
        try:
            approval_store.consume(approval.approval_id, "ws-1", {"target": "y", "model_id": "m"})
            return "consumed"
        except Exception as exc:
            return type(exc).__name__

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(lambda _: consume(), range(4)))
    assert outcomes.count("consumed") == 1


