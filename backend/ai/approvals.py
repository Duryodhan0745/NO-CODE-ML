from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timezone

from .contracts import TrainingApproval


class ApprovalError(ValueError):
    pass


class TrainingApprovalStore:
    """In-process foundation; replace with the project's durable store in Phase 2 persistence work."""

    def __init__(self, persistent_store=None) -> None:
        self._approvals: dict[str, TrainingApproval] = {}
        self._persistent_store = persistent_store

    def attach_store(self, persistent_store) -> None:
        self._persistent_store = persistent_store

    @staticmethod
    def fingerprint(configuration: dict) -> str:
        payload = json.dumps(configuration, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def issue(self, workspace_id: str, configuration: dict, approved_by: str) -> TrainingApproval:
        if not approved_by.strip():
            raise ApprovalError("approved_by is required")
        approval = TrainingApproval(
            approval_id=secrets.token_urlsafe(18),
            workspace_id=workspace_id,
            configuration_fingerprint=self.fingerprint(configuration),
            approved_by=approved_by,
        )
        self._approvals[approval.approval_id] = approval
        if self._persistent_store:
            self._persistent_store.save_approval(approval.approval_id, approved_by, workspace_id, approval.configuration_fingerprint, approval.status)
        return approval

    def consume(self, approval_id: str, workspace_id: str, configuration: dict, approved_by: str | None = None) -> TrainingApproval:
        fingerprint = self.fingerprint(configuration)
        if self._persistent_store:
            row = self._persistent_store.get_approval(approval_id)
            if row is None:
                raise ApprovalError("Training approval not found")
            if row["workspace_id"] != workspace_id:
                raise ApprovalError("Training approval is scoped to another workspace")
            if approved_by and row["user_id"] != approved_by:
                raise ApprovalError("Training approval belongs to another user")
            if row["configuration_fingerprint"] != fingerprint:
                raise ApprovalError("Training approval is stale for the current configuration")
            consumed = self._persistent_store.consume_approval_atomic(approval_id, workspace_id, fingerprint, approved_by)
            if consumed is None:
                raise ApprovalError("Training approval is already consumed or not active")
            approval = self._approvals.get(approval_id) or TrainingApproval(approval_id=row["approval_id"], workspace_id=row["workspace_id"], configuration_fingerprint=row["configuration_fingerprint"], approved_by=row["user_id"], status="approved")
            approval.status = "consumed"
            return approval
        approval = self._approvals.get(approval_id)
        if approval is None and self._persistent_store:
            row = self._persistent_store.get_approval(approval_id)
            if row:
                approval = TrainingApproval(approval_id=row["approval_id"], workspace_id=row["workspace_id"], configuration_fingerprint=row["configuration_fingerprint"], approved_by=row["user_id"], status=row["status"])
        if approval is None:
            raise ApprovalError("Training approval not found")
        if approval.status != "approved":
            raise ApprovalError(f"Training approval is {approval.status}")
        if approval.workspace_id != workspace_id:
            raise ApprovalError("Training approval is scoped to another workspace")
        if approval.configuration_fingerprint != fingerprint:
            approval.status = "expired"
            raise ApprovalError("Training approval is stale for the current configuration")
        approval.status = "consumed"
        if self._persistent_store:
            self._persistent_store.update_approval(approval_id, approval.status)
        approval.approved_at = approval.approved_at.astimezone(timezone.utc)
        return approval

    def revoke_for_workspace(self, workspace_id: str) -> None:
        for approval in self._approvals.values():
            if approval.workspace_id == workspace_id and approval.status == "approved":
                approval.status = "revoked"
        if self._persistent_store:
            self._persistent_store.revoke_approvals(workspace_id)

