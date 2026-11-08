from __future__ import annotations

import json
import sqlite3
import threading
import uuid
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AIStore:
    """Durable AI control-plane records, separate from ML workspace JSON state."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS workspace_owners (
              workspace_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversations (
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
              id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, user_id TEXT NOT NULL, role TEXT NOT NULL,
              content TEXT NOT NULL, tool_name TEXT, evidence_json TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT, fact TEXT NOT NULL,
              source_message_id TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_records (
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, scope_type TEXT NOT NULL CHECK(scope_type IN ('user','project','conversation')),
              workspace_id TEXT, conversation_id TEXT, category TEXT NOT NULL, memory_key TEXT NOT NULL,
              content TEXT NOT NULL, confidence REAL NOT NULL, importance INTEGER NOT NULL,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL, source_conversation_id TEXT,
              source_message_id TEXT, expires_at TEXT, review_policy TEXT NOT NULL DEFAULT 'review_on_change',
              version INTEGER NOT NULL DEFAULT 1, deleted_at TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_history (
              id TEXT PRIMARY KEY, memory_id TEXT NOT NULL, version INTEGER NOT NULL, content TEXT NOT NULL,
              metadata_json TEXT NOT NULL, source_conversation_id TEXT, source_message_id TEXT, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_settings (
              user_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS conversation_summaries (
              conversation_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, summary TEXT NOT NULL,
              covered_through TEXT, message_count INTEGER NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_preferences (
              user_id TEXT NOT NULL, preference_key TEXT NOT NULL, preference_value TEXT NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY(user_id, preference_key)
            );
            CREATE TABLE IF NOT EXISTS workflow_events (
              id TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT NOT NULL, event_type TEXT NOT NULL,
              payload_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tool_executions (
              idempotency_key TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT,
              tool_name TEXT NOT NULL, status TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS training_approvals (
              approval_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
              configuration_fingerprint TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
              consumed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS workflow_proposals (
              proposal_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, workspace_id TEXT NOT NULL,
              conversation_id TEXT, stage_id TEXT NOT NULL, tool_name TEXT NOT NULL,
              configuration_json TEXT NOT NULL, configuration_fingerprint TEXT NOT NULL,
              dataset_version TEXT NOT NULL, workflow_version TEXT NOT NULL,
              resource_json TEXT NOT NULL, rationale TEXT NOT NULL, expected_outcome TEXT NOT NULL,
              warnings_json TEXT NOT NULL, status TEXT NOT NULL, expires_at TEXT NOT NULL,
              execution_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS admin_accounts (
              email TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_user_scope ON memory_records(user_id,scope_type,workspace_id,updated_at);
            CREATE INDEX IF NOT EXISTS idx_memory_active ON memory_records(user_id,deleted_at,expires_at);
            CREATE INDEX IF NOT EXISTS idx_memory_history_record ON memory_history(memory_id,version);
            CREATE INDEX IF NOT EXISTS idx_events_workspace_type ON workflow_events(workspace_id,event_type,created_at);
            CREATE INDEX IF NOT EXISTS idx_proposals_workspace_user ON workflow_proposals(workspace_id,user_id,status,created_at);
            """)
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_upsert_key ON memory_records(user_id,scope_type,ifnull(workspace_id,''),ifnull(conversation_id,''),category,memory_key)")
            # Migrate legacy free-form facts without dropping the original table.
            legacy = db.execute("SELECT id,user_id,workspace_id,fact,source_message_id,created_at FROM memories").fetchall()
            for row in legacy:
                scope = "project" if row["workspace_id"] else "user"
                db.execute("INSERT OR IGNORE INTO memory_records(id,user_id,scope_type,workspace_id,category,memory_key,content,confidence,importance,created_at,updated_at,source_message_id,review_policy) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (row["id"], row["user_id"], scope, row["workspace_id"], "legacy_fact", row["id"], row["fact"], 0.5, 2, row["created_at"], row["created_at"], row["source_message_id"], "legacy_review"))

    @staticmethod
    def hash_password(password: str) -> str:
        salt = os.urandom(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
        return "scrypt$16384$8$1$" + salt.hex() + "$" + digest.hex()

    @staticmethod
    def verify_password(password: str, encoded: str) -> bool:
        try:
            _, n, r, p, salt_hex, digest_hex = encoded.split("$")
            digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p))
            return hmac.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False

    def provision_test_admin(self, email: str, password: str) -> None:
        if len(password) < 12 or "@" not in email:
            raise ValueError("ADMIN_EMAIL and ADMIN_PASSWORD do not meet test-admin requirements")
        with self._connect() as db:
            db.execute("INSERT OR IGNORE INTO admin_accounts VALUES(?,?,?,?)", (email.lower(), self.hash_password(password), "super_admin", self._now()))

    def verify_admin(self, email: str, password: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT password_hash FROM admin_accounts WHERE email=?", (email.lower(),)).fetchone()
        return bool(row and self.verify_password(password, row["password_hash"]))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def claim_workspace(self, workspace_id: str, user_id: str) -> bool:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT user_id FROM workspace_owners WHERE workspace_id=?", (workspace_id,)).fetchone()
            if row:
                return row["user_id"] == user_id
            db.execute("INSERT INTO workspace_owners(workspace_id,user_id,created_at) VALUES(?,?,?)", (workspace_id, user_id, self._now()))
            return True

    def owns_workspace(self, workspace_id: str, user_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM workspace_owners WHERE workspace_id=? AND user_id=?", (workspace_id, user_id)).fetchone()
            return row is not None

    def create_conversation(self, user_id: str, workspace_id: str | None) -> dict[str, Any]:
        cid = uuid.uuid4().hex
        now = self._now()
        with self._connect() as db:
            db.execute("INSERT INTO conversations VALUES(?,?,?,?,?)", (cid, user_id, workspace_id, now, now))
        return {"id": cid, "user_id": user_id, "workspace_id": workspace_id, "created_at": now, "updated_at": now}

    def add_message(self, conversation_id: str, user_id: str, role: str, content: str, tool_name: str | None = None, evidence: dict | None = None) -> dict[str, Any]:
        mid, now = uuid.uuid4().hex, self._now()
        with self._connect() as db:
            row = db.execute("SELECT user_id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
            if not row or row["user_id"] != user_id:
                raise PermissionError("Conversation is not owned by this user")
            db.execute("INSERT INTO messages VALUES(?,?,?,?,?,?,?,?)", (mid, conversation_id, user_id, role, content, tool_name, json.dumps(evidence) if evidence else None, now))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))
        return {"id": mid, "conversation_id": conversation_id, "role": role, "content": content, "created_at": now}

    def conversation_owner(self, conversation_id: str) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT user_id FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        return row["user_id"] if row else None

    def add_message_as_admin(self, conversation_id: str, admin_id: str, role: str, content: str, evidence: dict | None = None) -> dict[str, Any]:
        owner = self.conversation_owner(conversation_id)
        if owner is None:
            raise PermissionError("Conversation not found")
        return self.add_message(conversation_id, owner, role, content, evidence=evidence)

    def context_as_admin(self, conversation_id: str) -> dict[str, Any]:
        owner = self.conversation_owner(conversation_id)
        if owner is None:
            raise PermissionError("Conversation not found")
        return self.context(conversation_id, owner)

    def context(self, conversation_id: str, user_id: str, limit: int = 20) -> dict[str, Any]:
        with self._connect() as db:
            conv = db.execute("SELECT * FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
            if not conv:
                raise PermissionError("Conversation is not owned by this user")
            rows = db.execute("SELECT role,content,tool_name,evidence_json,created_at FROM messages WHERE conversation_id=? ORDER BY created_at DESC LIMIT ?", (conversation_id, min(limit, 20))).fetchall()
            summary = db.execute("SELECT summary,covered_through,message_count,updated_at FROM conversation_summaries WHERE conversation_id=? AND user_id=?", (conversation_id, user_id)).fetchone()
        memories = self.retrieve_relevant_memories(user_id, workspace_id=conv["workspace_id"], conversation_id=conversation_id)
        return {"conversation": dict(conv), "messages": [dict(row) for row in reversed(rows)],
                "summary": dict(summary) if summary else None,
                "memories": [f"[{m['scope_type']}:{m['category']}] {m['content']}" for m in memories]}

    def memory_settings(self, user_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT enabled,updated_at FROM memory_settings WHERE user_id=?", (user_id,)).fetchone()
        return {"enabled": bool(row["enabled"]) if row else True, "updated_at": row["updated_at"] if row else None}

    def set_memory_enabled(self, user_id: str, enabled: bool) -> dict[str, Any]:
        now = self._now()
        with self._connect() as db:
            db.execute("INSERT INTO memory_settings(user_id,enabled,updated_at) VALUES(?,?,?) ON CONFLICT(user_id) DO UPDATE SET enabled=excluded.enabled,updated_at=excluded.updated_at", (user_id, int(enabled), now))
        return {"enabled": enabled, "updated_at": now}

    def save_memory(self, user_id: str, category: str, memory_key: str, content: str, *,
                    scope_type: str = "user", workspace_id: str | None = None,
                    conversation_id: str | None = None, confidence: float = 0.9,
                    importance: int = 3, source_message_id: str | None = None,
                    source_conversation_id: str | None = None, expires_at: str | None = None,
                    review_policy: str = "review_on_change") -> dict[str, Any] | None:
        content = content.strip()
        if not content or len(content) > 1000 or not category or len(category) > 64:
            raise ValueError("Memory content and category are required and must be short")
        if scope_type not in {"user", "project", "conversation"}:
            raise ValueError("Invalid memory scope")
        if not 0 <= confidence <= 1 or not 1 <= importance <= 5:
            raise ValueError("Memory confidence or importance is out of range")
        if scope_type == "project" and not workspace_id:
            raise ValueError("Project memory requires a workspace")
        if scope_type == "conversation":
            if not conversation_id or self.conversation_owner(conversation_id) != user_id:
                raise PermissionError("Conversation memory must belong to the authenticated user")
        if scope_type == "project" and not self.owns_workspace(workspace_id, user_id):
            raise PermissionError("Project memory workspace is not owned by this user")
        if not self.memory_settings(user_id)["enabled"]:
            return None
        now = self._now()
        key = re.sub(r"\s+", " ", memory_key.strip().casefold())[:128]
        if not key:
            raise ValueError("Memory key is required")
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM memory_records WHERE user_id=? AND scope_type=? AND workspace_id IS ? AND conversation_id IS ? AND category=? AND memory_key=? AND deleted_at IS NULL", (user_id, scope_type, workspace_id, conversation_id, category, key)).fetchone()
            if row:
                version = row["version"]
                if row["content"] != content:
                    db.execute("INSERT INTO memory_history VALUES(?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, row["id"], version, row["content"], json.dumps({"confidence": row["confidence"], "importance": row["importance"], "scope_type": row["scope_type"]}), row["source_conversation_id"], row["source_message_id"], now))
                    version += 1
                db.execute("UPDATE memory_records SET content=?,confidence=?,importance=?,updated_at=?,source_conversation_id=?,source_message_id=?,expires_at=?,review_policy=?,version=? WHERE id=?", (content, confidence, importance, now, source_conversation_id, source_message_id, expires_at, review_policy, version, row["id"]))
                memory_id = row["id"]
            else:
                memory_id = uuid.uuid4().hex
                db.execute("INSERT INTO memory_records(id,user_id,scope_type,workspace_id,conversation_id,category,memory_key,content,confidence,importance,created_at,updated_at,source_conversation_id,source_message_id,expires_at,review_policy,version) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)", (memory_id, user_id, scope_type, workspace_id, conversation_id, category, key, content, confidence, importance, now, now, source_conversation_id, source_message_id, expires_at, review_policy))
        return self.get_memory(user_id, memory_id)

    def get_memory(self, user_id: str, memory_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM memory_records WHERE id=? AND user_id=? AND deleted_at IS NULL", (memory_id, user_id)).fetchone()
        return dict(row) if row else None

    @staticmethod
    def _memory_terms(value: str) -> set[str]:
        stop = {"the", "and", "for", "with", "that", "this", "from", "have", "your", "what", "when", "where", "would", "could", "about", "into", "then", "them", "they", "their", "please", "tell", "show", "explain"}
        return {word for word in re.findall(r"[a-z0-9_]+", value.casefold()) if len(word) > 2 and word not in stop}

    def retrieve_relevant_memories(self, user_id: str, query: str = "", *, workspace_id: str | None = None,
                                   conversation_id: str | None = None, limit: int = 8,
                                   scope_type: str | None = None) -> list[dict[str, Any]]:
        if not self.memory_settings(user_id)["enabled"]:
            return []
        if scope_type not in {None, "user", "project", "conversation"}:
            raise ValueError("Invalid memory scope")
        now = self._now()
        scope_filter = " AND scope_type=?" if scope_type else ""
        params: list[Any] = [user_id, now, workspace_id, conversation_id]
        if scope_type:
            params.append(scope_type)
        with self._connect() as db:
            rows = db.execute("SELECT * FROM memory_records WHERE user_id=? AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at>?) AND ((scope_type='user' AND workspace_id IS NULL) OR (scope_type='project' AND workspace_id=?) OR (scope_type='conversation' AND conversation_id=?))" + scope_filter, params).fetchall()
        query_terms = self._memory_terms(query)
        ranked = []
        for row in rows:
            item = dict(row)
            terms = self._memory_terms(item["content"] + " " + item["category"] + " " + item["memory_key"])
            overlap = len(query_terms & terms)
            relevance = (overlap / max(1, len(query_terms))) if query_terms else 0.0
            if query_terms and overlap == 0 and item["importance"] < 4:
                continue
            relevance += item["importance"] * 0.025 + item["confidence"] * 0.025
            if workspace_id and item["scope_type"] == "project":
                relevance += 0.08
            item["relevance_score"] = round(relevance, 4)
            item["why_retrieved"] = "keyword match" if overlap else "high-importance preference"
            ranked.append(item)
        ranked.sort(key=lambda item: (item["relevance_score"], item["updated_at"]), reverse=True)
        return ranked[:max(0, min(limit, 20))]

    def retrieve_user_memory(self, user_id: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        return self.retrieve_relevant_memories(user_id, query, scope_type="user", limit=limit)

    def retrieve_project_memory(self, user_id: str, workspace_id: str, query: str = "", limit: int = 8) -> list[dict[str, Any]]:
        if not self.owns_workspace(workspace_id, user_id):
            raise PermissionError("Project memory workspace is not owned by this user")
        return self.retrieve_relevant_memories(user_id, query, workspace_id=workspace_id, scope_type="project", limit=limit)

    def retrieve_conversation_context(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        return self.context(conversation_id, user_id)

    def retrieve_workflow_history(self, user_id: str, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        if not self.owns_workspace(workspace_id, user_id):
            raise PermissionError("Workflow history workspace is not owned by this user")
        return self.list_events(workspace_id, limit)

    def search_memory(self, user_id: str, query: str, *, workspace_id: str | None = None,
                      conversation_id: str | None = None, limit: int = 8) -> list[dict[str, Any]]:
        return self.retrieve_relevant_memories(user_id, query, workspace_id=workspace_id, conversation_id=conversation_id, limit=limit)

    def invalidate_stale_memory(self, user_id: str, memory_id: str) -> bool:
        return self.delete_memory(user_id, memory_id)

    def list_memories(self, user_id: str, *, workspace_id: str | None = None, scope_type: str | None = None) -> list[dict[str, Any]]:
        if scope_type not in {None, "user", "project", "conversation"}:
            raise ValueError("Invalid memory scope")
        query = "SELECT * FROM memory_records WHERE user_id=? AND deleted_at IS NULL AND (expires_at IS NULL OR expires_at>?)"
        params: list[Any] = [user_id, self._now()]
        if workspace_id:
            query += " AND scope_type='project' AND workspace_id=?"; params.append(workspace_id)
        elif scope_type:
            query += " AND scope_type=?"; params.append(scope_type)
        else:
            query += " AND scope_type='user'"
        query += " ORDER BY importance DESC,updated_at DESC LIMIT 100"
        with self._connect() as db:
            return [dict(row) for row in db.execute(query, params).fetchall()]

    def update_memory(self, user_id: str, memory_id: str, *, content: str | None = None,
                      confidence: float | None = None, importance: int | None = None) -> dict[str, Any] | None:
        current = self.get_memory(user_id, memory_id)
        if not current:
            return None
        next_content = (content if content is not None else current["content"]).strip()
        next_confidence = confidence if confidence is not None else current["confidence"]
        next_importance = importance if importance is not None else current["importance"]
        if not next_content or len(next_content) > 1000 or not 0 <= next_confidence <= 1 or not 1 <= next_importance <= 5:
            raise ValueError("Invalid memory update")
        now = self._now()
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM memory_records WHERE id=? AND user_id=? AND deleted_at IS NULL", (memory_id, user_id)).fetchone()
            if not row:
                return None
            db.execute("INSERT INTO memory_history VALUES(?,?,?,?,?,?,?,?)", (uuid.uuid4().hex, memory_id, row["version"], row["content"], json.dumps({"confidence": row["confidence"], "importance": row["importance"], "scope_type": row["scope_type"]}), row["source_conversation_id"], row["source_message_id"], now))
            db.execute("UPDATE memory_records SET content=?,confidence=?,importance=?,updated_at=?,version=version+1 WHERE id=? AND user_id=?", (next_content, next_confidence, next_importance, now, memory_id, user_id))
        return self.get_memory(user_id, memory_id)

    def delete_memory(self, user_id: str, memory_id: str) -> bool:
        with self._connect() as db:
            db.execute("DELETE FROM memory_history WHERE memory_id=? AND memory_id IN (SELECT id FROM memory_records WHERE id=? AND user_id=?)", (memory_id, memory_id, user_id))
            cur = db.execute("DELETE FROM memory_records WHERE id=? AND user_id=?", (memory_id, user_id))
        return cur.rowcount > 0

    def clear_memories(self, user_id: str, *, workspace_id: str | None = None, scope_type: str | None = None) -> int:
        if workspace_id:
            where, params = "user_id=? AND scope_type='project' AND workspace_id=?", (user_id, workspace_id)
        elif scope_type:
            where, params = "user_id=? AND scope_type=?", (user_id, scope_type)
        else:
            where, params = "user_id=? AND scope_type='user'", (user_id,)
        with self._connect() as db:
            ids = [row[0] for row in db.execute(f"SELECT id FROM memory_records WHERE {where}", params).fetchall()]
            if ids:
                marks = ",".join("?" for _ in ids)
                db.execute(f"DELETE FROM memory_history WHERE memory_id IN ({marks})", ids)
                db.execute(f"DELETE FROM memory_records WHERE id IN ({marks})", ids)
        return len(ids)

    def summarize_conversation(self, conversation_id: str, user_id: str, *, keep_recent: int = 12) -> dict[str, Any] | None:
        with self._connect() as db:
            conv = db.execute("SELECT id FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
            if not conv:
                raise PermissionError("Conversation is not owned by this user")
            rows = db.execute("SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY created_at", (conversation_id,)).fetchall()
        if len(rows) <= keep_recent + 8:
            with self._connect() as db:
                old = db.execute("SELECT summary,covered_through,message_count,updated_at FROM conversation_summaries WHERE conversation_id=? AND user_id=?", (conversation_id, user_id)).fetchone()
            return dict(old) if old else None
        older = rows[:-keep_recent]
        user_objectives = [row["content"].strip() for row in older if row["role"] == "user" and row["content"].strip()]
        assistant_decisions = [row["content"].strip() for row in older if row["role"] == "assistant" and row["content"].strip()]
        summary_lines = ["Earlier conversation context (summarized from persisted messages):"]
        if user_objectives:
            summary_lines.append("User objectives and instructions: " + " | ".join(text[:300] for text in user_objectives[-8:]))
        if assistant_decisions:
            summary_lines.append("Previous assistant responses and recommendations: " + " | ".join(text[:250] for text in assistant_decisions[-6:]))
        summary = "\n".join(summary_lines)[:6000]
        now = self._now(); covered = older[-1]["created_at"]
        with self._connect() as db:
            db.execute("INSERT INTO conversation_summaries VALUES(?,?,?,?,?,?) ON CONFLICT(conversation_id) DO UPDATE SET user_id=excluded.user_id,summary=excluded.summary,covered_through=excluded.covered_through,message_count=excluded.message_count,updated_at=excluded.updated_at", (conversation_id, user_id, summary, covered, len(older), now))
        return {"summary": summary, "covered_through": covered, "message_count": len(older), "updated_at": now}

    def clear_conversation_memory(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as db:
            if not db.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone():
                return False
            db.execute("DELETE FROM messages WHERE conversation_id=? AND user_id=?", (conversation_id, user_id))
            db.execute("DELETE FROM conversation_summaries WHERE conversation_id=? AND user_id=?", (conversation_id, user_id))
            ids = [row[0] for row in db.execute("SELECT id FROM memory_records WHERE user_id=? AND scope_type='conversation' AND conversation_id=?", (user_id, conversation_id)).fetchall()]
            if ids:
                marks = ",".join("?" for _ in ids); db.execute(f"DELETE FROM memory_history WHERE memory_id IN ({marks})", ids); db.execute(f"DELETE FROM memory_records WHERE id IN ({marks})", ids)
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (self._now(), conversation_id))
        return True

    def add_memory(self, user_id: str, fact: str, workspace_id: str | None = None, source_message_id: str | None = None) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO memories VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, user_id, workspace_id, fact, source_message_id, self._now()))

    def set_preference(self, user_id: str, key: str, value: Any) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO user_preferences VALUES(?,?,?,?)", (user_id, key, json.dumps(value), self._now()))

    def preference(self, user_id: str, key: str) -> Any:
        with self._connect() as db:
            row = db.execute("SELECT preference_value FROM user_preferences WHERE user_id=? AND preference_key=?", (user_id, key)).fetchone()
        return json.loads(row[0]) if row else None

    def record_event(self, user_id: str, workspace_id: str, event_type: str, payload: dict) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO workflow_events VALUES(?,?,?,?,?,?)", (uuid.uuid4().hex, user_id, workspace_id, event_type, json.dumps(payload), self._now()))

    def list_events(self, workspace_id: str, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT id,user_id,workspace_id,event_type,payload_json,created_at FROM workflow_events WHERE workspace_id=? ORDER BY created_at DESC LIMIT ?", (workspace_id, limit)).fetchall()
        events = []
        for row in reversed(rows):
            event = dict(row)
            event["payload"] = json.loads(event.pop("payload_json"))
            events.append(event)
        return events

    def record_tool_execution(self, user_id: str, workspace_id: str | None, idempotency_key: str, tool_name: str, status: str, result: dict) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO tool_executions VALUES(?,?,?,?,?,?,?)", (idempotency_key, user_id, workspace_id, tool_name, status, json.dumps(result), self._now()))

    def get_tool_execution(self, user_id: str, idempotency_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM tool_executions WHERE idempotency_key=? AND user_id=?", (idempotency_key, user_id)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def save_approval(self, approval_id: str, user_id: str, workspace_id: str, fingerprint: str, status: str) -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO training_approvals VALUES(?,?,?,?,?,?,?)", (approval_id, user_id, workspace_id, fingerprint, status, self._now(), None))

    def get_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM training_approvals WHERE approval_id=?", (approval_id,)).fetchone()
        return dict(row) if row else None

    def update_approval(self, approval_id: str, status: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE training_approvals SET status=?, consumed_at=? WHERE approval_id=?", (status, self._now() if status == "consumed" else None, approval_id))

    def consume_approval_atomic(self, approval_id: str, workspace_id: str, fingerprint: str, user_id: str | None = None) -> dict[str, Any] | None:
        """Atomically consume an approval; only one concurrent caller can win."""
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            where_user = " AND user_id=?" if user_id else ""
            params = [approval_id, workspace_id, fingerprint]
            if user_id:
                params.append(user_id)
            row = db.execute(f"SELECT * FROM training_approvals WHERE approval_id=? AND workspace_id=? AND configuration_fingerprint=?{where_user}", params).fetchone()
            if not row or row["status"] != "approved":
                db.rollback()
                return None
            now = self._now()
            db.execute("UPDATE training_approvals SET status='consumed', consumed_at=? WHERE approval_id=? AND status='approved'", (now, approval_id))
            db.commit()
            return dict(row)

    def revoke_approvals(self, workspace_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE training_approvals SET status='revoked' WHERE workspace_id=? AND status='approved'", (workspace_id,))

    def save_workflow_proposal(self, proposal: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("""INSERT INTO workflow_proposals VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                proposal["proposal_id"], proposal["user_id"], proposal["workspace_id"], proposal.get("conversation_id"),
                proposal["stage_id"], proposal["tool_name"], json.dumps(proposal["configuration"]),
                proposal["configuration_fingerprint"], str(proposal["dataset_version"]), proposal["workflow_version"],
                json.dumps(proposal["resource_usage"]), proposal["rationale"], proposal["expected_outcome"],
                json.dumps(proposal["warnings"]), proposal["status"], proposal["expires_at"], None,
                proposal["created_at"], proposal["updated_at"]
            ))

    def get_workflow_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM workflow_proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        if not row:
            return None
        value = dict(row)
        for key in ("configuration", "resource", "warnings", "execution"):
            raw = value.pop(f"{key}_json", None)
            value[f"{key}_usage" if key == "resource" else key] = json.loads(raw) if raw else ({} if key in {"configuration", "resource"} else [])
        return value

    def list_workflow_proposals(self, workspace_id: str, user_id: str, *, actionable_only: bool = False) -> list[dict[str, Any]]:
        condition = " AND status='awaiting_approval'" if actionable_only else ""
        with self._connect() as db:
            rows = db.execute(f"SELECT proposal_id FROM workflow_proposals WHERE workspace_id=? AND user_id=?{condition} ORDER BY created_at DESC", (workspace_id, user_id)).fetchall()
        return [proposal for row in rows if (proposal := self.get_workflow_proposal(row["proposal_id"]))]

    def transition_workflow_proposal(self, proposal_id: str, user_id: str, from_statuses: tuple[str, ...], status: str, *, execution: dict | None = None) -> dict[str, Any] | None:
        marks = ",".join("?" for _ in from_statuses)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(f"SELECT * FROM workflow_proposals WHERE proposal_id=? AND user_id=? AND status IN ({marks})", (proposal_id, user_id, *from_statuses)).fetchone()
            if not row:
                db.rollback(); return None
            db.execute("UPDATE workflow_proposals SET status=?, execution_json=COALESCE(?, execution_json), updated_at=? WHERE proposal_id=?", (status, json.dumps(execution) if execution is not None else None, self._now(), proposal_id))
            db.commit()
        return self.get_workflow_proposal(proposal_id)

