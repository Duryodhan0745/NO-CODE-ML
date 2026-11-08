# AI memory lifecycle

The assistant uses SQLite-backed memory in the existing AI control-plane database (`AI_DB_FILE`). `AIStore` creates the memory tables and indexes at startup and migrates legacy free-form `memories` rows into structured records without deleting the original table.

## Retrieval and authority

When a user sends a chat message, the backend persists the turn, extracts only clearly explicit durable preferences or project objectives/decisions, and upserts those records. It then retrieves at most eight records matching the authenticated user, current workspace, conversation, and message terms. Keyword overlap, importance, confidence, and project scope rank the results. Low-importance non-matches are omitted. Long conversations are summarized after 20 messages while the latest 12 are retained in the summary boundary.

Retrieved memory is passed to the AI provider with scope, category, confidence, importance, relevance score, and retrieval reason. The system instruction says memory is contextual guidance only: current workflow records, dataset suitability, and technical compatibility take precedence. Workflow stage history and current metrics are retrieved from backend records, not inferred from memories.

Preference extraction is deliberately conservative. It requires explicit language such as “I prefer,” “my default,” or “from now on,” ignores temporary run/session requests, rejects credential-like content, and updates an existing category/key instead of making a duplicate. User preferences and workspace facts are stored in separate scopes.

## Management endpoints

- `GET/PUT /api/ai/memory-settings` reads or changes long-term memory for the authenticated identity.
- `GET /api/ai/memories?scope=user` lists user memories; `?workspace_id=...` lists the owned workspace's memories.
- `PATCH/DELETE /api/ai/memories/{memory_id}` corrects or deletes one owned record.
- `DELETE /api/ai/memories` clears user memory; adding `workspace_id` clears only that workspace's memory.
- `DELETE /api/ai/conversations/{conversation_id}/memory` clears that conversation's messages, summary, and conversation-scoped memories.
- `GET /api/ai/tools` lists the internal memory operations available to the orchestration layer.

Every route uses the existing principal and workspace ownership checks. Disabling long-term memory prevents new saves and retrieval, but preserves existing records until the user deletes them or enables memory again. Deletion removes the active record and its retained version history.

The memory controls are available from AI Workspace settings and from the assistant's memory button. They show confidence, importance, and timestamps, allow correction/deletion, and explain that a memory is used when it matches the current question. Conversation memory can be cleared from the assistant panel.

## Local verification

Run `pytest backend/tests/test_ai_memory.py` for persistence across store reinitialization, user/workspace isolation, ranking, updates/versioning, disabled memory, deletion, explicit-extraction filtering, conversation summarization/clearing, and workflow-history ownership. Run `npm run build` from `frontend/` to validate the memory controls.



