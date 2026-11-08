# Phase 2 Acceptance Report

Date: 2026-10-31

## Decision

**Not accepted as complete.** Critical live provider and authenticated tool execution passed, but the complete approval-bound training workflow, exhaustive HTTP authorization matrix, deployment/backup validation, and full API/e2e suite are not all verified.

## Linux verification environment

- WSL2 was checked and is unavailable in this environment: `wsl.exe --status` and `wsl.exe -l -q` returned `Wsl/E_ACCESSDENIED`.
- Added `backend/run_linux_verification.sh`, which creates disposable upload/artifact/state directories, selects `gemini-2.5-flash` by default without printing credentials, installs the existing requirements, and runs the focused ML/AI/ASGI suites. It is intended for Ubuntu/WSL2 or Linux CI and does not alter production behavior.
- No Linux HTTP matrix result is claimed from this Windows session.

## Evidence completed

- Live Gemini `gemini-2.5-flash` request with a real CSV dataset.
- Authenticated bearer identity and workspace ownership claim.
- Conversation creation, message persistence, assistant persistence, and tool/workflow persistence.
- Gemini intent classification and allowlisted existing-ML tool execution.
- Provider contract hardening: allowlisted names are supplied to the model; invalid provider contracts return typed errors without fabricated output.
- Persistent approval consumption now uses SQLite `BEGIN IMMEDIATE` plus a conditional status transition. A concurrency test produced exactly one successful consume across four callers.
- Configuration changes revoke active approvals in the persistent store.
- Added configurable request guarding for conversation, tool, and dataset-upload routes, with HTTP 429 responses and `Retry-After`; added explicit environment-driven CORS origins and removed the permissive wildcard.
- SQLite backup/restore smoke validation passed using the native backup API; restored workspace ownership was readable.
- Focused AI/security/persistence tests: **12 passed**.
- Python compilation passed.

## Security and reliability checks

- Unit-level ownership, token validation, approval scope, stale approval, idempotency, and persistence tests pass.
- Persistent approval replay/concurrent consume protection is covered.
- Secret values were not printed.

## Not fully verified

- Complete HTTP-level two-user matrix for every conversation, memory, preference, event, tool-history, and approval operation.
- Full configuration recommendation → approval → configuration application → explicit training approval → training → evaluation → Gemini interpretation through live HTTP.
- Concurrent training execution and server-interruption semantics. Exactly-once execution is not claimed; idempotency is durable for recorded executions, but external ML side effects are not a distributed exactly-once transaction.
- SQLite backup/restore, corruption recovery, retention policy, migration framework, and multi-process deployment load testing.
- Rate limiting is not implemented.
- Rate limiting is implemented as a process-local sliding window; distributed deployments require a shared limiter (for example Redis) before horizontal scaling.
- CORS origins are environment-driven; production must set `ALLOWED_ORIGINS` explicitly.
- Full `tests/e2e_test.py` remains blocked by the environment socket policy (`WinError 10013`); an elevated standalone live workflow was used instead.
- The existing model-count tests are stale: `MODEL_REGISTRY` contains 18 supported models (8 classifiers, 10 regressors), while tests assert 12/6/6. No registry regression was found; expectations require a separately reviewed test update.
- Updated the stale ML expectations to the evidenced registry counts and added consistency coverage; `tests/test_ml.py`: **29 passed**.
- FastAPI `TestClient` endpoint execution is also hanging in this environment, so the requested exhaustive two-user HTTP matrix could not be safely completed; no passing HTTP authorization claim is made from that path.
- An HTTPX `ASGITransport` two-user matrix was implemented to avoid sockets, but its execution also hangs in this Windows environment before producing results. It is recorded as unverified rather than treated as passing.
- Bounded diagnostic isolation identified the hang below application initialization: a minimal ASGI app also hangs on the first `httpx.AsyncClient`/`ASGITransport` request. The real app imported successfully (`imports-start`/`imports-done`), so the block is in the environment's HTTPX ASGI transport/event-loop integration, not authentication, middleware, SQLite, provider initialization, or endpoint code. The matrix remains unverified.
- Existing ML model-count mismatch remains a stale expectation pending separate registry/test reconciliation.

## Files changed

- `backend/ai/approvals.py`
- `backend/ai/persistence.py`
- `backend/main.py`
- `backend/ai/orchestrator.py`
- `backend/tests/test_ai_security_persistence.py`
- `docs/implementation-reports/EXECUTION_PROGRESS.md`


The frontend was not modified and Phase 3 was not started.




