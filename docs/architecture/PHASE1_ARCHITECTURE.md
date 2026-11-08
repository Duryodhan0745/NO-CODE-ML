# Phase 1: AI integration architecture and workflow contracts

Status: complete for the current repository boundary. This is a design artifact, not a claim that the AI integration is implemented.

## Repository evidence

The frontend is React/Vite. `frontend/src/Workspace.jsx` owns the ten-step ML workflow and calls `frontend/src/api.js`. The backend is FastAPI (`backend/main.py`) with business logic in `backend/services.py`, a persisted JSON state file (`backend/state/workspace_state.json`), and scikit-learn/XGBoost/Optuna functionality in `backend/ml.py` and `backend/optimizer.py`.

The existing ML engine already provides the execution boundary. The AI layer must call these HTTP operations or a narrow service adapter; it must not reimplement preprocessing, training, evaluation, prediction, or optimization.

## Workflow contracts

Each workflow keeps only phase outputs that exist today. `conversation_id`, `project_id`, and AI memory identifiers are proposed additions and remain separate from ML workspace state.

### 1. Intent understanding

**Inputs:** user message, active conversation/project context, current ML workspace snapshot, optional attachment metadata.

**Output:** `IntentResult` with `kind` (`question`, `goal`, `recommendation_request`, `configuration_change`, `execution_request`, `approval`, `general`), normalized objective, referenced workspace, requested operation, missing information, and confidence.

**Dependencies:** conversation context and workspace snapshot; no ML mutation.

**Transition:** `received -> intent_resolved` or `received -> clarification_required`.

**Approval:** none. The result is advisory until a later command is validated.

### 2. Dataset intelligence

**Inputs:** real `dataset_id`, `GET /api/datasets/{id}/analysis`, preview, correlation, VIF, and optional profiling report.

**Output:** `DatasetContext` containing only returned dimensions, columns, dtypes, missing values, duplicates, correlations, VIF values, and profiling status. Unknown values stay unknown.

**Dependencies:** dataset must exist; analysis tools are read-only.

**Transition:** `dataset_uninspected -> dataset_context_ready` or `dataset_context_failed`.

**Approval:** none for inspection; profiling is a potentially expensive operation and should be explicitly requested or policy-allowed.

### 3. Workflow planning

**Inputs:** `IntentResult`, `DatasetContext`, current workspace status/configuration, supported model list.

**Output:** `WorkflowPlan` with ordered existing operations, prerequisites, missing decisions, proposed next action, and approval requirements.

**Dependencies:** actual workspace status and model registry; no invented parameters.

**Transition:** `intent_resolved -> plan_ready` or `plan_ready -> clarification_required`.

**Approval:** plan generation is not approval to mutate or train.

### 4. Decision intelligence

**Inputs:** plan, dataset facts, workspace configuration, prior tool results, user preferences.

**Output:** `DecisionResult` separating facts, assumptions, recommendations, unknowns, and invalidated downstream outputs.

**Dependencies:** validated tool results only.

**Transition:** `plan_ready -> decision_ready` or back to clarification.

**Approval:** decisions affecting configuration become proposals.

### 5. Recommendation generation

**Inputs:** current operation/model/task, supported model schema from `GET /api/models`, dataset context, decision result.

**Output:** `RecommendationCard` containing a proposed operation, exact supported parameter values, rationale, limitations, and actions (`continue`, `edit`, `explain`, `try_another`, `reject`).

**Dependencies:** parameter values must be validated by the existing backend configuration path before becoming validated configuration.

**Transition:** `decision_ready -> recommendation_proposed -> recommendation_validated` or `recommendation_rejected`.

**Approval:** user approval is required before applying a recommendation.

### 6. Configuration population

**Inputs:** approved recommendation, dataset/workspace identity, existing `ConfigureRequest` schema.

**Output:** `ConfigurationResult` with proposed, validated, approved, and applied configuration kept as distinct records.

**Dependencies:** target, problem type, preprocessing, model id, and params must pass backend validation.

**Transition:** `recommendation_validated -> awaiting_configuration_approval -> applying_configuration -> configured` or `configuration_failed`.

**Approval:** applying a recommendation requires explicit user approval. Training approval remains separate.

### 7. Tool execution

**Inputs:** allowlisted tool name, validated arguments, workspace identity, idempotency key.

**Output:** `ToolExecutionResult` with operation id, status, exact backend response, timestamps, and error details.

**Dependencies:** backend authorization, argument validation, and current workspace status.

**Transition:** `tool_requested -> executing -> completed` or `failed`; retries must be bounded and idempotent.

**Approval:** read-only inspection can be policy-allowed; configuration, optimization, training, and prediction follow explicit policy.

### 8. Training approval

**Inputs:** validated/applied configuration, current workspace fingerprint, user approval token scoped to that fingerprint.

**Output:** `TrainingApproval` with approval id, approver, timestamp, configuration fingerprint, expiration/revocation status.

**Dependencies:** backend must revalidate authorization, current configuration, stale status, and duplicate execution before calling `/train` or `/optimize`.

**Transition:** `configured -> awaiting_training_approval -> training_approved -> training_executing -> trained` or `approval_invalidated`/`training_failed`.

**Approval:** mandatory in every mode, including any future auto mode.

### 9. Result interpretation

**Inputs:** exact `GET /evaluation`, `GET /runs`, optimization result, artifact metadata, and tool status.

**Output:** `ResultInterpretation` with observed metrics, plain-language explanation, limitations, errors, and next valid actions.

**Dependencies:** actual completed tool output; no inferred or fabricated metrics.

**Transition:** `trained -> results_available -> interpreted` or `execution_failed -> recovery_advice`.

**Approval:** explanation is read-only; follow-up actions use their own approval rules.

### 10. Conversational intelligence

**Inputs:** user message, intent, current phase outputs, relevant memory, tool evidence.

**Output:** response text plus structured actions/cards and citations to actual tool evidence.

**Dependencies:** orchestrator controls context selection and tool routing; LLM output is not a security boundary.

**Transition:** `turn_received -> response_ready`; actions may create a new workflow phase.

**Approval:** response generation never implies approval; explicit approval actions are separate typed events.

### 11. Memory and context management

**Inputs:** conversation turns, extracted durable preferences, project/workspace references, tool results.

**Output:** `ContextBundle` containing active project, workspace snapshot, recent turns, relevant facts, and memory provenance.

**Dependencies:** persisted authoritative workflow state remains separate from free-form messages.

**Transition:** `context_requested -> context_assembled` or `context_unavailable`.

**Approval:** never store secrets or unverified claims; user-controlled deletion/retention is required before production use.

### 12. Error recovery

**Inputs:** typed backend error, failed tool operation, stale state, missing provider credentials, or malformed model output.

**Output:** `RecoveryAction` with safe retry/clarification/reconfiguration recommendation, retry count, and user-visible explanation.

**Dependencies:** actual error payload and idempotency record.

**Transition:** `failed -> recovery_advice -> retry_requested` or `blocked`.

**Approval:** retrying consequential operations requires the same approval boundary; stale training approvals are never reused.

### 13. Workflow persistence

**Inputs:** phase outputs, status transitions, approvals, operation ids, conversation/project ids.

**Output:** append-only event records plus current phase projection. ML workspace state remains owned by the existing service.

**Dependencies:** durable store and transaction/idempotency strategy.

**Transition:** every phase writes a state event before exposing its result.

**Approval:** persistence must enforce tenant/user ownership when authentication is added; no frontend-only authority.

## Security boundaries

- The LLM may propose structured actions, never execute arbitrary code or call arbitrary URLs.
- The registry allowlists read-only dataset tools, model listing, configuration validation, training, optimization, prediction, evaluation, runs, and artifacts.
- Every consequential tool call validates workspace ownership, current status, arguments, idempotency key, and approval scope on the backend.
- A configuration change invalidates training approvals and downstream results.
- Provider credentials stay server-side and missing credentials produce a typed configuration error.

## Phase 1 validation checklist

- All thirteen responsibility areas have inputs, outputs, dependencies, transitions, and approvals.
- Existing ML execution remains the only source of metrics, artifacts, and model behavior.
- Proposed configuration, validated configuration, approved configuration, applied configuration, and executed results are distinct.
- Read-only inspection is separated from mutations and mandatory training approval.
- Conversation and memory are contextual layers, not authoritative workflow state.
- Error recovery cannot bypass policy or reuse stale approvals.




