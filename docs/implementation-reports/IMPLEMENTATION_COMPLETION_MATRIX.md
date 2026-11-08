# Implementation completion matrix

This is the current evidence based checklist for the production directive. “Verified” means a local command or an existing automated test exercised the behavior; it does not imply deployment hardening.

| Requirement | Implementation | Evidence | Status |
|---|---|---|---|
| CSV upload, validation, preview | `backend/services.py`, `backend/main.py`, `frontend/src/Workspace.jsx` | Existing upload and service tests; manual API path | Implemented; API suite environment-limited |
| Workflow stage inventory | `docs/workflows/workflow-inventory.md` | Repository audit of frontend stages, service endpoints, supported model/task operations, persistence and gaps | Documented |
| Versioned column selection and restore | `backend/services.py`, `backend/main.py`, `frontend/src/Workspace.jsx`, `frontend/src/api.js` | `test_column_selection_creates_authoritative_version_and_restores_original`; service suite **46 passed**; frontend build passed | Implemented and verified |
| Shared typed workflow API output | `backend/ai/contracts.py`, `backend/main.py` | `workflow_output` envelope is attached to column selection, configuration, training and optimization responses while legacy fields remain; real training output is asserted by Playwright | Implemented for consequential workflow endpoints; read-only endpoints retain established payloads |
| Three-choice HITL proposal card | `backend/ai/approval_workflow.py`, `backend/main.py`, `frontend/src/AICompanion.jsx` | `frontend/tests/hitl.spec.js`: YES, NO, JUST FILL and refresh cases passed against real backend | Implemented and browser verified |
| Persistent approval lifecycle | SQLite `workflow_proposals` table; proposal create/get/decision APIs | `test_workflow_proposals.py`: YES, NO, JUST FILL, stale fingerprint/dataset/configuration and concurrent replay checks | Implemented and verified at service level |
| JUST FILL form population | `frontend/src/AICompanion.jsx`, `frontend/src/Workspace.jsx` | Playwright verifies AI-populated banner, editable numeric parameter (5→6), and zero execution requests | Implemented and browser verified |
| Open Access AI training approval gate | `backend/ai/tools.py`, `backend/main.py`, `backend/ai/orchestrator.py` | `test_training_tool_requires_fresh_approval_even_in_open_access`; `test_ai_execution` focused gate tests | Implemented for AI tools; manual ML routes remain user-operated |
| Unified resource estimation and limits | `backend/ai/approval_workflow.py`, `backend/ai/resource_limits.py`, `backend/config.py`, `backend/services.py` | `test_resource_limits_report_blocking_conditions` passed; proposal creation compares measured/estimated values against environment limits; execution uses bounded job slots | Implemented with local-safe defaults; training-duration enforcement is post-run detection rather than hard cancellation |
| Profiling, correlation, VIF | `backend/services.py` | Existing service tests and API routes | Implemented |
| Leakage-safe preprocessing | `backend/ml.py`, `backend/services.py`, `backend/optimizer.py` | `pytest tests/test_ml.py -k 'not registry'`: 26 passed before additions | Verified in ML tests |
| Model catalog and validation | `backend/ml.py` | Factory tests; 18 registered estimators | Implemented |
| XGBoost, SVM, gradient estimators | `backend/ml.py` | Registry/factory and dependency import | Implemented; integration test remains |
| Optuna optimization | `backend/optimizer.py`, `backend/services.py` | Existing optimizer path | Implemented; end-to-end environment-limited |
| Bayesian optimization | `backend/optimizer.py`, `backend/services.py`, `frontend/src/Workspace.jsx` | Optuna TPE smoke test and UI strategy | Implemented; end-to-end environment-limited |
| GridSearchCV | `backend/optimizer.py`, API optimize route | Leakage-safe sklearn pipeline implementation | Implemented; needs integration execution |
| RandomizedSearchCV | `backend/optimizer.py`, API optimize route | Leakage-safe sklearn pipeline implementation | Implemented; needs integration execution |
| Artifact export and prediction | `backend/services.py`, `backend/main.py` | Existing artifact/prediction tests | Implemented; API environment-limited |
| Persistent workspace state | `backend/config.py`, `backend/services.py` | JSON state written on dataset/workspace mutations | Implemented; restart test remains |
| Warm responsive accessible UI | `frontend/src/index.css`, `frontend/src/Workspace.jsx` | `npm run build` | Verified build |
| Durable database/auth/background jobs | AI data uses SQLite; workspace state uses JSON; no background job runner | Existing tests exercise persistence selectively | Partially implemented; production multi-process durability is not verified |
| Clustering, time-aware splits, advanced feature selection | No complete backend path | No verified implementation | Incomplete |

Known verification constraints: the repository's e2e collector makes a localhost request, which is denied by this sandbox's socket policy. Existing tests that assert exactly 12 models also conflict with the expanded catalog required by the product directive.

Current HITL verification: `python -m pytest backend/tests/test_workflow_proposals.py -q` completed with **6 passed** (including resource-limit assertions); the previously completed focused backend run was **57 passed**. `npm run build` completed successfully. `PW_BASE_URL=http://127.0.0.1:5174 npx playwright test tests/hitl.spec.js` completed with **4 passed** against the real FastAPI backend: YES executes once and returns actual training output, NO makes no execution request, JUST FILL updates editable fields without execution, and refresh restores one pending proposal. The repository-wide pytest collector still includes `tests/e2e_test.py`, which attempts a localhost socket during collection and is denied by this sandbox; that test was not counted as a product failure.

Remaining limitations: the browser suite covers A-D and backend tests cover stale fingerprint/dataset/configuration, expiry/replay and concurrent approval behavior; browser-specific stale/expiry/concurrency scenarios are not yet separately scripted. Read-only dataset/model endpoints still expose their established compatibility payloads rather than the consequential-stage `workflow_output` envelope. Global limits are process-local for concurrent-job admission and require coordinated enforcement for multi-process deployment.



