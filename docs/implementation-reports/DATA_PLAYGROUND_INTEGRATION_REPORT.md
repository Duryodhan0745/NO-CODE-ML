# Data Playground integration implementation report

Date: 2026-11-05

## Implemented

- Added first-class Data Playground mode to the No-Code ML landing navigation and app state. It has dedicated internal sections, an active dataset/version selector, breadcrumb, responsive workspace UI, upload for CSV/XLSX/Parquet, server-paged row preview/search, column profiles, analysis actions, the 25-entry Playground chart registry, transformation preview/approval/apply controls, AI Analyst context, and Review & Export.
- Added a private Python adapter that imports and calls the existing Data Playground FastAPI service and storage. Configuration uses `DATA_PLAYGROUND_ROOT` and `DATA_PLAYGROUND_STORAGE_ROOT`. The sibling services remain separate data stores and analysis engines.
- Added authenticated No-Code ML routes for upload, per-user legacy claim, summary/schema/lineage, rows, analysis, visualizations, AI messages, transformation actions, CSV/XLSX/Parquet exports, preparation report, and ML handoff.
- Added backend streamed-file exports for exact versions and a JSON report containing original/final schema, lineage, operations and export timestamp.
- Added ML handoff materialization: the backend validates an explicit Playground version, resolves its immutable version UUID, writes a new ML CSV reference, records source/version/transformation/handoff provenance, creates an existing ML workspace, and returns the selected references. The existing ML workflow is opened with that workspace/dataset; it does not select a target or train.
- Added additive `provenance` metadata to the existing ML dataset JSON record. Existing CSV uploads and ML pipeline remain available.
- Added an architecture/discovery map at `docs/architecture/DATA_PLAYGROUND_INTEGRATION_MAP.md`.

## Architecture and compatibility

Playground keeps its SQLite metadata, immutable Parquet versions, analysis/visualization/transform engines, and Gemini tool-calling. No-Code ML keeps its dataset/workspace JSON, CSV references, authentication and ML preprocessing/training. An in-process backend adapter is used because both repositories are on the same host; production must configure the private service/storage roots and package dependencies. No browser-visible filesystem paths are returned.

Legacy Playground datasets have no owner field in their original schema. The No-Code ML adapter requires explicit per-user claims for non-admin users and automatically claims newly uploaded datasets. This prevents authorization from relying on the dataset ID alone, but a formal owner migration and multi-user import policy remain necessary before production rollout.

## Files added or changed

- Added `backend/data_playground.py`.
- Added `frontend/src/DataPlayground.jsx`.
- Added `docs/architecture/DATA_PLAYGROUND_INTEGRATION_MAP.md`.
- Added this report.
- Updated `backend/config.py`, `backend/main.py`, `backend/services.py`, `backend/ai/persistence.py`, `frontend/src/App.jsx`, `frontend/src/Workspace.jsx`, and `frontend/src/index.css`.

## Verification results

**Verified:** No-Code ML production frontend build passed after integration changes. Data Playground TypeScript/Vite production build passed. Python bytecode compilation passed for changed backend modules. Direct adapter check loaded the actual Data Playground service/storage, observed 25 visualization definitions, and resolved an existing explicit immutable dataset version to its dataframe.

**Not verified:** Full ASGI upload → export → handoff smoke journey did not complete in this host. Full No-Code ML pytest collection includes `tests/e2e_test.py`, which opens localhost and failed with Windows `WinError 10013`. Focused No-Code ML tests reported two failures and then stalled; the final summary was unavailable. Data Playground pytest timed out while its TestClient attempted a Windows socketpair/event-loop setup. No browser journey was run. Therefore the complete integration and regression acceptance criteria are not claimed as passed.

## Known implementation gaps

- The provided React workspace is a functional integration shell for explorer, analyses, chart registry, transformation previews/approvals, AI request context and Review & Export; it does not yet render Playground's full chart artifacts/conversation history experience or expose every analysis parameter and chart configuration control.
- Transformation controls cover common operation paths, but do not yet provide complete parameter forms for every supported transformation (for example arbitrary multi-column selection/reorder and all calculated/filter expressions).
- Before-and-after comparison reports row/column/missing/duplicate deltas and added/removed names; it does not yet robustly classify renamed columns or dtype changes.
- Dataset export uses a temporary backend file response, but currently materializes the selected version as a pandas dataframe first. Very large datasets need chunked/streaming format writers.
- The first handoff creates a new ML dataset reference for each click; idempotent handoff reuse is not implemented. The typed response contains source IDs/version, ML dataset/workspace references, and handoff ID, but it is not persisted as a dedicated handoff table.
- Existing No-Code ML ML workspace ownership APIs are centered on AI workspace claims; ordinary ML API routes were not comprehensively retrofit with user authorization in this change.
- `DATA_PLAYGROUND_ROOT` and storage configuration plus Playground dependencies must be installed in the No-Code ML backend environment for deployment. The root requirements file was not unified to avoid silently pinning cross-project packages.
- Existing compatible No-Code ML datasets are not yet imported into Playground by shared ID; the implemented path supports upload into the Playground store and internal Playground-to-ML handoff.
