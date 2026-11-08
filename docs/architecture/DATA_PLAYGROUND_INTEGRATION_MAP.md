# Data Playground integration map

## Repository findings

| Concern | Unified No-Code ML | Embedded Data Playground | Integration decision |
|---|---|---|---|
| Frontend | React 18 + Vite, JavaScript, Tailwind; landing page switches to `AIWorkspace` and then existing ML wizard in `Workspace.jsx` | React + Vite, TypeScript; single-page explorer/studio in `frontend/src/main.tsx`, custom CSS | Mount the playground experience as an in-app mode; retain ML wizard and add a return/continue bridge. |
| Backend | FastAPI (`backend/main.py`), pandas/scikit-learn; dataset and workspace metadata persisted in `state/workspace_state.json`; uploaded CSV files in configured upload storage | FastAPI (`backend/app/main.py`), SQLite metadata, immutable Parquet version files under configured storage | Use an adapter/proxy contract; do not merge databases or copy storage internals into ML. |
| Identity | Existing bearer auth/Google ID token and AI workspace ownership (`AIStore`) | Dataset and version endpoints have no user auth; AI conversations/tools are separate and mutations use short-lived approvals | Browser requests must go through authenticated No-Code ML API. Resolve/authorize playground dataset ownership before every proxy operation and handoff. Existing playground dataset records currently lack owner IDs, so a migration/claim policy is required before multi-user production use. |
| Dataset contract | ID, filename, local CSV filepath, row/column/schema/missing/duplicate metadata, coarse integer version; no immutable general-purpose dataset version store | UUID dataset ID, active integer version plus UUID version IDs, storage refs, profile, Parquet snapshots, transformation records | Preserve Playground IDs/version IDs as source lineage. For ML, materialize the selected Parquet version through an internal service call into a new ML dataset reference; never trust a client path. |
| ML lifecycle | `services.create_workspace(dataset_id)` starts existing target, column-selection, preprocessing, model, training and evaluation flow | No ML workflow | Handoff creates/reuses an explicit ML dataset record with provenance, then opens existing `AIWorkspace`/`Workspace` with dataset preselected. No target/model/training is selected automatically. |
| Explorer | Small preview; no server-side filters/sorts | Server-side pagination, search, sorting, columns and filters; column stats/distinct values | Reuse Playground row/schema/stat endpoints through adapter. |
| Analysis | Descriptive analysis, correlations, VIF, ydata profiling | Shared Python analysis service exposes descriptive/distribution/correlation/covariance/outliers/hypothesis/grouped/normality operations | Reuse Playground analysis service and preserve the full request/version contract. Existing No-Code ML EDA remains available in ML mode. |
| Visualization | Correlation/profiling outputs | Python visualization registry/generators, 25 chart types, AI tool calling | Reuse Playground registry and generator. Keep one engine. |
| Transformations | ML column selection writes a derived CSV; training preprocessing is a separate leakage-safe scikit-learn pipeline | Preview, approval, immutable Parquet versions and lineage for playground transformations | Keep preparation transformations in Playground and ML train-fold preprocessing in ML. Do not treat these as duplicates: they serve distinct stages. |
| AI | Existing Gemini-backed No-Code ML assistant with workflow approvals and auth | Gemini AI Analyst, context, charts, tools; explicit approval for dataset mutation | Prefer the existing Playground AI tool architecture/context while exposing it in the in-app mode. Proxy/auth context must bind active dataset/version and section. Do not create another provider. |
| Export | ML prediction/artifact downloads | Existing visualization artifact routes; dataset CSV/XLSX/Parquet export/report need inspection/extension | Add backend-generated export endpoints against exact immutable version. Use streamed/file response and opaque authorized references; no paths to browser. |
| Deployment | One Vite frontend and FastAPI backend; frontend `/api` proxy | Backend package at `backend/playground` with versioned storage at `backend/playground/storage` | The engine is imported as a first-party backend module. Browser calls remain on the authenticated No-Code ML API; no iframe or cross-origin service is used. |

## Reuse and change map

**Reuse directly behind an adapter:** Playground row explorer, Python analysis service, visualization registry/generators, transformation preview/approval/apply/history, versioned Parquet storage, AI Analyst tools/context, React studio UI and styles.

**Adapt:** Dataset/version references, authentication and ownership, uploads/compatible dataset selection, environment configuration, internal ML handoff, exact-version export and preparation report. Add a typed handoff record binding Playground dataset/version UUID to ML dataset/workspace and principal.

**Keep exclusive to ML:** target/feature selection, ML workspace state, leakage-safe train/test preprocessing, estimator selection, parameter validation, optimization, training, evaluation and inference.

**Avoid duplicating:** playground datasets/uploads, analysis algorithms, chart generation, preparation transformations, or databases. ML references a handoff materialization and provenance rather than adopting Playground's SQLite schema as its own.

## Current gaps and migration notes

1. Playground data metadata has no authenticated owner/workspace mapping. Add an owner mapping or a controlled legacy claim policy before serving existing IDs to multiple authenticated users. A proxy alone cannot infer ownership from the current SQLite schema.
2. `backend/services.py` stores ML datasets in JSON and local CSV paths and understands CSV only. Add an internal byte/stream materialization entry point that validates parsed schema/rows and records `origin`, source dataset/version IDs, handoff ID and transformation lineage. Preserve ordinary uploads and workspace APIs unchanged.
3. Playground already has immutable version IDs but restore changes the active pointer. Every proxy/export/handoff action must accept the explicit version and verify it exists, rather than trusting mutable active-version state.
4. The current deployment uses a private in-process module boundary. Do not pass filesystem paths to the browser.
5. No-Code ML currently offers one workspace wizard, not the suggested Dashboard/Models/Experiments navigation. Add Data Playground as a distinct top-level mode with internal sections without claiming absent product areas.

## Recommended delivery sequence

1. Bind a stable in-app Data Playground mode and shared active dataset/version state.
2. Add authenticated adapter routes and a mapping/ownership policy.
3. Add exact-version summary, comparison, exports and report.
4. Add a typed, authorized handoff that internally resolves the Parquet version, creates an ML dataset reference and opens the existing wizard preselected.
5. Verify the end-to-end path and run focused regression checks; record unsupported capabilities and deployment assumptions.
