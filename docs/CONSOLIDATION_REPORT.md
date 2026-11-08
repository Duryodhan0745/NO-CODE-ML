# Consolidation report

## Unified application layout

- `frontend/` is the only product frontend and contains the No-Code ML workspace
  plus the native Data Playground mode.
- `backend/` is the only product API.
- `backend/playground/` contains the embedded Data Playground engine.
- `backend/playground/storage/` contains the migrated Playground SQLite metadata,
  original uploads, immutable Parquet versions, and visualization artifacts.

The former `data play1/` source directory remains temporarily as a recovery
source. Its `backend/storage/` directory was moved intact to
`backend/playground/storage/`; no datasets or metadata were deleted.

## Verification performed

| Area | Status | Evidence |
|---|---|---|
| Codebase consolidation | IMPLEMENTED BUT NOT FULLY VERIFIED | Engine embedded under `backend/playground`; adapter now imports it in-process. |
| Path repair | IMPLEMENTED BUT NOT FULLY VERIFIED | No-Code ML configuration defaults to project-relative embedded paths. |
| Data Playground storage | FULLY IMPLEMENTED AND VERIFIED | SQLite metadata opens after migration: 6 datasets and 6 versions found. |
| Chatbot context | IMPLEMENTED BUT NOT FULLY VERIFIED | Active-version profile and same-version analysis history are passed as structured context. |
| ML handoff and export | IMPLEMENTED BUT NOT FULLY VERIFIED | Existing authenticated adapter routes retain explicit version references. |
| Frontend production build | BLOCKED | Vite could not create a temporary config file because this merged root is outside the active write sandbox. |
| Backend runtime tests | BLOCKED | The root virtual environment lacks the merged dependencies; the installation attempt did not complete. |

## Changes made

- Copied the existing Data Playground backend source into `backend/playground/`.
- Moved its storage directory without data conversion or deletion.
- Replaced the old configurable sibling-service import with an embedded package import.
- Added missing Data Playground runtime dependencies to `backend/requirements.txt`.
- Added structured per-request Analyst context support in the embedded engine.
- Updated the root README and architecture map for the unified topology.

## Next verification command

After installing the root requirements, run:

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest tests -v
cd ..\frontend
npm run build
```

Then exercise upload, transformation, export, and ML handoff with an authenticated browser session.
