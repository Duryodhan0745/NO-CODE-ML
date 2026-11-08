# Repository structure

The repository is organized by runtime responsibility while retaining the existing FastAPI and Vite entry points.

```text
no_code_ml/
├── frontend/                  React/Vite application, browser tests, and public assets
│   ├── src/                   UI, workflow screens, AI companion, and API client
│   └── tests/                 Playwright acceptance tests
├── backend/                  FastAPI application and ML runtime
│   ├── ai/                    Agents, orchestration, providers, memory, contracts, and HITL approvals
│   ├── tests/                 Backend unit, API, persistence, and AI tests
│   ├── state/                 Runtime persistent state for the backend launch mode
│   ├── main.py                FastAPI entry point
│   ├── services.py            Dataset/workspace/training services
│   ├── ml.py                  Model registry and preprocessing
│   ├── optimizer.py           Hyperparameter optimization
│   └── config.py              Runtime settings and data-directory configuration
├── docs/                     Architecture, AI, workflow, implementation, and testing documentation
├── scripts/verification/     Reproducible backend verification harnesses
├── scripts/maintenance/      Maintenance/document-generation scripts
├── state/                    Alternate root-launched persistent state location
└── README.md
```

## Boundaries and conventions

- Start the backend from `backend/` so its relative data directories resolve to `backend/state`, `backend/uploads`, and `backend/artifacts`. Environment variables can point these locations elsewhere.
- `backend/ai` contains AI-specific code only. ML behavior remains in the backend root modules until a future extraction can be made without changing runtime contracts.
- Add new AI agents, tools, providers, memory behavior, contracts, or approvals under `backend/ai`.
- Add new algorithms and ML pipeline behavior to `backend/ml.py`, `backend/optimizer.py`, or a future `backend/ml_engine/` package after introducing explicit compatibility tests.
- Add API routes to `backend/main.py` or extract route groups only with an import-compatible FastAPI router migration.
- Add backend tests under `backend/tests`; add browser acceptance tests under `frontend/tests`.
- Maintain architecture and lifecycle documents under `docs/architecture`; AI design under `docs/ai`; workflow inventory under `docs/workflows`; implementation evidence under `docs/implementation-reports`.
- Use absolute imports relative to the backend launch directory (`ai`, `services`, `ml`, `config`) consistently with the existing startup commands.

The restructuring intentionally avoids duplicating modules or introducing empty placeholder packages. Runtime data and SQLite files are not source-controlled architecture assets and remain in their existing data locations.



