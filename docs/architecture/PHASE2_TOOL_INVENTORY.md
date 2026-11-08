# Existing ML tool inventory for Phase 2

This inventory is derived from `backend/main.py`, `backend/services.py`, `backend/schemas.py`, and `frontend/src/api.js`. The AI tool registry must wrap these operations instead of duplicating them.

| Tool | Existing endpoint/function | Inputs | Output / behavior | Validation and approval |
|---|---|---|---|---|
| `upload_dataset` | `POST /api/datasets/upload` → `services.upload_dataset(filename, content)` | multipart filename and bytes | dataset id, filename, filepath, rows, columns, dtypes, missing values, duplicates | CSV extension, non-empty/size checks; user initiated upload; no training approval |
| `get_dataset_preview` | `GET /api/datasets/{dataset_id}/preview` → `get_preview(dataset_id, rows)` | dataset id, rows 1–200 | preview rows and dataset metadata | dataset existence; read-only |
| `analyze_dataset` | `GET /api/datasets/{dataset_id}/analysis` → `get_analysis(dataset_id)` | dataset id | dimensions, dtypes, missing data, summary and correlation data | dataset existence; read-only |
| `get_correlation` | `GET /api/datasets/{dataset_id}/correlation` → `get_correlation(dataset_id)` | dataset id | correlation payload | dataset existence; read-only |
| `get_vif` | `GET /api/datasets/{dataset_id}/vif` → `get_vif(dataset_id)` | dataset id | feature VIF list and warnings | dataset existence; read-only |
| `generate_profiling` | `POST /api/datasets/{dataset_id}/profiling` → `generate_profiling_report(dataset_id)` | dataset id | profiling status/report metadata | expensive; explicit approval or policy permission |
| `list_models` | `GET /api/models` → `ml.list_models(problem_type)` | optional classification/regression | supported model schemas and parameters | allowlisted problem type; read-only |
| `get_model` | `GET /api/models/{model_id}` → `ml.get_model_info(model_id)` | model id | supported model and parameter schema | model id allowlist; read-only |
| `create_workspace` | `POST /api/workspaces` → `create_workspace(dataset_id)` | dataset id | workspace id/status/dataset reference | dataset existence; creates persisted workspace |
| `get_workspace` | `GET /api/workspaces/{workspace_id}` → `get_workspace(workspace_id)` | workspace id | current authoritative workspace projection | ownership check required when auth exists; read-only |
| `configure_workspace` | `POST /api/workspaces/{workspace_id}/configure` → `configure_workspace(...)` | target, problem type, preprocessing, model id, model params | configured workspace projection | backend validates target/type/model/params/preprocessing; approval to apply recommendation |
| `train_workspace` | `POST /api/workspaces/{workspace_id}/train` → `train_workspace(workspace_id)` | workspace id | run status, metrics, artifact directory/run id | mandatory fresh training approval; backend status check |
| `optimize_workspace` | `POST /api/workspaces/{workspace_id}/optimize` → `optimize_workspace(workspace_id, req_data)` | mode, candidates, metric, trials, CV, timeout, strategy/grid | optimization result, selected model, run data | expensive and consequential; fresh approval policy; bounded timeout |
| `get_optimization` | `GET /api/workspaces/{workspace_id}/optimization` | workspace id | stored optimization result | read-only |
| `get_evaluation` | `GET /api/workspaces/{workspace_id}/evaluation` | workspace id | problem type and actual metrics | only trained workspace; read-only |
| `get_runs` | `GET /api/workspaces/{workspace_id}/runs` | workspace id | run history | read-only |
| `predict` | `POST /api/workspaces/{workspace_id}/predict` → `predict_workspace(...)` | workspace id and CSV bytes | predictions, row count, download URL, warnings | trained workspace, required feature columns; explicit user request |
| `list_artifacts` | `GET /api/workspaces/{workspace_id}/artifacts` | workspace id | artifact metadata | read-only |
| `download_artifact` | `GET /api/workspaces/{workspace_id}/artifacts/{filename}` | workspace id, filename | file response | path traversal protection; read-only |

The current API has no authentication or conversation/project endpoints. Phase 2 must add an internal AI execution layer without pretending those persistence features already exist.




