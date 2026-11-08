from __future__ import annotations

import io
import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .analysis_service import run as run_analysis
from .transform_service import apply as apply_transform
from .ai_service import AnalysisInput, DatasetInput, Tool, provider_config, require_provider, tool_specs, run_conversation
from . import config as _config
from . import visualizations as viz


BASE = Path(__file__).resolve().parents[1]
ROOT = Path(os.getenv("DATA_ROOT", str(BASE / "storage"))).resolve()
DB = Path(os.getenv("DATABASE_PATH", str(ROOT / "metadata.sqlite3"))).resolve()
MAX_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "52428800"))
MAX_PAGE = int(os.getenv("MAX_PAGE_SIZE", "200"))


def db() -> sqlite3.Connection:
    ROOT.mkdir(parents=True, exist_ok=True)
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,name TEXT,fmt TEXT,size INT,rows INT,cols INT,profile TEXT,created TEXT,archived INT DEFAULT 0)")
    if "active_version" not in {row[1] for row in con.execute("PRAGMA table_info(datasets)")}: con.execute("ALTER TABLE datasets ADD COLUMN active_version INT DEFAULT 1")
    con.execute("CREATE TABLE IF NOT EXISTS versions(dataset_id TEXT,version INT,version_id TEXT,storage TEXT,original TEXT,created TEXT,PRIMARY KEY(dataset_id,version))")
    con.execute("CREATE TABLE IF NOT EXISTS analyses(operation_id TEXT PRIMARY KEY,dataset_id TEXT,version INT,operation_type TEXT,parameters TEXT,result TEXT,created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS transformations(operation_id TEXT PRIMARY KEY,dataset_id TEXT,parent_version INT,new_version INT,operation TEXT,parameters TEXT,created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS conversations(id TEXT PRIMARY KEY,mode TEXT,created TEXT,updated TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS messages(id TEXT PRIMARY KEY,conversation_id TEXT,role TEXT,content TEXT,created TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS approvals(id TEXT PRIMARY KEY,dataset_id TEXT,source_version INT,operation TEXT,parameters TEXT,fingerprint TEXT,status TEXT,created TEXT,expires TEXT)")
    con.commit()
    return con


def fail(status: int, code: str, message: str) -> None:
    raise HTTPException(status, {"code": code, "message": message})


def dataset(dataset_id: str, include_archived: bool = False) -> dict[str, Any]:
    with db() as con:
        row = con.execute("SELECT * FROM datasets WHERE id=?" + ("" if include_archived else " AND archived=0"), (dataset_id,)).fetchone()
        if not row:
            fail(404, "DATASET_NOT_FOUND", "Dataset was not found.")
        result = dict(row)
        result["profile"] = json.loads(result["profile"])
        versions = con.execute("SELECT version,version_id,created FROM versions WHERE dataset_id=? ORDER BY version", (dataset_id,)).fetchall()
        result["versions"] = [dict(v) for v in versions]
        result["current_version"] = int(result.get("active_version") or max((v["version"] for v in versions), default=1))
        return result


def load(dataset_id: str, version: int) -> pd.DataFrame:
    dataset(dataset_id)
    with db() as con:
        row = con.execute("SELECT storage FROM versions WHERE dataset_id=? AND version=?", (dataset_id, version)).fetchone()
    if not row:
        fail(404, "VERSION_NOT_FOUND", "Dataset version was not found.")
    path = (ROOT / row["storage"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        fail(500, "STORAGE_ERROR", "Dataset storage reference is invalid.")
    return pd.read_parquet(path)


def profile(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result = []
    for name in frame.columns:
        series = frame[name]
        numeric = pd.api.types.is_numeric_dtype(series)
        entry: dict[str, Any] = {"name": str(name), "dtype": str(series.dtype), "inferred_type": "numeric" if numeric else "categorical", "missing_count": int(series.isna().sum()), "missing_percent": round(float(series.isna().mean() * 100), 2), "unique_count": int(series.nunique(dropna=True))}
        if numeric:
            entry["statistics"] = {k: (None if pd.isna(v) else float(v)) for k, v in series.agg(["mean", "median", "std", "min", "max"]).items()}
        else:
            counts = series.dropna().astype(str).value_counts().head(20)
            entry["statistics"] = {"most_frequent": str(counts.index[0]) if len(counts) else None, "frequencies": [{"value": str(k), "count": int(v)} for k, v in counts.items()]}
        result.append(entry)
    return result


app = FastAPI(title="Data Analysis Playground", version="1.1.0")
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/ready")
def ready():
    with db() as con:
        con.execute("SELECT 1")
    return {"status": "ready"}


@app.post("/api/v1/datasets", status_code=201)
async def upload(file: UploadFile = File(...)):
    name = Path((file.filename or "").replace("\\", "/")).name
    fmt = Path(name).suffix.lower().lstrip(".")
    if fmt not in {"csv", "xlsx", "parquet"}:
        fail(415, "UNSUPPORTED_FORMAT", "Supported formats are CSV, XLSX and Parquet.")
    data = await file.read(MAX_BYTES + 1)
    if not data:
        fail(400, "EMPTY_FILE", "The uploaded file is empty.")
    if len(data) > MAX_BYTES:
        fail(413, "FILE_TOO_LARGE", "Upload exceeds configured limit.")
    try:
        frame = pd.read_csv(io.BytesIO(data), sep=None, engine="python") if fmt == "csv" else pd.read_excel(io.BytesIO(data), engine="openpyxl") if fmt == "xlsx" else pd.read_parquet(io.BytesIO(data))
    except Exception:
        fail(400, "INVALID_FILE", "File could not be parsed as the selected format.")
    cols = [str(c).strip() for c in frame.columns]
    if not cols or any(not c for c in cols):
        fail(400, "INVALID_HEADERS", "Headers must be non-empty.")
    if len(cols) != len(set(cols)):
        fail(400, "DUPLICATE_HEADERS", "Duplicate headers are not supported.")
    frame.columns = cols
    dataset_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    original = Path("originals") / f"{dataset_id}.{fmt}"
    storage = Path("versions") / dataset_id / "v1.parquet"
    (ROOT / storage).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / original).parent.mkdir(parents=True, exist_ok=True)
    try:
        (ROOT / original).write_bytes(data)
        frame.to_parquet(ROOT / storage, index=False)
        with db() as con:
            con.execute("INSERT INTO datasets(id,name,fmt,size,rows,cols,profile,created,archived,active_version) VALUES(?,?,?,?,?,?,?,?,?,?)", (dataset_id, name, fmt, len(data), len(frame), len(cols), json.dumps(profile(frame)), now, 0, 1))
            con.execute("INSERT INTO versions VALUES(?,?,?,?,?,?)", (dataset_id, 1, str(uuid.uuid4()), str(storage), str(original), now))
    except Exception:
        for path in (ROOT / original, ROOT / storage):
            path.unlink(missing_ok=True)
        fail(500, "PERSISTENCE_ERROR", "Dataset could not be stored.")
    return dataset(dataset_id)


@app.get("/api/v1/datasets")
def datasets(archived: bool = False):
    with db() as con:
        ids = [r["id"] for r in con.execute("SELECT id FROM datasets WHERE archived=? ORDER BY created DESC", (int(archived),))]
    return {"items": [dataset(x, include_archived=archived) for x in ids]}


@app.get("/api/v1/datasets/{dataset_id}")
def detail(dataset_id: str, archived: bool = False):
    return dataset(dataset_id, include_archived=archived)


@app.post("/api/v1/datasets/{dataset_id}/archive")
def archive(dataset_id: str):
    dataset(dataset_id)
    with db() as con:
        con.execute("UPDATE datasets SET archived=1 WHERE id=?", (dataset_id,))
    return {"dataset_id": dataset_id, "archived": True}


@app.post("/api/v1/datasets/{dataset_id}/restore")
def restore(dataset_id: str):
    dataset(dataset_id, include_archived=True)
    with db() as con:
        con.execute("UPDATE datasets SET archived=0 WHERE id=?", (dataset_id,))
    return dataset(dataset_id)


@app.delete("/api/v1/datasets/{dataset_id}")
def delete(dataset_id: str, confirm: bool = False):
    if not confirm:
        fail(400, "CONFIRMATION_REQUIRED", "Set confirm=true to permanently delete this managed dataset.")
    item = dataset(dataset_id, include_archived=True)
    with db() as con:
        refs = con.execute("SELECT storage,original FROM versions WHERE dataset_id=?", (dataset_id,)).fetchall()
        for ref in refs:
            for key in ("storage", "original"):
                path = (ROOT / ref[key]).resolve()
                if not path.is_relative_to(ROOT):
                    fail(500, "STORAGE_ERROR", "Dataset storage reference is outside managed storage.")
                path.unlink(missing_ok=True)
        con.execute("DELETE FROM versions WHERE dataset_id=?", (dataset_id,))
        con.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
    return {"dataset_id": item["id"], "deleted": True}


@app.get("/api/v1/datasets/{dataset_id}/schema")
def schema(dataset_id: str, version: int = 1):
    item = dataset(dataset_id)
    frame = load(dataset_id, version)
    return {"dataset_id": dataset_id, "version": version, "row_count": len(frame), "column_count": len(frame.columns), "duplicate_row_count": int(frame.duplicated().sum()), "columns": profile(frame)}


@app.get("/api/v1/datasets/{dataset_id}/rows")
def rows(dataset_id: str, version: int = 1, offset: int = 0, limit: int = 50, search: str | None = None, sort_by: str | None = None, sort_dir: str = "asc", columns: list[str] | None = Query(None), filters: list[str] | None = Query(None)):
    if offset < 0 or limit < 1 or limit > MAX_PAGE:
        fail(422, "INVALID_PAGINATION", "Invalid offset or page size.")
    frame = load(dataset_id, version)
    for raw in filters or []:
        try:
            spec = json.loads(raw)
            col, op, value = spec["column"], spec["operator"], spec.get("value")
        except Exception:
            fail(422, "INVALID_FILTER", "Filter must include column, operator and value.")
        if col not in frame:
            fail(422, "INVALID_COLUMN", "Unknown filter column.")
        series = frame[col]
        try:
            if op == "eq": mask = series.eq(value)
            elif op == "ne": mask = series.ne(value)
            elif op in {"gt", "gte", "lt", "lte"}:
                v = pd.to_numeric(pd.Series([value]), errors="raise").iloc[0] if pd.api.types.is_numeric_dtype(series) else value
                mask = {"gt": series.gt, "gte": series.ge, "lt": series.lt, "lte": series.le}[op](v)
            elif op == "contains": mask = series.astype("string").str.contains(str(value), case=False, regex=False, na=False)
            elif op == "is_null": mask = series.isna()
            elif op == "not_null": mask = series.notna()
            elif op == "in": mask = series.isin(value if isinstance(value, list) else [value])
            else: fail(422, "INVALID_FILTER_OPERATOR", "Unsupported filter operator.")
            frame = frame[mask.fillna(False)]
        except HTTPException:
            raise
        except Exception:
            fail(422, "INVALID_FILTER_VALUE", "Filter value is incompatible with the column.")
    if search:
        frame = frame[frame.astype("string").apply(lambda s: s.str.contains(search, case=False, regex=False, na=False)).any(axis=1)]
    if sort_by:
        if sort_by not in frame or sort_dir not in {"asc", "desc"}:
            fail(422, "INVALID_SORT", "Invalid sort column or direction.")
        frame = frame.sort_values(sort_by, ascending=sort_dir == "asc", kind="stable", na_position="last")
    if columns and any(c not in frame for c in columns):
        fail(422, "INVALID_COLUMN", "Unknown selected column.")
    selected = columns or list(frame.columns)
    total = len(frame)
    return {"dataset_id": dataset_id, "version": version, "offset": offset, "limit": limit, "total": total, "columns": selected, "rows": json.loads(frame.iloc[offset:offset + limit][selected].to_json(orient="records", date_format="iso"))}


@app.get("/api/v1/datasets/{dataset_id}/columns/{column}/statistics")
def stats(dataset_id: str, column: str, version: int = 1):
    frame = load(dataset_id, version)
    if column not in frame:
        fail(422, "INVALID_COLUMN", "Unknown column.")
    return next(x for x in profile(frame) if x["name"] == column)


@app.get("/api/v1/datasets/{dataset_id}/columns/{column}/distinct")
def distinct(dataset_id: str, column: str, version: int = 1, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0)):
    frame = load(dataset_id, version)
    if column not in frame:
        fail(422, "INVALID_COLUMN", "Unknown column.")
    counts = frame[column].value_counts(dropna=False, sort=False)
    values = [{"value": None if pd.isna(value) else str(value), "count": int(count)} for value, count in counts.items()]
    page = values[offset:offset + limit]
    return {"dataset_id": dataset_id, "version": version, "column": column, "offset": offset, "limit": limit, "total": len(values), "truncated": offset + len(page) < len(values), "values": page}


class AnalysisRequest(BaseModel):
    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    version: int = 1


@app.post("/api/v1/datasets/{dataset_id}/analysis")
def analysis(dataset_id: str, request: AnalysisRequest):
    frame = load(dataset_id, request.version)
    try: result = run_analysis(frame, dataset_id, request.version, request.operation, request.parameters)
    except ValueError as exc: fail(422, "INVALID_ANALYSIS", str(exc))
    with db() as con:
        con.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?)", (result["operation_id"], dataset_id, request.version, request.operation, json.dumps(request.parameters), json.dumps(result["result"]), datetime.now(timezone.utc).isoformat()))
    return result


@app.get("/api/v1/datasets/{dataset_id}/analysis/history")
def analysis_history(dataset_id: str, limit: int = Query(50, ge=1, le=100)):
    dataset(dataset_id)
    with db() as con: rows_ = con.execute("SELECT operation_id,version,operation_type,parameters,created FROM analyses WHERE dataset_id=? ORDER BY created DESC LIMIT ?", (dataset_id, limit)).fetchall()
    return {"items": [{**dict(row), "parameters": json.loads(row["parameters"])} for row in rows_]}


class VisualizationRequest(BaseModel):
    chart_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    export: str = "png"


@app.post("/api/v1/datasets/{dataset_id}/visualizations")
def visualization(dataset_id: str, request: VisualizationRequest):
    try: return viz.generate(load(dataset_id, request.version), request.chart_type, request.parameters, request.export, ROOT / "artifacts", dataset_id, request.version)
    except ValueError as exc: fail(422, "INVALID_VISUALIZATION", str(exc))
    except Exception as exc: fail(422, "VISUALIZATION_GENERATION_FAILED", f"Chart could not be generated: {type(exc).__name__}.")


@app.get("/api/v1/visualizations/registry")
def visualization_registry(): return {"charts": list(viz.REGISTRY.values()), "limits": {"max_rows": viz.MAX_ROWS, "max_points": viz.MAX_POINTS}}


@app.get("/api/v1/artifacts/{artifact_name}")
def artifact(artifact_name: str):
    path = (ROOT / "artifacts" / Path(artifact_name).name).resolve()
    if not path.is_relative_to((ROOT / "artifacts").resolve()) or not path.is_file(): fail(404, "ARTIFACT_NOT_FOUND", "Artifact was not found.")
    return FileResponse(path)


class TransformRequest(BaseModel):
    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    version: int = 1
    approval_id: str | None = None


def _transform_preview(dataset_id: str, request: TransformRequest):
    frame = load(dataset_id, request.version)
    try: output, impact, warnings = apply_transform(frame, request.operation, request.parameters, preview=True)
    except ValueError as exc: fail(422, "INVALID_TRANSFORMATION", str(exc))
    return {"dataset_id": dataset_id, "source_version": request.version, "operation": request.operation, "parameters": request.parameters, "validation_status": "valid", "warnings": warnings, "impact": impact}


@app.post("/api/v1/datasets/{dataset_id}/transformations/preview")
def transformation_preview(dataset_id: str, request: TransformRequest): return _transform_preview(dataset_id, request)


@app.post("/api/v1/datasets/{dataset_id}/transformations/approval")
def transformation_approval(dataset_id: str, request: TransformRequest):
    preview = _transform_preview(dataset_id, request); fingerprint = hashlib.sha256(json.dumps(preview, sort_keys=True, default=str).encode()).hexdigest(); approval_id = str(uuid.uuid4()); now = datetime.now(timezone.utc); expires = (now + timedelta(minutes=10)).isoformat()
    with db() as con: con.execute("INSERT INTO approvals VALUES(?,?,?,?,?,?,?,?,?)", (approval_id, dataset_id, request.version, request.operation, json.dumps(request.parameters, sort_keys=True), fingerprint, "pending", now.isoformat(), expires))
    return {"approval_id": approval_id, "fingerprint": fingerprint, "preview": preview, "status": "pending"}


@app.post("/api/v1/datasets/{dataset_id}/transformations/apply")
def transformation_apply(dataset_id: str, request: TransformRequest, confirm: bool = False):
    if not confirm: fail(400, "CONFIRMATION_REQUIRED", "Set confirm=true after reviewing the preview.")
    if request.approval_id:
        with db() as con: approval = con.execute("SELECT * FROM approvals WHERE id=? AND dataset_id=?", (request.approval_id, dataset_id)).fetchone()
        expired = approval and datetime.fromisoformat(approval["expires"]) <= datetime.now(timezone.utc)
        if not approval or approval["status"] != "pending" or expired or int(approval["source_version"]) != request.version or approval["operation"] != request.operation or json.loads(approval["parameters"]) != request.parameters: fail(409, "INVALID_APPROVAL", "Approval is stale, expired, reused, or does not match the requested operation.")
    preview = _transform_preview(dataset_id, request); frame = load(dataset_id, request.version)
    try: output, impact, warnings = apply_transform(frame, request.operation, request.parameters)
    except ValueError as exc: fail(422, "INVALID_TRANSFORMATION", str(exc))
    with db() as con:
        current = con.execute("SELECT COALESCE(MAX(version),0) AS v FROM versions WHERE dataset_id=?", (dataset_id,)).fetchone()["v"]; new_version = int(current) + 1
        storage = Path("versions") / dataset_id / f"v{new_version}.parquet"; (ROOT / storage).parent.mkdir(parents=True, exist_ok=True); output.to_parquet(ROOT / storage, index=False)
        operation_id = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat(); con.execute("INSERT INTO versions VALUES(?,?,?,?,?,?)", (dataset_id, new_version, str(uuid.uuid4()), str(storage), None, now)); con.execute("INSERT INTO transformations VALUES(?,?,?,?,?,?,?)", (operation_id, dataset_id, request.version, new_version, request.operation, json.dumps(request.parameters), now)); con.execute("UPDATE datasets SET rows=?,cols=?,profile=?,active_version=? WHERE id=?", (len(output), len(output.columns), json.dumps(profile(output)), new_version, dataset_id))
    if request.approval_id:
        with db() as con: con.execute("UPDATE approvals SET status='consumed' WHERE id=?", (request.approval_id,))
    return {"operation_id": operation_id, "dataset_id": dataset_id, "source_version": request.version, "new_version": new_version, "impact": impact, "warnings": warnings, "status": "completed"}


@app.get("/api/v1/datasets/{dataset_id}/transformations/history")
def transformation_history(dataset_id: str):
    dataset(dataset_id)
    with db() as con: rows_ = con.execute("SELECT * FROM transformations WHERE dataset_id=? ORDER BY created DESC", (dataset_id,)).fetchall()
    return {"items": [dict(row) for row in rows_]}


@app.post("/api/v1/datasets/{dataset_id}/versions/{version}/restore")
def restore_version(dataset_id: str, version: int):
    frame = load(dataset_id, version)
    with db() as con: con.execute("UPDATE datasets SET rows=?,cols=?,profile=?,active_version=? WHERE id=?", (len(frame), len(frame.columns), json.dumps(profile(frame)), version, dataset_id))
    return {"dataset_id": dataset_id, "active_version": version, "rows": len(frame), "columns": len(frame.columns)}


def ai_tools() -> dict[str, Tool]:
    class AITransformInput(DatasetInput):
        operation: str
        parameters: dict[str, Any] = Field(default_factory=dict)
    class VisualizationInput(DatasetInput):
        chart_type: str
        parameters: dict[str, Any] = Field(default_factory=dict)
        export: str = "html"
    return {
        "inspect_dataset": Tool("inspect_dataset", "Inspect dataset metadata and version history.", DatasetInput, lambda x: dataset(x.dataset_id)),
        "inspect_schema": Tool("inspect_schema", "Inspect the authoritative schema for a dataset version.", DatasetInput, lambda x: schema(x.dataset_id, x.version)),
        "descriptive_analysis": Tool("descriptive_analysis", "Run the shared descriptive statistics service.", AnalysisInput, lambda x: run_analysis(load(x.dataset_id, x.version), x.dataset_id, x.version, x.operation, x.parameters)),
        "correlation_analysis": Tool("correlation_analysis", "Run the shared correlation service.", AnalysisInput, lambda x: run_analysis(load(x.dataset_id, x.version), x.dataset_id, x.version, "correlation", x.parameters)),
        "outlier_detection": Tool("outlier_detection", "Run the shared outlier service.", AnalysisInput, lambda x: run_analysis(load(x.dataset_id, x.version), x.dataset_id, x.version, "outliers", x.parameters)),
        "generate_visualization": Tool("generate_visualization", "Generate a validated Python visualization artifact for the active dataset version.", VisualizationInput, lambda x: visualization(x.dataset_id, VisualizationRequest(chart_type=x.chart_type, parameters=x.parameters, version=x.version, export=x.export))),
        "preview_transformation": Tool("preview_transformation", "Preview a transformation without mutation.", AITransformInput, lambda x: _transform_preview(x.dataset_id, TransformRequest(operation=x.operation, parameters=x.parameters, version=x.version))),
    }


@app.get("/api/v1/ai/tools")
def list_ai_tools(): return {"tools": tool_specs(ai_tools()), "provider": provider_config()}


@app.post("/api/v1/ai/tools/{tool_name}")
def invoke_ai_tool(tool_name: str, payload: dict[str, Any]):
    tools = ai_tools()
    if tool_name not in tools: fail(404, "TOOL_NOT_FOUND", "The requested AI tool is not allowlisted.")
    try: return {"tool": tool_name, "result": tools[tool_name].invoke(payload)}
    except (ValueError, KeyError) as exc: fail(422, "INVALID_TOOL_ARGUMENTS", str(exc))


@app.get("/api/v1/ai/context/{dataset_id}")
def ai_context(dataset_id: str, version: int = 1):
    item = dataset(dataset_id); current = schema(dataset_id, version)
    with db() as con: history = [dict(row) for row in con.execute("SELECT operation,parent_version,new_version,created FROM transformations WHERE dataset_id=? ORDER BY created DESC LIMIT 20", (dataset_id,))]
    return {"dataset": {"id": dataset_id, "name": item["name"], "version": version}, "schema": current, "transformation_history": history, "authoritative": True}


@app.post("/api/v1/ai/conversations")
def create_conversation(mode: str = "ask"):
    if mode not in {"ask", "agent"}: fail(422, "INVALID_MODE", "Mode must be ask or agent.")
    conversation_id = str(uuid.uuid4()); now = datetime.now(timezone.utc).isoformat()
    with db() as con: con.execute("INSERT INTO conversations VALUES(?,?,?,?)", (conversation_id, mode, now, now))
    return {"conversation_id": conversation_id, "mode": mode, "messages": []}


@app.get("/api/v1/ai/conversations")
def list_conversations():
    with db() as con:
        rows = con.execute("SELECT id,mode,created,updated FROM conversations ORDER BY updated DESC").fetchall()
    return {"items": [dict(row) for row in rows]}


@app.delete("/api/v1/ai/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    with db() as con:
        if not con.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone():
            fail(404, "CONVERSATION_NOT_FOUND", "Conversation was not found.")
        con.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        con.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    return {"deleted": True}


@app.get("/api/v1/ai/conversations/{conversation_id}")
def conversation(conversation_id: str):
    with db() as con:
        row = con.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone(); messages = con.execute("SELECT role,content,created FROM messages WHERE conversation_id=? ORDER BY created", (conversation_id,)).fetchall()
    if not row: fail(404, "CONVERSATION_NOT_FOUND", "Conversation was not found.")
    return {"conversation_id": conversation_id, "mode": row["mode"], "created": row["created"], "updated": row["updated"], "messages": [dict(m) for m in messages]}


class AIMessageRequest(BaseModel):
    content: str
    dataset_id: str | None = None
    version: int = Field(default=1, ge=1)
    page: str = ""
    selected_visualization: dict[str, Any] | None = None
    context: dict[str, Any] | None = None


@app.post("/api/v1/ai/conversations/{conversation_id}/messages")
def ai_message(conversation_id: str, request: AIMessageRequest | None = Body(default=None), content: str | None = None, dataset_id: str | None = None, version: int = 1):
    page_context = ""
    visualization_context = None
    if request is not None:
        content, dataset_id, version = request.content, request.dataset_id, request.version
        page_context, visualization_context = request.page, request.selected_visualization
    content = content or ""
    if not content.strip(): fail(422, "EMPTY_MESSAGE", "Message content is required.")
    existing = conversation(conversation_id)
    now = datetime.now(timezone.utc).isoformat(); message_id = str(uuid.uuid4())
    with db() as con: con.execute("INSERT INTO messages VALUES(?,?,?,?,?)", (message_id, conversation_id, "user", content, now))
    try: require_provider()
    except RuntimeError as exc: fail(503, "AI_PROVIDER_NOT_CONFIGURED", str(exc))
    context = ai_context(dataset_id, version) if dataset_id else None
    if context is not None and request is not None and request.context:
        context.update(request.context)
    if context is not None: context.update({"current_page": page_context, "selected_visualization": visualization_context})
    try:
        result = run_conversation(mode=existing["mode"], message=content, context=context, tools=ai_tools())
    except RuntimeError as exc:
        fail(502, "AI_PROVIDER_ERROR", str(exc))
    assistant_time = datetime.now(timezone.utc).isoformat()
    with db() as con:
        con.execute("INSERT INTO messages VALUES(?,?,?,?,?)", (str(uuid.uuid4()), conversation_id, "assistant", result["response"], assistant_time))
        con.execute("UPDATE conversations SET updated=? WHERE id=?", (assistant_time, conversation_id))
    artifacts = [x.get("result") for x in result["tools"] if x.get("status") == "completed" and isinstance(x.get("result"), dict) and x["result"].get("artifact_id")]
    return {"conversation_id": conversation_id, "message": {"role": "assistant", "content": result["response"], "created": assistant_time, "artifacts": artifacts}, "tools": result["tools"]}
