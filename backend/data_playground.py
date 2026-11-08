"""Private adapter for the embedded Data Playground engine.

The adapter calls the existing Python analysis/visualization/transformation
engine directly. It intentionally returns storage references only internally;
API responses never contain filesystem paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from config import DATA_PLAYGROUND_ROOT, DATA_PLAYGROUND_STORAGE_ROOT


def _load_service():
    source = DATA_PLAYGROUND_ROOT / "main.py"
    if not source.is_file():
        raise HTTPException(503, detail={"code": "playground_unavailable", "message": "Embedded Data Playground engine is unavailable."})
    import os
    os.environ.setdefault("DATA_ROOT", str(DATA_PLAYGROUND_STORAGE_ROOT))
    try:
        from playground import main as service
        return service
    except Exception as exc:
        raise HTTPException(503, detail={"code": "playground_unavailable", "message": f"Data Playground service could not be loaded: {exc}"}) from exc


def dataset(dataset_id: str) -> dict[str, Any]:
    return _load_service().dataset(dataset_id)


def schema(dataset_id: str, version: int) -> dict[str, Any]:
    return _load_service().schema(dataset_id, version)


def rows(dataset_id: str, **params) -> dict[str, Any]:
    # The embedded function is also a FastAPI route, whose omitted parameters
    # otherwise retain Query(...) objects during an in-process call.
    return _load_service().rows(dataset_id, sort_by=None, sort_dir="asc", columns=None, filters=None, **params)


def transformations(dataset_id: str) -> dict[str, Any]:
    return _load_service().transformation_history(dataset_id)


def analyses(dataset_id: str, limit: int = 50) -> dict[str, Any]:
    return _load_service().analysis_history(dataset_id, limit)


def import_dataframe(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Import an existing DataFrame into Data Playground as a new dataset version 1."""
    import io
    import json
    import uuid
    from datetime import datetime, timezone
    service = _load_service()
    ROOT = service.ROOT
    cols = [str(c).strip() for c in frame.columns]
    if not cols or any(not c for c in cols):
        raise ValueError("Headers must be non-empty.")
    clean_frame = frame.copy()
    clean_frame.columns = cols
    dataset_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    parquet_buf = io.BytesIO()
    clean_frame.to_parquet(parquet_buf, index=False)
    data = parquet_buf.getvalue()
    original = Path("originals") / f"{dataset_id}.parquet"
    storage = Path("versions") / dataset_id / "v1.parquet"
    (ROOT / storage).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / original).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / original).write_bytes(data)
    (ROOT / storage).write_bytes(data)
    with service.db() as con:
        con.execute(
            "INSERT INTO datasets(id,name,fmt,size,rows,cols,profile,created,archived,active_version) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (dataset_id, name, "parquet", len(data), len(clean_frame), len(cols), json.dumps(service.profile(clean_frame)), now, 0, 1),
        )
        con.execute("INSERT INTO versions VALUES(?,?,?,?,?,?)", (dataset_id, 1, str(uuid.uuid4()), str(storage), str(original), now))
    return service.dataset(dataset_id)


def create_ml_materialization(dataset_id: str, version: int) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    service = _load_service()
    item = service.dataset(dataset_id)
    version_row = next((v for v in item["versions"] if int(v["version"]) == version), None)
    if not version_row:
        raise HTTPException(404, detail={"code": "version_not_found", "message": "Selected Data Playground version does not exist."})
    frame = service.load(dataset_id, version)
    if frame.empty or len(frame.columns) == 0:
        raise HTTPException(422, detail={"code": "empty_dataset", "message": "The selected version has no rows or columns."})
    if frame.columns.duplicated().any():
        raise HTTPException(422, detail={"code": "invalid_schema", "message": "The selected version has duplicate column names."})
    return str(version_row["version_id"]), frame, item


def export_frame(dataset_id: str, version: int) -> pd.DataFrame:
    service = _load_service()
    service.dataset(dataset_id)
    return service.load(dataset_id, version)


def invoke(operation: str, dataset_id: str, payload: dict[str, Any]):
    service = _load_service()
    if operation == "upload":
        return service.upload
    if operation == "analysis":
        return service.analysis
    if operation == "visualization":
        return service.visualization
    if operation == "preview":
        return service.transformation_preview
    if operation == "approval":
        return service.transformation_approval
    if operation == "apply":
        return service.transformation_apply
    if operation == "registry":
        return service.visualization_registry
    raise ValueError("Unsupported Data Playground operation")


def ai_message(dataset_id: str, version: int, content: str, page: str, context: dict[str, Any] | None = None):
    service = _load_service()
    conversation_id = service.create_conversation("ask")["conversation_id"]
    return service.ai_message(conversation_id, service.AIMessageRequest(content=content, dataset_id=dataset_id, version=version, page=page, context=context))


def registry():
    return _load_service().visualization_registry()

