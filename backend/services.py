"""Business logic: dataset ops, workspace management, training pipeline, prediction."""

import io
import json
import hashlib
import uuid
import logging
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import (
    UPLOAD_DIR, ARTIFACTS_DIR, MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS, MAX_PREVIEW_ROWS, TEST_SIZE, RANDOM_STATE, STATE_DIR, STATE_FILE, MAX_DATASET_ROWS,
)
from ai.resource_limits import ResourceLimitError, enforce, job_slot
from ml import MODEL_REGISTRY, create_model, build_preprocessing_pipeline, compute_metrics

log = logging.getLogger(__name__)

# ponytail: in-memory stores, add SQLite/postgres when persistence matters
_datasets: dict[str, dict] = {}
_workspaces: dict[str, dict] = {}

def _save_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"datasets": _datasets, "workspaces": _workspaces}, indent=2), encoding="utf-8")

def load_state() -> None:
    if not STATE_FILE.exists():
        return
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        _datasets.update(state.get("datasets", {}))
        _workspaces.update(state.get("workspaces", {}))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load persisted state: %s", exc)

load_state()


# ──────────────────────────── Dataset ops ────────────────────────────

def upload_dataset(filename: str, content: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(f"File too large ({size_mb:.1f} MB). Maximum: {MAX_UPLOAD_SIZE_MB} MB")
    if not content:
        raise ValueError("File is empty")

    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = UPLOAD_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filepath = dataset_dir / filename
    filepath.write_bytes(content)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        filepath.unlink(missing_ok=True)
        dataset_dir.rmdir()
        raise ValueError(f"Failed to parse CSV: {e}")

    if df.empty or len(df.columns) == 0:
        filepath.unlink(missing_ok=True)
        dataset_dir.rmdir()
        raise ValueError("Dataset is empty (no rows or columns)")
    if len(df) > MAX_DATASET_ROWS:
        filepath.unlink(missing_ok=True)
        dataset_dir.rmdir()
        raise ValueError(f"Dataset rows ({len(df)}) exceed the configured limit ({MAX_DATASET_ROWS})")

    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {dupes}")

    info = {
        "id": dataset_id,
        "filename": filename,
        "filepath": str(filepath),
        "rows": len(df),
        "columns": df.columns.tolist(),
        "num_columns": len(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_values": {c: int(df[c].isnull().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
        "validation": {"status": "passed", "checks": ["csv_extension", "upload_size", "non_empty_file", "csv_parse", "non_empty_dataset", "unique_column_names"]},
    }
    _datasets[dataset_id] = info
    _save_state()
    log.info("Dataset uploaded: %s (%d rows, %d cols)", dataset_id, info["rows"], info["num_columns"])
    return info


def import_playground_dataset(filename: str, frame: pd.DataFrame, provenance: dict) -> dict:
    """Materialize one validated Playground version as an ML dataset reference."""
    if frame.empty or not len(frame.columns):
        raise ValueError("Dataset is empty (no rows or columns)")
    if len(frame) > MAX_DATASET_ROWS:
        raise ValueError(f"Dataset rows ({len(frame)}) exceed the configured limit ({MAX_DATASET_ROWS})")
    if frame.columns.duplicated().any() or any(not str(c).strip() for c in frame.columns):
        raise ValueError("Dataset has invalid or duplicate column names")
    dataset_id = uuid.uuid4().hex[:12]
    dataset_dir = UPLOAD_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filepath = dataset_dir / (Path(filename).stem + ".csv")
    try:
        frame.to_csv(filepath, index=False)
    except Exception:
        filepath.unlink(missing_ok=True)
        raise
    info = {
        "id": dataset_id, "filename": filename, "filepath": str(filepath),
        "rows": len(frame), "columns": [str(c) for c in frame.columns],
        "num_columns": len(frame.columns), "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "missing_values": {str(c): int(frame[c].isna().sum()) for c in frame.columns},
        "duplicate_rows": int(frame.duplicated().sum()), "version": 1,
        "provenance": provenance,
        "validation": {"status": "passed", "checks": ["playground_version_resolved", "non_empty_dataset", "unique_column_names"]},
    }
    _datasets[dataset_id] = info
    _save_state()
    return info


def get_dataset(dataset_id: str) -> dict:
    if dataset_id not in _datasets:
        raise ValueError(f"Dataset not found: {dataset_id}")
    return _datasets[dataset_id]


def get_column_statistics(dataset_id: str, target: str | None = None) -> dict:
    """Return dataset-grounded column evidence and conservative role suggestions."""
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    rows = len(df)
    result = []
    for column in df.columns:
        series = df[column]
        unique = int(series.nunique(dropna=True))
        ratio = unique / rows if rows else 0.0
        missing_pct = float(series.isna().mean() * 100) if rows else 0.0
        examples = [None if pd.isna(value) else str(value)[:100] for value in series.dropna().head(3).tolist()]
        duplicated_of = next((other for other in df.columns[:df.columns.get_loc(column)]
                              if series.equals(df[other])), None)
        reasons = []
        if column == target:
            role, reasons = "target", ["Selected as the prediction target"]
        elif unique <= 1:
            role, reasons = "review_remove", ["Column is constant or contains only missing values"]
        elif duplicated_of:
            role, reasons = "review_remove", [f"Values duplicate column '{duplicated_of}'"]
        elif missing_pct >= 80:
            role, reasons = "review_remove", [f"{missing_pct:.1f}% of values are missing"]
        elif ratio >= 0.98 and unique > 20:
            role, reasons = "review_remove", [f"High cardinality ({unique}/{rows} distinct values); inspect for identifier or time semantics"]
        elif unique / max(rows, 1) <= 0.005:
            role, reasons = "review", ["Near-constant distribution; predictive usefulness needs review"]
        else:
            role, reasons = "retain", ["No strong metadata-only reason to remove this column"]
        result.append({"name": column, "dtype": str(series.dtype),
                       "missing_pct": round(missing_pct, 2), "unique_count": unique,
                       "unique_ratio": round(ratio, 4), "example_values": examples,
                       "suggested_role": role, "recommendation_reasons": reasons,
                       "duplicate_of": duplicated_of, "is_target": column == target})
    return {"dataset_id": dataset_id, "rows": rows, "dataset_version": ds.get("version", 1), "columns": result}


def apply_column_selection(workspace_id: str, retained_columns: list[str], target: str | None = None) -> dict:
    """Create a versioned transformed CSV and make it the workspace's dataset."""
    ws = get_workspace(workspace_id)
    configured_target = ws.get("target")
    if configured_target and target and configured_target != target:
        raise ValueError(f"Selection target '{target}' differs from configured target '{configured_target}'")
    target = configured_target or target
    source_id = ws.get("original_dataset_id") or ws["dataset_id"]
    source = get_dataset(source_id)
    original_columns = source["columns"]
    if not retained_columns or len(set(retained_columns)) != len(retained_columns):
        raise ValueError("Select at least one unique column to retain")
    unknown = [column for column in retained_columns if column not in original_columns]
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}")
    if target and target not in retained_columns:
        raise ValueError("The target column must be retained")
    if target and not [column for column in retained_columns if column != target]:
        raise ValueError("At least one usable feature besides the target must remain")
    df = pd.read_csv(source["filepath"], usecols=retained_columns)[retained_columns]
    removed = [column for column in original_columns if column not in retained_columns]
    metadata = {item["name"]: item for item in get_column_statistics(source_id, target).get("columns", [])}
    removal_reasons = {
        column: "Removed by user selection. " + (
            "; ".join(metadata[column]["recommendation_reasons"])
            if metadata.get(column, {}).get("suggested_role") in {"review_remove", "review"}
            else "No strong metadata-only reason to remove it was detected."
        ) for column in removed
    }
    selection = {"retained_columns": retained_columns, "removed_columns": removed, "target": target,
                 "source_dataset_id": source_id, "removal_reasons": removal_reasons}
    fingerprint = hashlib.sha256(json.dumps(selection, sort_keys=True).encode()).hexdigest()[:12]
    transformed_id = f"{source_id[:6]}{fingerprint}"
    transformed_path = Path(source["filepath"]).parent / f"selection_{fingerprint}.csv"
    if not transformed_path.exists():
        df.to_csv(transformed_path, index=False)
    transformed = {**source, "id": transformed_id, "filepath": str(transformed_path),
                   "parent_dataset_id": source_id, "version": source.get("version", 1) + 1,
                   "rows": len(df), "columns": retained_columns, "num_columns": len(retained_columns),
                   "dtypes": {column: str(df[column].dtype) for column in retained_columns},
                   "missing_values": {column: int(df[column].isna().sum()) for column in retained_columns},
                   "duplicate_rows": int(df.duplicated().sum()), "selection_fingerprint": fingerprint}
    _datasets[transformed_id] = transformed
    ws["original_dataset_id"] = source_id
    ws["dataset_id"] = transformed_id
    ws["dataset_version"] = transformed["version"]
    ws["column_selection"] = {**selection, "dataset_id": transformed_id, "fingerprint": fingerprint}
    ws.update(status="created", target=None, problem_type=None, preprocessing={}, model_id=None,
              model_params={}, metrics=None, artifacts_dir=None, run_id=None, optimization=None,
              training_error=None)
    _save_state()
    return {"workspace_id": workspace_id, "original_dataset_id": source_id,
            "dataset_id": transformed_id, "dataset_version": transformed["version"],
            "rows": len(df), "original_columns": original_columns,
            "retained_columns": retained_columns, "removed_columns": removed,
            "removal_reasons": removal_reasons,
            "schema": transformed["dtypes"], "fingerprint": fingerprint}


def restore_column_selection(workspace_id: str) -> dict:
    ws = get_workspace(workspace_id)
    original_id = ws.get("original_dataset_id")
    if not original_id:
        return {"workspace_id": workspace_id, "dataset_id": ws["dataset_id"], "restored": False}
    original = get_dataset(original_id)
    ws["dataset_id"] = original_id
    ws["dataset_version"] = original.get("version", 1)
    ws.pop("column_selection", None)
    ws.pop("original_dataset_id", None)
    ws.update(status="created", target=None, problem_type=None, preprocessing={}, model_id=None,
              model_params={}, metrics=None, artifacts_dir=None, run_id=None, optimization=None,
              training_error=None)
    _save_state()
    return {"workspace_id": workspace_id, "dataset_id": original_id, "restored": True,
            "columns": original["columns"], "dataset_version": original.get("version", 1)}


def get_preview(dataset_id: str, rows: int | None = None) -> dict:
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    n = min(rows or MAX_PREVIEW_ROWS, len(df))
    preview = df.head(n)
    return {
        "columns": df.columns.tolist(),
        "data": preview.where(preview.notna(), None).to_dict(orient="records"),
        "total_rows": len(df),
    }


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def get_analysis(dataset_id: str) -> dict:
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    stats = []
    total_rows = len(df)
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        missing_pct = round(missing / total_rows * 100, 2) if total_rows > 0 else 0
        warning = None
        if missing_pct > 50:
            warning = "High missingness (>50%)"
        elif missing_pct > 30:
            warning = "Moderate missingness (>30%)"
            
        s = {"name": col, "dtype": str(df[col].dtype),
             "missing": missing, "missing_pct": missing_pct, "unique": int(df[col].nunique())}
        if warning:
            s["warning"] = warning
            
        if col in numerical:
            d = df[col].describe()
            s.update(mean=_safe_float(d.get("mean")), std=_safe_float(d.get("std")),
                     min=_safe_float(d.get("min")), max=_safe_float(d.get("max")),
                     q25=_safe_float(d.get("25%")), q50=_safe_float(d.get("50%")),
                     q75=_safe_float(d.get("75%")))
        if col in categorical:
            top = df[col].value_counts().head(10).to_dict()
            s["top_values"] = {str(k): int(v) for k, v in top.items()}
        stats.append(s)

    correlation = None
    if len(numerical) >= 2:
        corr_df = df[numerical].corr()
        correlation = {
            "columns": numerical,
            "matrix": corr_df.round(4).values.tolist()
        }

    dataset_size_bytes = Path(ds['filepath']).stat().st_size

    return {
        "rows": len(df), "columns": len(df.columns),
        "numerical_columns": numerical, "categorical_columns": categorical,
        "missing_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_stats": stats,
        "correlation": correlation,
        "dataset_size_bytes": dataset_size_bytes,
    }


def get_correlation(dataset_id: str) -> dict:
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numerical) < 2:
        return {"columns": numerical, "matrix": [], "message": "Not enough numerical columns for correlation"}
    corr_df = df[numerical].corr()
    return {
        "columns": numerical,
        "matrix": corr_df.round(4).values.tolist()
    }


def generate_profiling_report(dataset_id: str) -> dict:
    """Generate YData profiling report as HTML artifact."""
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    
    try:
        from ydata_profiling import ProfileReport
        profile = ProfileReport(
            df,
            title=f"Dataset Profile: {ds['filename']}",
            minimal=True,  # Faster, less memory for large datasets
            explorative=False,
            progress_bar=False,
        )
        
        # Save report as HTML in the upload directory
        report_dir = UPLOAD_DIR / dataset_id
        report_path = report_dir / "profiling_report.html"
        profile.to_file(report_path)
        
        log.info("Profiling report generated: %s", dataset_id)
        return {
            "dataset_id": dataset_id,
            "report_available": True,
            "report_url": f"/api/datasets/{dataset_id}/profiling/report",
        }
    except Exception as e:
        log.warning("Profiling failed for %s: %s", dataset_id, e)
        return {
            "dataset_id": dataset_id,
            "report_available": False,
            "error": str(e),
        }


def get_vif(dataset_id: str) -> dict:
    """Compute Variance Inflation Factor for numerical features."""
    ds = get_dataset(dataset_id)
    df = pd.read_csv(ds["filepath"])
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    
    if len(numerical) < 2:
        return {"features": [], "message": "Not enough numerical columns for VIF analysis"}
    
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    # Drop rows with NaN for VIF computation
    num_df = df[numerical].dropna()
    if len(num_df) < 2:
        return {"features": [], "message": "Not enough valid rows for VIF analysis"}
    
    # Add constant for VIF calculation
    from statsmodels.tools.tools import add_constant
    X = add_constant(num_df)
    
    vif_data = []
    for i, col in enumerate(numerical):
        try:
            vif_val = variance_inflation_factor(X.values, i + 1)  # +1 because of constant
            warning = None
            if vif_val > 10:
                warning = "High multicollinearity (VIF > 10)"
            elif vif_val > 5:
                warning = "Moderate multicollinearity (VIF > 5)"
            vif_data.append({
                "feature": col,
                "vif": round(float(vif_val), 4) if not np.isinf(vif_val) else None,
                "warning": warning
            })
        except Exception:
            vif_data.append({"feature": col, "vif": None, "warning": "Could not compute VIF"})
    
    return {"features": vif_data}


# ──────────────────────────── Workspace ops ────────────────────────────

def create_workspace(dataset_id: str) -> dict:
    get_dataset(dataset_id)  # validate
    ws_id = uuid.uuid4().hex[:12]
    ws = {
        "id": ws_id, "status": "created", "dataset_id": dataset_id,
        "target": None, "problem_type": None, "preprocessing": {},
        "model_id": None, "model_params": {}, "metrics": None,
        "artifacts_dir": None, "run_id": None, "runs": [],
        "optimization": None,
    }
    _workspaces[ws_id] = ws
    _save_state()
    log.info("Workspace created: %s (dataset=%s)", ws_id, dataset_id)
    return ws


def get_workspace(workspace_id: str) -> dict:
    if workspace_id not in _workspaces:
        raise ValueError(f"Workspace not found: {workspace_id}")
    return _workspaces[workspace_id]


def workspaces_for_dataset(dataset_id: str) -> list[str]:
    return [workspace_id for workspace_id, workspace in _workspaces.items() if workspace.get("dataset_id") == dataset_id]


def mark_training_failed(workspace_id: str, error: str) -> None:
    ws = get_workspace(workspace_id)
    ws.update(status="failed", training_error=error)
    _save_state()


def configure_workspace(workspace_id: str, target: str, problem_type: str,
                        preprocessing: dict, model_id: str, model_params: dict) -> dict:
    ws = get_workspace(workspace_id)
    ds = get_dataset(ws["dataset_id"])

    # --- validate target ---
    if target not in ds["columns"]:
        raise ValueError(f"Target '{target}' not in dataset columns: {ds['columns']}")
    features = [c for c in ds["columns"] if c != target]
    if not features:
        raise ValueError("Dataset must have at least one feature column besides the target")

    # load to inspect target column
    df = pd.read_csv(ds["filepath"])
    tcol = df[target]
    if tcol.nunique() < 2:
        raise ValueError(f"Target '{target}' has fewer than 2 unique values")

    # --- validate problem type ---
    if problem_type not in ("classification", "regression"):
        raise ValueError(f"Unsupported problem type: {problem_type}")
    if problem_type == "regression" and not pd.api.types.is_numeric_dtype(tcol):
        raise ValueError(f"Target '{target}' must be numeric for regression")

    # --- validate model ---
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_id}")
    if MODEL_REGISTRY[model_id]["problem_type"] != problem_type:
        raise ValueError(
            f"Model '{model_id}' is for {MODEL_REGISTRY[model_id]['problem_type']}, not {problem_type}")
    create_model(model_id, model_params)  # dry-run param validation

    # --- validate preprocessing ---
    mv = preprocessing.get("missing_values", "mean")
    if mv not in ("mean", "median", "mode", "drop"):
        raise ValueError(f"Invalid missing_values strategy: {mv}")
    sc = preprocessing.get("scaling")
    if sc not in (None, "standard", "minmax"):
        raise ValueError(f"Invalid scaling: {sc}")
    if preprocessing.get("smote") and problem_type != "classification":
        raise ValueError("SMOTE is only available for classification problems")

    ws.update(target=target, problem_type=problem_type, preprocessing=preprocessing,
              model_id=model_id, model_params=model_params, status="configured",
              metrics=None, artifacts_dir=None, run_id=None, optimization=None)
    _save_state()
    log.info("Workspace configured: %s (target=%s, model=%s)", workspace_id, target, model_id)
    return ws


# ──────────────────────────── Training ────────────────────────────

def train_workspace(
    workspace_id: str,
    *,
    job_id: str | None = None,
    progress_callback=None,
) -> dict:
    ws = get_workspace(workspace_id)
    ds = get_dataset(ws["dataset_id"])
    estimate = {"rows": ds["rows"], "features": max(0, ds["num_columns"] - 1),
                "dataset_bytes": Path(ds["filepath"]).stat().st_size,
                "estimated_memory_bytes": ds["rows"] * max(1, ds["num_columns"] - 1) * 16}
    enforce(estimate, "train_workspace", {})
    with job_slot():
        started = time.perf_counter()
        result = _train_workspace(
            workspace_id,
            job_id=job_id or uuid.uuid4().hex,
            progress_callback=progress_callback,
        )
        from config import MAX_TRAINING_DURATION_SECONDS
        elapsed = time.perf_counter() - started
        if elapsed > MAX_TRAINING_DURATION_SECONDS:
            raise ResourceLimitError(f"Training exceeded the configured duration limit ({MAX_TRAINING_DURATION_SECONDS} seconds)")
        return result


def _train_workspace(workspace_id: str, *, job_id: str | None = None, progress_callback=None) -> dict:
    training_started = time.perf_counter()
    job_id = job_id or uuid.uuid4().hex

    def report(stage: str, message: str) -> None:
        if progress_callback:
            progress_callback(stage, message)

    report("PREPROCESSING", "Reading the dataset and preparing the training split.")
    ws = get_workspace(workspace_id)
    if ws["status"] not in ("configured", "trained"):
        raise ValueError(f"Configure the workspace before training (status={ws['status']})")

    ds = get_dataset(ws["dataset_id"])
    df = pd.read_csv(ds["filepath"])

    # 1. Separate target
    target = ws["target"]
    X = df.drop(columns=[target])
    y = df[target]

    # 2. Drop rows with missing values if selected (before split, affects both)
    if ws["preprocessing"].get("missing_values") == "drop":
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        if len(X) == 0:
            raise ValueError("No rows remaining after dropping missing values")

    # 3. Train/test split — BEFORE any preprocessing fitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 4. Column types (from training data)
    numerical_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_columns = X_train.columns.tolist()

    # 5. Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numerical_cols, categorical_cols, ws["preprocessing"])

    # 6. Fit on train ONLY, transform both
    if preprocessor is not None:
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
    else:
        X_train_proc = X_train.values
        X_test_proc = X_test.values
    metadata_train_rows = len(X_train)

    # 6b. Apply SMOTE if enabled (ONLY on training data, ONLY for classification)
    if ws["preprocessing"].get("smote") and ws["problem_type"] == "classification":
        from imblearn.over_sampling import SMOTE as SMOTEResampler
        smote = SMOTEResampler(random_state=RANDOM_STATE)
        try:
            X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
            log.info("SMOTE applied: %d -> %d training samples", metadata_train_rows, len(X_train_proc))
        except Exception as e:
            log.warning("SMOTE failed (proceeding without): %s", e)

    # 7. Create and train model
    model = create_model(ws["model_id"], ws["model_params"])
    ws["status"] = "training"
    log.info("Training started: %s", workspace_id)
    report("TRAINING", "Fitting the selected model on the prepared training data.")
    model.fit(X_train_proc, y_train)

    # 8. Evaluate on test set
    report("EVALUATING", "Evaluating the fitted model on the held-out test data.")
    y_pred = model.predict(X_test_proc)
    y_proba = None
    if ws["problem_type"] == "classification" and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test_proc)
        except Exception:
            pass
    metrics = compute_metrics(y_test, y_pred, ws["problem_type"], y_proba=y_proba)

    # 9. Save artifacts
    run_id = uuid.uuid4().hex[:12]
    workspace_artifacts = ARTIFACTS_DIR / workspace_id
    workspace_artifacts.mkdir(parents=True, exist_ok=True)
    art_dir = workspace_artifacts / run_id
    staging_dir = workspace_artifacts / f".training-{job_id}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "workspace_id": workspace_id, "run_id": run_id,
        "dataset_id": ws["dataset_id"], "target": target,
        "problem_type": ws["problem_type"], "model_id": ws["model_id"],
        "model_params": ws["model_params"], "preprocessing": ws["preprocessing"],
        "feature_columns": feature_columns,
        "numerical_columns": numerical_cols, "categorical_columns": categorical_cols,
        "train_rows": metadata_train_rows, "test_rows": len(X_test),
        "has_preprocessor": preprocessor is not None,
        "smote_applied": ws["preprocessing"].get("smote", False) and ws["problem_type"] == "classification",
    }
    config_data = {
        "workspace_id": workspace_id,
        "run_id": run_id,
        "dataset_id": ws["dataset_id"],
        "target": target,
        "problem_type": ws["problem_type"],
        "model_id": ws["model_id"],
        "model_params": ws["model_params"],
        "preprocessing": ws["preprocessing"],
        "metrics": metrics,
    }

    try:
        joblib.dump(model, staging_dir / "model.joblib")
        if preprocessor is not None:
            joblib.dump(preprocessor, staging_dir / "preprocessor.joblib")
        joblib.dump(metadata, staging_dir / "metadata.joblib")
        joblib.dump(metrics, staging_dir / "metrics.joblib")
        pred_df = pd.DataFrame({"actual": y_test.values, "predicted": y_pred})
        pred_df.to_csv(staging_dir / "test_predictions.csv", index=False)
        with open(staging_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        staging_dir.replace(art_dir)
    except Exception:
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    run_entry = {
        "run_id": run_id,
        "status": "COMPLETED",
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_id": ws["model_id"],
        "model_params": ws["model_params"],
        "preprocessing": ws["preprocessing"],
        "metrics": metrics,
        "train_rows": metadata_train_rows,
        "test_rows": len(X_test),
        "feature_columns": feature_columns,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "training_duration_seconds": round(time.perf_counter() - training_started, 4),
        "smote_applied": metadata["smote_applied"],
        "artifacts_dir": str(art_dir),
    }
    ws.setdefault("runs", []).append(run_entry)

    ws.update(status="trained", metrics=metrics, artifacts_dir=str(art_dir), run_id=run_id)
    _save_state()
    log.info("Training complete: %s (run=%s)", workspace_id, run_id)

    return {"workspace_id": workspace_id, "run_id": run_id, "status": "trained", "metrics": metrics}


def optimize_workspace(workspace_id: str, req_data: dict) -> dict:
    """Run Optuna hyperparameter optimization without test set data leakage."""
    from config import MAX_OPTIMIZATION_TRIALS
    requested_trials = req_data.get("n_trials") or req_data.get("n_iter")
    if requested_trials and int(requested_trials) > MAX_OPTIMIZATION_TRIALS:
        raise ResourceLimitError(f"Optimization trials ({requested_trials}) exceed the configured limit ({MAX_OPTIMIZATION_TRIALS})")
    ws = get_workspace(workspace_id); ds = get_dataset(ws["dataset_id"])
    estimate = {"rows": ds["rows"], "features": max(0, ds["num_columns"] - 1),
                "dataset_bytes": Path(ds["filepath"]).stat().st_size,
                "estimated_memory_bytes": ds["rows"] * max(1, ds["num_columns"] - 1) * 16}
    enforce(estimate, "optimize_workspace", req_data)
    with job_slot():
        return _optimize_workspace(workspace_id, req_data)


def _optimize_workspace(workspace_id: str, req_data: dict) -> dict:
    """Internal optimization implementation under the shared job slot."""
    ws = get_workspace(workspace_id)
    if not ws.get("target") or not ws.get("problem_type"):
        raise ValueError("Workspace must have target and problem_type configured before optimization")

    ds = get_dataset(ws["dataset_id"])
    df = pd.read_csv(ds["filepath"])

    target = ws["target"]
    X = df.drop(columns=[target])
    y = df[target]

    if ws["preprocessing"].get("missing_values") == "drop":
        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]
        if len(X) == 0:
            raise ValueError("No rows remaining after dropping missing values")

    # Strict Train/Test split BEFORE optimization
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    candidate_models = req_data.get("candidate_models")
    if not candidate_models:
        if ws.get("model_id"):
            candidate_models = [ws["model_id"]]
        else:
            candidate_models = [
                m["model_id"] for m in MODEL_REGISTRY.values()
                if m["problem_type"] == ws["problem_type"]
            ]

    metric = req_data.get("metric") or ("accuracy" if ws["problem_type"] == "classification" else "rmse")
    n_trials = int(req_data.get("n_trials", 15))
    cv_folds = int(req_data.get("cv_folds", 3))
    timeout = req_data.get("timeout", 60)

    from optimizer import run_optuna_optimization, run_bayesian_optimization, run_sklearn_search
    strategy = req_data.get("strategy", "optuna")
    if strategy not in ("optuna", "bayesian", "grid", "random"):
        raise ValueError("strategy must be one of: optuna, bayesian, grid, random")
    if strategy in ("grid", "random"):
        model_id = ws.get("model_id") or (req_data.get("candidate_models") or [None])[0]
        if not model_id:
            raise ValueError("Select a model before running GridSearchCV or RandomizedSearchCV")
        search_result = run_sklearn_search(X_train, y_train, ws["problem_type"], ws["preprocessing"], model_id,
                                            strategy, req_data.get("param_grid"), req_data.get("n_iter", 10), cv_folds, RANDOM_STATE)
        configure_workspace(workspace_id, target, ws["problem_type"], ws["preprocessing"], model_id, search_result["best_params"])
        train_result = train_workspace(workspace_id)
        ws["optimization"] = search_result
        _save_state()
        return {"workspace_id": workspace_id, "run_id": train_result["run_id"], "optimization": search_result, "metrics": train_result["metrics"], "status": "trained"}
    optimizer_fn = run_bayesian_optimization if strategy == "bayesian" else run_optuna_optimization
    opt_result = optimizer_fn(
        X_train=X_train,
        y_train=y_train,
        problem_type=ws["problem_type"],
        preprocessing_config=ws["preprocessing"],
        candidate_models=candidate_models,
        metric=metric,
        n_trials=n_trials,
        cv_folds=cv_folds,
        timeout=timeout,
        random_state=RANDOM_STATE,
    )
    if strategy == "optuna":
        opt_result["strategy"] = "optuna"
        opt_result["optimizer"] = "optuna_tpe"

    # Reconfigure workspace with best model and best params
    configure_workspace(
        workspace_id,
        target=target,
        problem_type=ws["problem_type"],
        preprocessing=ws["preprocessing"],
        model_id=opt_result["best_model_id"],
        model_params=opt_result["best_params"],
    )

    # This function already holds the single job slot acquired by
    # optimize_workspace. Calling the public train_workspace wrapper here would
    # attempt to acquire it a second time and reject every optimization.
    train_result = _train_workspace(workspace_id)

    # Attach optimization result to workspace state and artifacts
    ws["optimization"] = opt_result
    _save_state()
    if ws.get("artifacts_dir"):
        opt_path = Path(ws["artifacts_dir"]) / "optuna_results.json"
        with open(opt_path, "w", encoding="utf-8") as f:
            json.dump(opt_result, f, indent=2)

    return {
        "workspace_id": workspace_id,
        "run_id": train_result["run_id"],
        "optimization": opt_result,
        "metrics": train_result["metrics"],
        "status": "trained",
    }


def get_workspace_optimization(workspace_id: str) -> dict | None:
    ws = get_workspace(workspace_id)
    return ws.get("optimization")


def get_workspace_runs(workspace_id: str) -> list[dict]:
    """Retrieve all training runs for a workspace."""
    ws = get_workspace(workspace_id)
    return ws.get("runs", [])


# ──────────────────────────── Evaluation ────────────────────────────

def get_evaluation(workspace_id: str) -> dict:
    ws = get_workspace(workspace_id)
    if ws["status"] != "trained":
        raise ValueError("No training results. Train the model first.")
    return {"problem_type": ws["problem_type"], "metrics": ws["metrics"]}


# ──────────────────────────── Prediction ────────────────────────────

def _safe_json(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val


def predict_workspace(workspace_id: str, filename: str, content: bytes) -> dict:
    ws = get_workspace(workspace_id)
    if ws["status"] != "trained":
        raise ValueError("Model must be trained before prediction")

    art_dir = Path(ws["artifacts_dir"])
    model = joblib.load(art_dir / "model.joblib")
    metadata = joblib.load(art_dir / "metadata.joblib")

    preprocessor = None
    if metadata["has_preprocessor"]:
        preprocessor = joblib.load(art_dir / "preprocessor.joblib")

    feature_columns = metadata["feature_columns"]

    try:
        pred_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise ValueError(f"Failed to parse prediction CSV: {e}")

    missing = [c for c in feature_columns if c not in pred_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    warnings = []
    unexpected = [c for c in pred_df.columns if c not in feature_columns]
    if unexpected:
        warnings.append(f"Unexpected columns ignored: {unexpected}")

    # Verify numerical columns can be converted to numeric
    for col in metadata.get("numerical_columns", []):
        if col in pred_df.columns and not pd.api.types.is_numeric_dtype(pred_df[col]):
            try:
                pred_df[col] = pd.to_numeric(pred_df[col])
            except Exception:
                raise ValueError(f"Column '{col}' must be numeric, but contains non-numeric values")

    X_pred = pred_df[feature_columns]

    # Transform — NOT fit_transform
    if preprocessor is not None:
        X_pred_proc = preprocessor.transform(X_pred)
    else:
        X_pred_proc = X_pred.values

    predictions = model.predict(X_pred_proc)
    preds_list = [_safe_json(p) for p in predictions]

    result_df = pred_df.copy()
    result_df["prediction"] = predictions
    result_df.to_csv(art_dir / "predictions.csv", index=False)

    log.info("Prediction complete: %s (%d rows)", workspace_id, len(predictions))
    return {
        "predictions": preds_list,
        "rows": len(predictions),
        "download_url": f"/api/workspaces/{workspace_id}/artifacts/predictions.csv",
        "warnings": warnings,
    }


# ──────────────────────────── Artifacts ────────────────────────────

def get_artifacts(workspace_id: str) -> list[dict]:
    ws = get_workspace(workspace_id)
    if not ws.get("artifacts_dir"):
        return []
    art_dir = Path(ws["artifacts_dir"])
    if not art_dir.exists():
        return []
    def _art_type(name: str, suffix: str) -> str:
        if name == "config.json":
            return "config"
        if name == "optuna_results.json":
            return "optimization"
        if suffix == ".joblib":
            return "model"
        return "data"

    return [
        {"filename": f.name,
         "type": _art_type(f.name, f.suffix),
         "url": f"/api/workspaces/{workspace_id}/artifacts/{f.name}"}
        for f in sorted(art_dir.iterdir()) if f.is_file()
    ]
