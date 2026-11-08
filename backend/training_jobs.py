from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import services
from config import ARTIFACTS_DIR, MAX_CONCURRENT_JOBS, STATE_DIR

log = logging.getLogger(__name__)
_JOBS_FILE = STATE_DIR / "training_jobs.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TrainingJobManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(_JOBS_FILE.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            log.exception("Could not load persisted training job states")
            return
        for job in data:
            if job.get("status") not in {"COMPLETED", "FAILED", "CANCELLED"}:
                job.update(status="FAILED", stage="FAILED", message="Training process was interrupted by a server restart.", updated_at=_now())
            self._jobs[job["job_id"]] = job
        self._persist()

    def _persist(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = _JOBS_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(list(self._jobs.values()), default=str), encoding="utf-8")
        temporary.replace(_JOBS_FILE)

    def _update(self, job_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            job.update(changes, updated_at=_now())
            self._persist()
            return dict(job)

    def _update_progress(self, job_id: str, stage: str, message: str) -> None:
        with self._lock:
            if self._jobs[job_id]["status"] == "CANCELLING":
                return
            self._jobs[job_id].update(status=stage, stage=stage, message=message, updated_at=_now())
            self._persist()

    def start(self, workspace_id: str, owner_id: str) -> dict[str, Any]:
        with self._lock:
            active = [job for job in self._jobs.values() if job["status"] in {
                "INITIALIZING", "PREPROCESSING", "TRAINING", "EVALUATING", "CANCELLING"
            }]
            if len(active) >= MAX_CONCURRENT_JOBS:
                raise RuntimeError("The configured training job limit has been reached.")
            if any(job["workspace_id"] == workspace_id for job in active):
                raise RuntimeError("A training job is already active for this workspace.")
            workspace = services.get_workspace(workspace_id)
            dataset = services.get_dataset(workspace["dataset_id"])
            job_id = uuid.uuid4().hex
            now = _now()
            job = {
                "job_id": job_id,
                "workspace_id": workspace_id,
                "owner_id": owner_id,
                "dataset_id": dataset["id"],
                "dataset_name": dataset["filename"],
                "model_id": workspace.get("model_id"),
                "status": "INITIALIZING",
                "stage": "INITIALIZING",
                "message": "Starting an isolated training worker.",
                "started_at": now,
                "updated_at": now,
                "finished_at": None,
                "result": None,
                "error": None,
            }
            self._jobs[job_id] = job
            self._persist()
            thread = threading.Thread(target=self._run, args=(job_id,), daemon=True, name=f"training-{job_id[:8]}")
            self._threads[job_id] = thread
            thread.start()
            return dict(job)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = dict(self._jobs[job_id])
        command = [sys.executable, "-m", "training_worker", job["workspace_id"], job_id]
        environment = os.environ.copy()
        backend_dir = Path(__file__).resolve().parent
        environment["PYTHONPATH"] = str(backend_dir) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            with self._lock:
                if self._jobs[job_id]["status"] == "CANCELLING":
                    self._update(
                        job_id,
                        status="CANCELLED",
                        stage="CANCELLED",
                        message="Training was cancelled before the worker started.",
                        finished_at=_now(),
                    )
                    return
                process = subprocess.Popen(
                    command,
                    cwd=backend_dir,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                self._processes[job_id] = process
            final_event = None
            assert process.stdout is not None
            for line in process.stdout:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Ignoring malformed training progress from job %s", job_id)
                    continue
                if event.get("type") == "progress":
                    self._update_progress(job_id, event["stage"], event["message"])
                elif event.get("type") in {"completed", "failed"}:
                    final_event = event
            return_code = process.wait()
            with self._lock:
                cancelling = self._jobs[job_id]["status"] == "CANCELLING"
            if cancelling:
                self._cleanup_staging(job_id, job["workspace_id"])
                self._update(job_id, status="CANCELLED", stage="CANCELLED", message="Training process stopped.", finished_at=_now())
            elif return_code == 0 and final_event and final_event.get("type") == "completed":
                services.load_state()
                self._update(
                    job_id,
                    status="COMPLETED",
                    stage="COMPLETED",
                    message="Training and evaluation completed.",
                    result=final_event.get("result"),
                    finished_at=_now(),
                )
            else:
                error = (final_event or {}).get("error") or f"Training worker exited with code {return_code}."
                log.error("Training job %s failed: %s", job_id, error)
                try:
                    services.mark_training_failed(job["workspace_id"], error)
                except (KeyError, ValueError):
                    log.exception("Could not persist failed training state for job %s", job_id)
                self._update(
                    job_id,
                    status="FAILED",
                    stage="FAILED",
                    message="Training failed. Your configuration is preserved; review the error and retry.",
                    error=error,
                    finished_at=_now(),
                )
        except Exception as exc:
            log.exception("Could not run training job %s", job_id)
            self._update(
                job_id,
                status="FAILED",
                stage="FAILED",
                message="Training could not be started. Your configuration is preserved; retry when ready.",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_now(),
            )
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
                self._threads.pop(job_id, None)

    @staticmethod
    def _cleanup_staging(job_id: str, workspace_id: str) -> None:
        staging = ARTIFACTS_DIR / workspace_id / f".training-{job_id}"
        if staging.is_dir():
            import shutil
            shutil.rmtree(staging)

    def get(self, job_id: str, owner_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job and job["owner_id"] == owner_id else None

    def cancel(self, job_id: str, owner_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job["owner_id"] != owner_id:
                return None
            if job["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return dict(job)
            if job["status"] == "CANCELLING":
                return dict(job)
            process = self._processes.get(job_id)
            if process and process.poll() is not None:
                return dict(job)
            job.update(status="CANCELLING", stage="CANCELLING", message="Stopping the isolated training process.", updated_at=_now())
            self._persist()
            if process and process.poll() is None:
                process.terminate()
        return self.get(job_id, owner_id)

    def shutdown(self) -> None:
        with self._lock:
            active_ids = [job_id for job_id, job in self._jobs.items() if job["status"] not in {"COMPLETED", "FAILED", "CANCELLED"}]
            threads = [self._threads.get(job_id) for job_id in active_ids]
            for job_id in active_ids:
                process = self._processes.get(job_id)
                if process and process.poll() is not None:
                    continue
                self._jobs[job_id].update(status="CANCELLING", stage="CANCELLING", message="Stopping training because the server is shutting down.", updated_at=_now())
                if process and process.poll() is None:
                    process.terminate()
            self._persist()
        for thread in threads:
            if thread and thread is not threading.current_thread():
                thread.join(timeout=10)


training_jobs = TrainingJobManager()
