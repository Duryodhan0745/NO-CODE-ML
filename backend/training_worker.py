from __future__ import annotations

import json
import logging
import sys

import services


def emit(event: dict) -> None:
    print(json.dumps(event, default=str), flush=True)


def main() -> int:
    workspace_id, job_id = sys.argv[1:3]
    try:
        result = services.train_workspace(
            workspace_id,
            job_id=job_id,
            progress_callback=lambda stage, message: emit(
                {"type": "progress", "stage": stage, "message": message}
            ),
        )
        emit({"type": "completed", "result": result})
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Training worker failed for job %s", job_id)
        emit({"type": "failed", "error": str(exc), "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
