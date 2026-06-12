from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Callable
from uuid import uuid4

from .config import settings
from .utils import to_jsonable


_jobs: dict[str, dict] = {}
_lock = Lock()
_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(input_filename: str, input_path: str) -> str:
    job_id = uuid4().hex

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "stage": "Waiting to start",
            "progress": 0,
            "input_filename": input_filename,
            "input_path": input_path,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }

    return job_id


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def update_job(job_id: str, **updates) -> None:
    with _lock:
        if job_id not in _jobs:
            return

        _jobs[job_id].update(to_jsonable(updates))
        _jobs[job_id]["updated_at"] = utc_now_iso()


def submit_processing_job(
    job_id: str,
    processing_function: Callable,
    *args,
    **kwargs,
) -> None:

    def runner() -> None:
        try:
            update_job(
                job_id,
                status="processing",
                stage="Starting processing",
                progress=1,
            )

            result = processing_function(
                job_id,
                *args,
                progress_callback=lambda **data: update_job(job_id, **data),
                **kwargs,
            )

            update_job(
                job_id,
                status="completed",
                stage="Completed",
                progress=100,
                result=result,
            )
        except Exception as exc:
            update_job(
                job_id,
                status="failed",
                stage="Failed",
                progress=100,
                error=str(exc),
            )

    _executor.submit(runner)
