import copy
import threading
import time
import uuid
from typing import Any, Dict, Optional


class UploadJobManager:
    def __init__(self, ttl_seconds: int = 60 * 60) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()

    def create_job(self, *, filename: str, document_id: str) -> Dict[str, Any]:
        now = time.time()
        task_id = uuid.uuid4().hex
        job = {
            "task_id": task_id,
            "status": "queued",
            "stage": "queued",
            "message": "文件已上传，等待开始处理...",
            "progress_percent": 15.0,
            "stage_current": 0,
            "stage_total": 0,
            "filename": filename,
            "document_id": document_id,
            "page_count": 0,
            "timing": {},
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "revision": 0,
        }
        with self._lock:
            self._cleanup_locked(now)
            self._jobs[task_id] = job
            return copy.deepcopy(job)

    def get_job(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            self._cleanup_locked(now)
            job = self._jobs.get(task_id)
            if job is None:
                return None
            return copy.deepcopy(job)

    def update_job(self, task_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            self._cleanup_locked(now)
            job = self._jobs.get(task_id)
            if job is None:
                return None

            timing_update = fields.pop("timing", None)
            if isinstance(timing_update, dict):
                merged_timing = dict(job.get("timing") or {})
                merged_timing.update(timing_update)
                job["timing"] = merged_timing

            job.update(fields)
            job["updated_at"] = now
            job["revision"] = int(job.get("revision", 0)) + 1
            return copy.deepcopy(job)

    def delete_job(self, task_id: str) -> None:
        with self._lock:
            self._jobs.pop(task_id, None)

    def _cleanup_locked(self, now: float) -> None:
        expired_task_ids = [
            task_id
            for task_id, job in self._jobs.items()
            if now - float(job.get("updated_at", now)) > self._ttl_seconds
        ]
        for task_id in expired_task_ids:
            self._jobs.pop(task_id, None)
