"""Port for job lifecycle + queue operations. Only the methods real callers
use are here — JobService (the adapter) has far more (Mongo/Redis-backed
internals, background refresh threads); those stay infra-internal. The
lifecycle bootstrap methods main.py calls at startup
(start_processing_times_refresh_thread, refresh_service_processing_times)
are composition-root wiring, not part of this port either.

# ponytail: status/priority are typed Any instead of api.schemas' Status/JobPriority
# enums so domain doesn't import from api (api.schemas -> domain is the correct
# direction, not the reverse). Those enums are plain str-Enums with zero framework
# dependency, so relocating them to domain would be the fuller fix — out of scope
# here given how many files already import them from api.schemas.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class IJobRepository(Protocol):
    """Create, track, and manage the lifecycle of async generation jobs."""

    def create_job(
        self,
        user_id: str,
        service_type: str,
        request_data: Dict[str, Any],
        priority: Any = None,
    ) -> Dict[str, Any]:
        """Create a new job and add it to the queue."""

    def update_job_status(
        self,
        job_id: str,
        status: Any,
        result_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        celery_task_id: Optional[str] = None,
    ) -> bool:
        """Update job status and related result/error info."""

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get a job's current status and details."""

    def update_job_request_data(self, job_id: str, request_data_updates: Dict[str, Any]) -> bool:
        """Patch fields under a job's stored request_data (e.g. url backfilled after async conversion)."""

    def get_user_jobs(self, user_id: str, limit: int = 0, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """List jobs belonging to a user."""

    def get_user_queue_lanes_snapshot(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """Snapshot of a user's position/state across all queue lanes."""

    def calculate_wait_time_for_job(self, job_id: str, job_doc: Optional[Dict[str, Any]] = None) -> float:
        """Estimate remaining wait time for a queued job."""

    def lock_job(self, job_id: str, user_id: str, lock_duration: int = 36000) -> bool:
        """Acquire a processing lock on a job."""

    def unlock_job(self, job_id: str, user_id: str) -> bool:
        """Release a processing lock on a job."""

    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a job on behalf of a user."""

    def retry_failed_jobs(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Re-enqueue failed jobs, up to an optional limit."""

    def retry_failed_job(self, job_id: str) -> Dict[str, Any]:
        """Re-enqueue one specific failed job."""

    def synchronize_jobs_async(self) -> Dict[str, Any]:
        """Kick off async reconciliation between queue state and job records."""
