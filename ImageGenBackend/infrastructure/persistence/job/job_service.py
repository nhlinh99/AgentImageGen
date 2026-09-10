"""
Job Service Module
Handles job creation, status management, and lifecycle operations
"""

import uuid
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from common.base_services import BaseServiceSingleton
from infrastructure.persistence.mongo.mongo_service import MongoService
from infrastructure.persistence.mongo.repositories.diffusion_repository import MongoDiffusionJobs
from infrastructure.cache.redis.redis import RedisClient
from infrastructure.cache.redis.utils import delete_keys_with_prefix
from infrastructure.persistence.job.queue_service import QueueService
from infrastructure.persistence.job.queue_lane import lane_for_service_type, LANE_FAST, LANE_MEDIUM, LANE_SLOW, QueueLane
from api.schemas.model import JobPriority, JobDocument
from api.schemas.data.job_models import Status
from config.settings import Settings
import pytz

class JobService(BaseServiceSingleton):
    """
    Service for managing job lifecycle and operations
    """

    def __init__(self, config: Settings = Settings()):
        super(JobService, self).__init__(config)
        self.config = config
        self.mongo_service = MongoService(self.config)
        self.redis_client = RedisClient(self.config)
        self.queue_service = QueueService(self.config)

        # MongoDB collection
        self.diffusion_jobs = MongoDiffusionJobs(self.mongo_service, config)

        # Default ETA time in seconds (1 minutes)
        self.default_eta_seconds = 60

        # Timezone setting (GMT+7 for Vietnam/Bangkok)
        self.timezone = pytz.timezone('Asia/Ho_Chi_Minh')  # GMT+7

        # Cached service processing times refreshed periodically in the background
        self._service_processing_times_cache: Dict[str, float] = {}
        self._service_processing_times_lock = threading.Lock()
        self._service_processing_times_last_updated: float = 0.0
        self._service_processing_times_stop_event = threading.Event()
        self._service_processing_times_thread: Optional[threading.Thread] = None

        # Prime cache immediately to avoid cold-start latency

        # State tracking for asynchronous job synchronization
        self._synchronize_jobs_lock = threading.Lock()
        self._synchronize_jobs_thread: Optional[threading.Thread] = None
        self._last_synchronize_jobs_result: Optional[Dict[str, Any]] = None
        self._last_synchronize_jobs_error: Optional[str] = None

    def clear_pipeline_redis_for_job(self, job_id: str) -> None:
        """
        Remove Redis keys for a finished pipeline: legacy per-job task hash (if present)
        and intermediate keys written by tasks (``delete_keys_with_prefix('*', job_id)``).

        Task status lives in MongoDB only; this only cleans auxiliary Redis data.
        """
        if not job_id:
            return
        tasks_hash = f"{self.config.redis.prefix_key}:job:{job_id}:tasks"
        try:
            self.redis_client.delete(tasks_hash)
        except Exception as e:
            self.logger.warning('Could not delete legacy Redis task hash for job %s: %s', job_id, e)
        try:
            delete_keys_with_prefix("*", job_id)
        except Exception as e:
            self.logger.warning('Could not delete Redis pipeline keys for job %s: %s', job_id, e)

    def _resolve_tasks_for_job(self, job_id: str, job_doc: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        doc = job_doc if job_doc is not None else self._get_job_from_mongodb(job_id)
        if not doc:
            return []
        return doc.get("tasks") or []

    def get_current_time(self) -> str:
        """
        Get current time in GMT+7 timezone as ISO format string

        Returns:
            ISO format datetime string in GMT+7
        """
        return datetime.now(self.timezone).isoformat()

    def parse_datetime(self, datetime_str: str) -> datetime:
        """
        Parse ISO format datetime string and convert to GMT+7 timezone

        Args:
            datetime_str: ISO format datetime string

        Returns:
            Datetime object in GMT+7 timezone
        """
        dt = datetime.fromisoformat(datetime_str)
        if dt.tzinfo is None:
            # If no timezone info, assume it's already in GMT+7
            dt = self.timezone.localize(dt)
        else:
            # Convert to GMT+7 if it has timezone info
            dt = dt.astimezone(self.timezone)
        return dt

    def create_job(self, user_id: str,
                    service_type: str,
                    request_data: Dict[str, Any],
                    priority: JobPriority = JobPriority.NORMAL) -> Dict[str, Any]:
        """
        Create a new job and add it to the queue

        Args:
            user_id: User identifier
            service_type: Type of AI service (tryon, faceswap, humangen, background_swap)
            request_data: Request data for the job
            priority: Job priority level

        Returns:
            Job information including job_id and queue position
        """
        try:
            # Extract job_id from request_data
            job_id = request_data.get("job_id")
            if not job_id:
                raise ValueError("job_id is required in request_data")

            queue_lane = lane_for_service_type(service_type)
            # Create job document (parse_obj: basedpyright does not infer optional Field defaults on __init__)
            job_doc = JobDocument.parse_obj(
                {
                    "job_id": job_id,
                    "user_id": user_id,
                    "service_type": service_type,
                    "queue_lane": queue_lane,
                    "status": Status.QUEUED,
                    "priority": priority,
                    "request_data": request_data,
                    "created_at": self.get_current_time(),
                    "max_retries": 3,
                    "tasks": [],
                }
            )

            # Save to MongoDB
            self._save_job_to_mongodb(job_doc.dict())

            # Add to queue
            self.queue_service.add_job_to_queue(
                job_id=job_id,
                priority=priority,
                service_type=service_type,
                user_id=user_id,
            )

            st = time.time()
            queue_position = self.queue_service.get_queue_position(job_id)
            num_job_processing = self.queue_service.get_processing_jobs_count(queue_lane)
            self.logger.info('Time get_queue_position: %s', time.time() - st)

            # Lane-local: waiting rank within this service's lane + jobs processing in that lane only
            first_queue_position = (queue_position if queue_position is not None else 0) + num_job_processing
            self._update_job_in_mongodb(job_id, {"first_queue_position": first_queue_position})

            # Calculate estimated wait time
            st = time.time()
            estimated_wait_time = self.calculate_wait_time_for_job(job_id)
            self.logger.info('Time calculate_wait_time_for_job: %s', time.time() - st)

            # Queue position is tracked in Redis; first_queue_position is persisted for analytics / wait estimates

            self.logger.info('Job created successfully: %s, position: %s, estimated_wait: %ss', job_id, queue_position, estimated_wait_time)

            return {
                "job_id": job_id,
                "status": Status.QUEUED.value,
                "queue_position": first_queue_position,
                "first_queue_position": first_queue_position,
                "estimated_wait_time": estimated_wait_time,
                "created_at": job_doc.created_at
            }

        except Exception as e:
            self.logger.error('Error creating job: %s', str(e))
            raise

    def update_job_request_data(self, job_id: str, request_data_updates: Dict[str, Any]) -> bool:
        """Patch fields under a job's stored request_data (dot-path $set, e.g. list_image -> new urls)."""
        return self._update_job_in_mongodb(
            job_id, {f"request_data.{k}": v for k, v in request_data_updates.items()}
        )

    def _apply_job_queue_fields_to_doc(self, job_doc: Dict[str, Any]) -> None:
        """Attach is_locked, queue_position, and processing_order to job_doc (mutates in place)."""
        jid = job_doc.get("job_id")
        if not jid:
            return
        is_locked = self.queue_service.is_job_locked(jid)
        queue_position = -1
        if job_doc["status"] == Status.QUEUED.value and not is_locked:
            qp = self.queue_service.get_queue_position(jid)
            queue_position = qp if qp is not None else -1

        # Processing jobs are removed from waiting ZSETs — they live only in the lane's processing ZSET.
        # Clients otherwise see queue_position -1 despite Redis still tracking the job; expose a 1-based
        # slot within that lane's active set so UIs stay consistent with queue_service state.
        processing_order = -1
        if job_doc["status"] == Status.PROCESSING.value:
            po = self.queue_service.get_processing_job_order(jid)
            processing_order = po if po is not None else -1
            if po is not None and po >= 0:
                queue_position = po + 1
            else:
                queue_position = 1

        job_doc["is_locked"] = is_locked
        job_doc["queue_position"] = queue_position
        job_doc["processing_order"] = processing_order

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and queue information

        Args:
            job_id: Job identifier

        Returns:
            Job status information
        """
        try:
            # Get job from MongoDB
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            self._apply_job_queue_fields_to_doc(job_doc)

            return job_doc

        except Exception as e:
            self.logger.error('Error getting job status: %s', str(e))
            raise

    def update_job_status(self, job_id: str, status: Status,
                         result_data: Optional[Dict[str, Any]] = None,
                         error_message: Optional[str] = None,
                         celery_task_id: Optional[str] = None) -> bool:
        """
        Update job status and related information

        Args:
            job_id: Job identifier
            status: New job status
            result_data: Result data if job completed
            error_message: Error message if job failed
            celery_task_id: Celery task ID if processing

        Returns:
            Success status
        """
        try:
            time_current = self.get_current_time()
            update_data = {
                "status": status.value,
                "updated_at": time_current
            }
            if status == Status.PROCESSING:
                update_data["started_at"] = time_current
                if celery_task_id:
                    update_data["celery_task_id"] = celery_task_id

            elif status == Status.COMPLETED:
                update_data["completed_at"] = time_current
                if result_data is not None:
                    update_data["result_data"] = result_data

            elif status == Status.FAILED:
                update_data["completed_at"] = time_current
                if error_message:
                    update_data["error_message"] = error_message

            elif status == Status.CANCELLED:
                update_data["cancelled_at"] = time_current
                if error_message:
                    update_data["cancellation_reason"] = error_message

            job_doc = self._get_job_from_mongodb(job_id)

            # Update MongoDB
            self._update_job_in_mongodb(job_id, update_data)

            # Update processing queue (lane derived from service_type — single source of truth)
            if status == Status.PROCESSING:
                service_type = job_doc.get("service_type", "unknown") if job_doc else "unknown"
                qlane = lane_for_service_type(service_type)
                owner_id = str(job_doc.get("user_id") or "") if job_doc else ""
                self.queue_service.add_job_to_processing_queue(job_id, service_type, qlane, owner_id)
            elif status in [Status.COMPLETED, Status.FAILED, Status.CANCELLED]:
                self.queue_service.remove_job_from_processing_queue(job_id, None)

            # Remove from queue if processing started or job completed/failed/cancelled
            if status in [Status.PROCESSING, Status.COMPLETED, Status.FAILED, Status.CANCELLED]:
                self.queue_service.remove_job_from_queue(job_id)

            self.logger.info('Job %s status updated to %s', job_id, status.value)
            return True

        except Exception as e:
            self.logger.error('Error updating job status: %s', str(e))
            return False

    def update_task_id(self, job_id: str, celery_task_id: str) -> bool:
        """
        Update celery_task_id information

        Args:
            job_id: Job identifier
            celery_task_id: task identifier of full pipeline

        Returns:
            Success status
        """
        try:
            status = self._update_job_in_mongodb(job_id=job_id, update_data={"celery_task_id": celery_task_id})

            self.logger.info('celery_task_id: %s updated to %s', celery_task_id, status)
            return True

        except Exception as e:
            self.logger.error('Error updating celery_task_id: %s', str(e))
            return False

    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """
        Cancel a job if it's still pending or queued

        Args:
            job_id: Job identifier
            user_id: User identifier for authorization

        Returns:
            Success status
        """
        try:
            # Get job and verify ownership
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            # Update status to cancelled with cancellation message
            cancellation_message = f"Job cancelled by user: {user_id}"
            self.update_job_status(job_id, Status.CANCELLED, error_message=cancellation_message)
            self.clear_pipeline_redis_for_job(job_id)

            return True

        except Exception as e:
            self.logger.error('Error cancelling job: %s', str(e))
            return False

    def delete_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """
        Delete a job from MongoDB and remove from all queues

        Args:
            job_id: Job identifier
            user_id: Optional user identifier for authorization

        Returns:
            Success status
        """
        try:
            # Get job to verify it exists
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            # Optional: Verify ownership if user_id is provided
            if user_id and job_doc.get("user_id") != user_id:
                raise ValueError(f"User {user_id} is not authorized to delete job {job_id}")

            # Remove from Redis queues (waiting queue, processing queue, and locked jobs)
            self.queue_service.remove_job_from_queue(job_id)
            self.queue_service.remove_job_from_processing_queue(job_id)

            # If job is locked, unlock it first
            if self.queue_service.is_job_locked(job_id):
                self.queue_service.unlock_job(job_id)

            # Delete from MongoDB
            collection = self.diffusion_jobs.collection
            result = collection.delete_one({"job_id": job_id})

            if result.deleted_count > 0:
                self.logger.info('Job %s deleted successfully', job_id)
                return True
            else:
                self.logger.warning('Job %s not found in MongoDB', job_id)
                return False

        except Exception as e:
            self.logger.error('Error deleting job %s: %s', job_id, str(e))
            return False

    def get_user_jobs(self, user_id: str, limit: int = 0, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """
        Get jobs for a specific user with optional additional query conditions

        Args:
            user_id: User identifier
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            **kwargs: Additional query conditions (e.g., service_type, status, locked)

        Returns:
            List of job information
        """
        try:
            collection = self.diffusion_jobs.collection

            # Build query with user_id and additional conditions
            query = {"user_id": user_id}

            # Add additional query conditions from kwargs
            for key, value in kwargs.items():
                if value is not None:
                    query[key] = value

            cursor = collection.find(query).sort("created_at", -1).skip(offset).limit(limit)

            jobs = []
            for job_doc in cursor:
                job_doc["_id"] = str(job_doc["_id"])  # Convert ObjectId to string
                jobs.append(job_doc)

            return jobs

        except Exception as e:
            self.logger.error('Error getting user jobs: %s', str(e))
            return []

    def get_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get all jobs with a specific status

        Args:
            status: Job status to filter by (e.g., 'queued', 'processing', 'completed', 'failed')

        Returns:
            List of job information
        """
        try:
            collection = self.diffusion_jobs.collection

            # Query for jobs with the specified status
            query = {"status": status}

            cursor = collection.find(query).sort("created_at", 1)  # Sort by oldest first

            jobs = []
            for job_doc in cursor:
                job_doc["_id"] = str(job_doc["_id"])  # Convert ObjectId to string
                jobs.append(job_doc)

            self.logger.info("Found %s jobs with status '%s'", len(jobs), status)
            return jobs

        except Exception as e:
            self.logger.error("Error getting jobs by status '%s': %s", status, str(e))
            return []

    def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get job statistics for monitoring

        Returns:
            Job statistics
        """
        try:
            collection = self.diffusion_jobs.collection

            # Count by status
            status_counts = {}
            for status in Status:
                count = collection.count_documents({"status": status.value})
                status_counts[status.value] = count

            # Count by service type
            service_counts = {}
            pipeline = [
                {"$group": {"_id": "$service_type", "count": {"$sum": 1}}}
            ]
            service_stats = list(collection.aggregate(pipeline))
            for stat in service_stats:
                service_counts[stat["_id"]] = stat["count"]

            # Queue statistics
            queue_stats = self.queue_service.get_queue_statistics()

            # Active jobs count
            active_jobs_count = self.queue_service.get_processing_jobs_count()

            return {
                "status_counts": status_counts,
                "service_counts": service_counts,
                "queue_stats": queue_stats,
                "active_jobs_count": active_jobs_count,
                "total_jobs": sum(status_counts.values())
            }

        except Exception as e:
            self.logger.error('Error getting job statistics: %s', str(e))
            return {}

    def get_processing_jobs_count(self) -> int:
        """
        Get current count of active (processing) jobs

        Returns:
            Number of currently active jobs
        """
        try:
            return self.queue_service.get_processing_jobs_count()
        except Exception as e:
            self.logger.error('Error getting active jobs count: %s', str(e))
            return 0

    def get_user_queue_lanes_snapshot(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        For one user: waiting/processing counts in each Redis lane ZSET, plus max ETA (seconds)
        across that user's active jobs in that lane from ``calculate_wait_time_for_job``.
        Uses one Mongo query and one Redis pipeline for lane counts; ETA uses ZRANK + prefix ZRANGE.
        """
        uid = str(user_id)
        empty: Dict[str, Any] = {"waiting": 0, "processing": 0, "estimated_wait_seconds": 0.0}
        try:
            counts_by_lane = self.queue_service.count_user_jobs_all_lanes(uid)
            max_eta: Dict[QueueLane, float] = {
                LANE_FAST: 0.0,
                LANE_MEDIUM: 0.0,
                LANE_SLOW: 0.0,
            }
            collection = self.diffusion_jobs.collection
            query = {
                "user_id": uid,
                "status": {"$in": [Status.QUEUED.value, Status.PROCESSING.value]},
            }
            projection = {"job_id": 1, "service_type": 1, "user_id": 1}
            for doc in collection.find(query, projection):
                jid = doc.get("job_id")
                if not jid:
                    continue
                st = doc.get("service_type") or ""
                ln = lane_for_service_type(st)
                try:
                    w = float(self.calculate_wait_time_for_job(jid, job_doc=doc))
                    if w > max_eta[ln]:
                        max_eta[ln] = w
                except Exception:
                    continue

            out: Dict[str, Dict[str, Any]] = {}
            for key, ln in (
                ("fast", LANE_FAST),
                ("medium", LANE_MEDIUM),
                ("slow", LANE_SLOW),
            ):
                row = counts_by_lane[ln]
                out[key] = {
                    "waiting": row["waiting"],
                    "processing": row["processing"],
                    "estimated_wait_seconds": round(max_eta[ln], 2),
                }
            return out
        except Exception as e:
            self.logger.error('Error counting lane jobs for user %s: %s', user_id, str(e))
            return {"fast": dict(empty), "medium": dict(empty), "slow": dict(empty)}


    def _save_job_to_mongodb(self, job_data: Dict[str, Any]) -> bool:
        """Save job document to MongoDB"""
        try:
            collection = self.diffusion_jobs.collection
            result = collection.insert_one(job_data)
            return result.inserted_id is not None
        except Exception as e:
            self.logger.error('Error saving job to MongoDB: %s', str(e))
            return False

    def _get_job_from_mongodb(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job document from MongoDB"""
        try:
            collection = self.diffusion_jobs.collection
            job_doc = collection.find_one({"job_id": job_id})
            if job_doc:
                job_doc["_id"] = str(job_doc["_id"])
            return job_doc
        except Exception as e:
            self.logger.error('Error getting job from MongoDB: %s', str(e))
            return None

    def _update_job_in_mongodb(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Update job document in MongoDB"""
        try:
            collection = self.diffusion_jobs.collection
            result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error('Error updating job in MongoDB: %s', str(e))
            return False


    def retry_failed_jobs(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Retry all failed jobs by resetting their state and re-queueing them.

        Args:
            limit: Optional limit on the number of failed jobs to retry.

        Returns:
            Summary dictionary with retry statistics.
        """
        try:
            failed_jobs = self.get_jobs_by_status(Status.FAILED.value)

            if limit is not None and limit > 0:
                failed_jobs = failed_jobs[:limit]

            if not failed_jobs:
                return {
                    "total_failed_jobs": 0,
                    "retried_jobs": [],
                    "skipped_jobs": [],
                    "failed_jobs": []
                }

            retried_jobs = []
            skipped_jobs = []
            failed_retries = []

            for job_doc in failed_jobs:
                job_id = job_doc.get("job_id")
                service_type = job_doc.get("service_type")

                if not job_id or not service_type:
                    skipped_jobs.append({
                        "job_id": job_id,
                        "reason": "Missing job_id or service_type"
                    })
                    continue

                try:
                    retry_result = self._retry_single_failed_job(job_doc)
                    retried_jobs.append(retry_result)
                except Exception as exc:
                    self.logger.error('Failed to retry job %s: %s', job_id, exc)
                    failed_retries.append({
                        "job_id": job_id,
                        "error": str(exc)
                    })

            return {
                "total_failed_jobs": len(failed_jobs),
                "retried_count": len(retried_jobs),
                "skipped_count": len(skipped_jobs),
                "failed_retry_count": len(failed_retries),
                "retried_jobs": retried_jobs,
                "skipped_jobs": skipped_jobs,
                "failed_jobs": failed_retries
            }
        except Exception as exc:
            self.logger.error('Error retrying failed jobs: %s', exc)
            raise

    def retry_failed_job(self, job_id: str) -> Dict[str, Any]:
        """
        Retry a specific failed job.

        Args:
            job_id: Identifier of the job to retry.

        Returns:
            Information about the retried job.
        """
        try:
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")

            if job_doc.get("status") != Status.FAILED.value:
                raise ValueError(f"Job {job_id} is not in failed status")

            return self._retry_single_failed_job(job_doc)
        except Exception as exc:
            self.logger.error('Error retrying failed job %s: %s', job_id, exc)
            raise

    def _retry_single_failed_job(self, job_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reset a failed job and requeue it.

        Args:
            job_doc: MongoDB job document representing a failed job.

        Returns:
            Information about the retried job.
        """
        job_id = job_doc["job_id"]
        service_type = job_doc.get("service_type")
        priority_value = job_doc.get("priority", JobPriority.NORMAL.value)
        try:
            priority = priority_value if isinstance(priority_value, JobPriority) else JobPriority(priority_value)
        except ValueError:
            priority = JobPriority.NORMAL

        current_time = self.get_current_time()

        # Reset task states
        tasks = job_doc.get("tasks") or []
        reset_tasks = []
        for task in tasks:
            reset_tasks.append({
                "task_name": task.get("task_name"),
                "task_order": task.get("task_order"),
                "status": Status.QUEUED.value,
                "created_at": task.get("created_at") or current_time,
                "updated_at": current_time,
                "celery_task_id": None,
                "result": None,
                "error": None,
                "time_process": None
            })

        update_data: Dict[str, Any] = {
            "status": Status.QUEUED.value,
            "updated_at": current_time,
            "error_message": None,
            "result_data": None,
            "celery_task_id": None,
            "started_at": None,
            "completed_at": None,
            "locked": False,
            "locked_by": None,
            "lock_expires_at": None,
            "cancelled_at": None,
            "cancellation_reason": None,
            "queue_lane": lane_for_service_type(service_type),
        }

        if reset_tasks:
            update_data["tasks"] = reset_tasks

        # Update MongoDB document
        self._update_job_in_mongodb(job_id, update_data)

        # Add job back to the queue
        self.queue_service.remove_job_from_queue(job_id)
        self.queue_service.remove_job_from_processing_queue(job_id)
        if self.queue_service.is_job_locked(job_id):
            self.queue_service.unlock_job(job_id)

        queue_position = self.queue_service.add_job_to_queue(
            job_id=job_id,
            priority=priority,
            service_type=service_type,
            user_id=str(job_doc.get("user_id") or ""),
        )

        self.logger.info('Job %s retried successfully', job_id)

        return {
            "job_id": job_id,
            "service_type": service_type,
            "priority": priority.value,
            "queue_position": queue_position
        }

    def create_job_tasks(self, job_id: str, task_names: List[str]) -> bool:
        """
        Create task tracking entries for a job

        Args:
            job_id: Job identifier
            task_names: List of task names in the pipeline

        Returns:
            Success status
        """
        try:
            tasks = []
            for i, task_name in enumerate(task_names):
                task = {
                    "task_name": task_name,
                    "task_order": i,
                    "status": Status.QUEUED.value,  # Initial status for tasks
                    "created_at": self.get_current_time(),
                    "updated_at": self.get_current_time(),
                    "celery_task_id": None,
                    "result": None,
                    "error": None,
                    "time_process": None  # Initialize time_process field
                }
                tasks.append(task)

            # Update job with tasks
            update_data = {
                "tasks": tasks
            }

            success = self._update_job_in_mongodb(job_id, update_data)

            if success:
                self.logger.info('Created %s tasks for job %s', len(tasks), job_id)

            return success

        except Exception as e:
            self.logger.error('Error creating job tasks: %s', str(e))
            return False

    def _resolve_task_index_for_status_update(
        self,
        tasks: List[Dict[str, Any]],
        task_name: str,
        target_status: str,
        job_task_order: Optional[int] = None,
    ) -> Tuple[Optional[int], bool]:
        """
        Find a task row to update, or detect an idempotent no-op.

        Returns:
            (task_index, already_at_target): ``already_at_target`` is True when the
            matching task already has ``target_status`` (e.g. duplicate Celery success).
            ``task_index`` is None when no queued or matching terminal row exists.
        """
        def _matches(task: Dict[str, Any]) -> bool:
            if task.get("task_name") != task_name:
                return False
            if job_task_order is not None and task.get("task_order") != job_task_order:
                return False
            return True

        if job_task_order is not None:
            for idx, task in enumerate(tasks):
                if not _matches(task):
                    continue
                current = task.get("status")
                if current == Status.QUEUED.value:
                    return idx, False
                if current == target_status:
                    return idx, True
                return None, False

        indexed_by_pipeline = sorted(
            enumerate(tasks),
            key=lambda item: (item[1].get("task_order", item[0]), item[0]),
        )
        for orig_idx, task in indexed_by_pipeline:
            if not _matches(task):
                continue
            if task.get("status") == Status.QUEUED.value:
                return orig_idx, False
        for orig_idx, task in indexed_by_pipeline:
            if not _matches(task):
                continue
            if task.get("status") == target_status:
                return orig_idx, True
        return None, False

    def update_task_status(self, job_id: str, task_name: str, status: str,
                          celery_task_id: Optional[str] = None,
                          result: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None,
                          time_process: Optional[float] = None,
                          job_task_order: Optional[int] = None) -> bool:
        """
        Update task status for a specific task in a job (MongoDB ``jobs.tasks``).

        When the same Celery task name appears multiple times in a pipeline, pass
        job_task_order (must match tasks[].task_order from create_job_tasks). If
        omitted, the QUEUED task with the smallest task_order matching task_name
        is updated.

        Idempotent: if the matching task already has ``status``, returns True without
        writing (handles duplicate Celery success signals).

        Args:
            job_id: Job identifier
            task_name: Name of the task
            status: Task status (QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED)
            celery_task_id: Celery task ID
            result: Task result data
            error: Error message if task failed
            time_process: Time taken to process the task in seconds
            job_task_order: Pipeline step index for duplicate task names

        Returns:
            Success status
        """
        try:
            collection = self.diffusion_jobs.collection

            # First, check if the job exists
            job_doc = collection.find_one({"job_id": job_id})
            if not job_doc:
                self.logger.warning('Job %s not found', job_id)
                return False

            tasks = job_doc.get("tasks", [])
            if not tasks:
                self.logger.info('No tasks on job %s', job_id)
                return False

            task_index, already_at_target = self._resolve_task_index_for_status_update(
                tasks, task_name, status, job_task_order
            )

            if task_index is None:
                order_suffix = f" (job_task_order={job_task_order})" if job_task_order is not None else ""
                self.logger.info(
                    "Task %s not found as QUEUED in job %s%s", task_name, job_id, order_suffix
                )
                return False

            if already_at_target:
                order_suffix = f" (job_task_order={job_task_order})" if job_task_order is not None else ""
                self.logger.info(
                    "Task %s already %s in job %s%s; skipping duplicate status update",
                    task_name, status, job_id, order_suffix,
                )
                return True

            prefix = f"tasks.{task_index}."
            update_fields = {
                f"{prefix}status": status,
                f"{prefix}updated_at": self.get_current_time(),
            }

            if celery_task_id:
                update_fields[f"{prefix}celery_task_id"] = celery_task_id
            if result is not None:
                update_fields[f"{prefix}result"] = result
            if error is not None:
                update_fields[f"{prefix}error"] = error
            if time_process is not None:
                update_fields[f"{prefix}time_process"] = time_process

            update_result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_fields},
            )

            if update_result.modified_count > 0:
                self.logger.info('Updated task %s status to %s for job %s', task_name, status, job_id)
                if time_process is not None:
                    self.logger.info('Updated task %s time_process to %ss for job %s', task_name, time_process, job_id)

                return True
            else:
                self.logger.warning('No task %s found to update in job %s', task_name, job_id)
                return False

        except Exception as e:
            self.logger.error('Error updating task status: %s', str(e))
            return False

    def check_all_tasks_completed(self, job_id: str) -> bool:
        """
        Check if all tasks in a job are completed

        Args:
            job_id: Job identifier

        Returns:
            True if all tasks are completed, False otherwise
        """
        try:
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                self.logger.warning('Job %s not found', job_id)
                return False

            tasks = self._resolve_tasks_for_job(job_id, job_doc)
            if not tasks:
                self.logger.warning('No tasks found for job %s', job_id)
                return False

            total_tasks = len(tasks)
            completed_tasks = sum(1 for task in tasks if task.get("status") == Status.COMPLETED.value)
            failed_tasks = sum(1 for task in tasks if task.get("status") == Status.FAILED.value)

            # Check if all tasks are in final state (completed or failed)
            all_final = completed_tasks + failed_tasks == total_tasks

            if all_final:
                self.logger.info('All tasks completed for job %s: %s/%s completed, %s/%s failed', job_id, completed_tasks, total_tasks, failed_tasks, total_tasks)

                # If any task failed, update job status to failed
                if failed_tasks > 0:
                    self.update_job_status(
                        job_id=job_id,
                        status=Status.FAILED,
                        error_message=f"{failed_tasks} out of {total_tasks} tasks failed"
                    )
                    return False

                return True

            return False

        except Exception as e:
            self.logger.error('Error checking task completion for job %s: %s', job_id, str(e))
            return False

    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """
        Get job progress information

        Args:
            job_id: Job identifier

        Returns:
            Dictionary with progress information
        """
        try:
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                return {}

            tasks = self._resolve_tasks_for_job(job_id, job_doc)
            total_tasks = len(tasks)
            completed_tasks = sum(1 for task in tasks if task.get("status") == Status.COMPLETED.value)
            failed_tasks = sum(1 for task in tasks if task.get("status") == Status.FAILED.value)
            processing_tasks = sum(1 for task in tasks if task.get("status") == Status.PROCESSING.value)

            # Calculate total processing time from all tasks
            total_time_process = self.get_job_total_time_process(job_id)

            progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
            steps_finished_percentage = (
                round((completed_tasks + failed_tasks) / total_tasks * 100, 2)
                if total_tasks > 0
                else None
            )

            return {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "processing_tasks": processing_tasks,
                "progress_percentage": round(progress_percentage, 2),
                "steps_finished_percentage": steps_finished_percentage,
                "total_time_process": total_time_process,
                "is_completed": completed_tasks + failed_tasks == total_tasks and total_tasks > 0,
                "has_failures": failed_tasks > 0
            }

        except Exception as e:
            self.logger.error('Error getting job progress for %s: %s', job_id, str(e))
            return {}

    def get_job_total_time_process(self, job_id: str) -> float:
        """
        Get total processing time for a job by summing all task processing times

        Args:
            job_id: Job identifier

        Returns:
            Total processing time in seconds (rounded to 3 decimal places)
        """
        try:
            tasks = self._resolve_tasks_for_job(job_id)
            if not tasks:
                return 0.0

            # Sum all task processing times
            total_time = sum(
                task.get("time_process", 0) or 0
                for task in tasks
                if task.get("time_process") is not None
            )

            return round(total_time, 3)

        except Exception as e:
            self.logger.error('Error getting job total time process for %s: %s', job_id, str(e))
            return 0.0

    def lock_job(self, job_id: str, user_id: str, lock_duration: int = 36000) -> bool:
        """
        Lock a job by removing it from the active queue

        Args:
            job_id: Job identifier
            user_id: User requesting the lock (for authorization)
            lock_duration: Lock duration in seconds (default 1 hour)

        Returns:
            Success status
        """
        try:
            # Verify job exists and user has permission
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")


            # Only allow locking of queued jobs
            if job_doc["status"] not in [Status.QUEUED.value]:
                raise ValueError(f"Cannot lock job in status: {job_doc['status']}")

            # Get job priority and service type from the job document
            priority = JobPriority(job_doc.get("priority", JobPriority.NORMAL.value))
            service_type = job_doc.get("service_type", "tryon")

            # Lock job by removing from queue
            success = self.queue_service.lock_job(job_id, priority, service_type, lock_duration)

            if success:
                # Update job status to indicate it's locked
                current_time = datetime.now(self.timezone)
                self._update_job_in_mongodb(job_id, {
                    "locked": True,
                    "locked_at": self.get_current_time(),
                    "locked_by": user_id,
                    "lock_expires_at": (current_time + timedelta(seconds=lock_duration)).isoformat()
                })
                self.logger.info('Job %s locked by user %s for %s seconds', job_id, user_id, lock_duration)

            return success

        except Exception as e:
            self.logger.error('Error locking job: %s', str(e))
            return False

    def unlock_job(self, job_id: str, user_id: str) -> bool:
        """
        Unlock a job by adding it back to the active queue

        Args:
            job_id: Job identifier
            user_id: User requesting the unlock (for authorization)

        Returns:
            Success status
        """
        try:
            # Verify authorization
            job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                raise ValueError(f"Job not found: {job_id}")


            # Unlock job by adding back to queue
            success = self.queue_service.unlock_job(job_id)

            if success:
                # Update job status
                self._update_job_in_mongodb(job_id, {
                    "locked": False,
                    "unlocked_at": self.get_current_time(),
                    "locked_by": None,
                    "lock_expires_at": None
                })
                self.logger.info('Job %s unlocked by user %s', job_id, user_id)

            return success

        except Exception as e:
            self.logger.error('Error unlocking job: %s', str(e))
            return False

    def is_job_locked(self, job_id: str) -> bool:
        """
        Check if a job is currently locked

        Args:
            job_id: Job identifier

        Returns:
            True if job is locked, False otherwise
        """
        try:
            return self.queue_service.is_job_locked(job_id)
        except Exception as e:
            self.logger.error('Error checking lock status for job %s: %s', job_id, str(e))
            return False

    def get_processing_job_order(self, job_id: str) -> Optional[int]:
        """
        Get the order/position of a job in its processing lane only

        Args:
            job_id: Job identifier

        Returns:
            0-based rank within that lane's processing ZSET, or None if not processing
        """
        try:
            return self.queue_service.get_processing_job_order(job_id)
        except Exception as e:
            self.logger.error('Error getting processing job order for %s: %s', job_id, str(e))
            return None

    def get_processing_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get list of jobs currently in processing queue

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of processing job information
        """
        try:
            return self.queue_service.get_processing_jobs(limit)
        except Exception as e:
            self.logger.error('Error getting processing jobs: %s', str(e))
            return []

    def _calculate_estimated_wait_time(self, service_type: str, queue_position: Optional[int]) -> float:
        """
        Calculate estimated wait time for a job

        Args:
            service_type: Type of AI service
            queue_position: Current queue position

        Returns:
            Estimated wait time in seconds
        """
        wait_time_single = self._calculate_estimated_wait_time_single(service_type)
        total_wait_time = wait_time_single * queue_position
        return total_wait_time

    def _calculate_estimated_wait_time_single(self, service_type: str) -> float:
        """
        Calculate estimated wait time for a single job by averaging the sum of task processing times

        Args:
            service_type: Type of AI service

        Returns:
            Estimated wait time in seconds (average of total task processing times)
        """
        try:
            # Get historical average processing time
            collection = self.diffusion_jobs.collection

            # Get last 10 completed jobs of same service type with tasks
            pipeline = [
                {"$match": {
                    "service_type": service_type,
                    "status": Status.COMPLETED.value,
                    "tasks": {"$exists": True, "$ne": []}
                }},
                {"$sort": {"completed_at": -1}},
                {"$limit": 10}
            ]

            jobs = list(collection.aggregate(pipeline))

            if not jobs:
                # Use default wait time if no historical data
                return self.default_eta_seconds

            # Calculate average processing time by summing task times
            total_time = 0
            valid_jobs = 0

            for job in jobs:
                try:
                    tasks = job.get("tasks", [])
                    if not tasks:
                        continue

                    # Sum all task processing times for this job
                    job_total_time = sum(
                        task.get("time_process", 0) or 0
                        for task in tasks
                        if task.get("time_process") is not None
                    )

                    # Only include jobs with valid processing times (between 1 second and 1 hour)
                    if 1 <= job_total_time <= 3600:
                        total_time += job_total_time
                        valid_jobs += 1

                except (ValueError, KeyError, TypeError) as e:
                    self.logger.debug('Error processing job %s: %s', job.get('job_id', 'unknown'), e)
                    continue

            if valid_jobs == 0:
                # Use default wait time if no valid historical data
                self.logger.warning('No valid historical data for service type: %s', service_type)
                return self.default_eta_seconds

            avg_processing_time = total_time / valid_jobs

            self.logger.debug(
                "Average processing time for %s: %.2fs (based on %s jobs)",
                service_type, avg_processing_time, valid_jobs,
            )

            return avg_processing_time

        except Exception as e:
            self.logger.error('Error calculating estimated wait time: %s', str(e))
            return self.default_eta_seconds

    def _schedule_service_processing_times_refresh(self, interval_seconds: int = 7200) -> None:
        """Background worker that refreshes cached service processing times every given interval."""
        delay = 0
        while not self._service_processing_times_stop_event.wait(delay):
            try:
                self.refresh_service_processing_times()
            except Exception as exc:  # pragma: no cover - defensive logging
                self.logger.error('Failed to refresh service processing times: %s', exc)
            delay = interval_seconds

    def start_processing_times_refresh_thread(self, interval_seconds: int = 7200) -> None:
        """
        Start the background thread for refreshing service processing times.
        This should be called from the main application startup.

        Args:
            interval_seconds: Interval in seconds between refreshes (default: 7200 = 2 hours)
        """
        # Check if thread is already running
        if hasattr(self, '_service_processing_times_thread') and self._service_processing_times_thread is not None:
            if self._service_processing_times_thread.is_alive():
                self.logger.warning("Service processing times refresh thread is already running")
                return

        # Schedule periodic refresh in a dedicated daemon thread
        self._service_processing_times_thread = threading.Thread(
            target=self._schedule_service_processing_times_refresh,
            args=(interval_seconds,),
            name="job-service-processing-times",
            daemon=True,
        )
        self._service_processing_times_thread.start()
        self.logger.info('Started service processing times refresh thread (interval: %ss)', interval_seconds)

    def refresh_service_processing_times(self, initial_run: bool = False) -> None:
        """Refresh cached service processing times for known service types."""
        try:
            service_types = set()

            # Include configured service queues when available
            if hasattr(self.queue_service, "service_queues"):
                service_types.update(self.queue_service.service_queues)

            # Include service types currently present in waiting and processing queues
            waiting_jobs = self.queue_service.get_jobs_in_queue(limit=1000, queue_type="waiting")
            processing_jobs = self.queue_service.get_jobs_in_queue(limit=1000, queue_type="processing")

            for job_info in waiting_jobs + processing_jobs:
                service_type = job_info.get("service_type")
                if service_type:
                    service_types.add(service_type)

            if not service_types:
                return

            refreshed_times: Dict[str, float] = {}
            for service_type in service_types:
                refreshed_times[service_type] = self._calculate_estimated_wait_time_single(service_type)

            with self._service_processing_times_lock:
                self._service_processing_times_cache.update(refreshed_times)
                self._service_processing_times_last_updated = time.time()

            if initial_run:
                self.logger.debug("Initialized service processing time cache for types: %s", list(service_types))
            else:
                self.logger.debug("Refreshed service processing time cache for types: %s", list(service_types))

        except Exception as exc:
            self.logger.error('Error refreshing service processing times cache: %s', exc)

    def _get_cached_service_processing_time(self, service_type: str) -> float:
        """Return cached processing time for a service type, refreshing on cache miss."""
        with self._service_processing_times_lock:
            cached_value = self._service_processing_times_cache.get(service_type)

        if cached_value is not None:
            return cached_value

        # Fallback to direct calculation and update cache
        calculated_value = self._calculate_estimated_wait_time_single(service_type)

        with self._service_processing_times_lock:
            self._service_processing_times_cache[service_type] = calculated_value
            self._service_processing_times_last_updated = time.time()

        return calculated_value

    def calculate_wait_time_for_job(self, job_id: str, job_doc: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate estimated wait for this job using only its queue lane (fast / medium / slow).
        Lanes run in parallel—jobs in other lanes are not counted as ahead of this one.

        Uses Redis ZRANK + prefix ZRANGE on the lane ZSET (no full-queue scans when possible).
        Pass ``job_doc`` from callers that already loaded Mongo (avoids redundant find_one).

        Args:
            job_id: Job identifier to calculate wait time for
            job_doc: Optional Mongo job document (must include service_type, user_id when provided)

        Returns:
            Estimated wait time in seconds
        """
        try:
            if job_doc is None:
                job_doc = self._get_job_from_mongodb(job_id)
            if not job_doc:
                return float(self.default_eta_seconds)

            service_type = job_doc.get("service_type") or ""
            user_id = str(job_doc.get("user_id") or "")

            qt, lane, rank0 = self.queue_service.resolve_job_queue_state(job_id, service_type, user_id)
            if qt is None or lane is None or rank0 is None:
                return 0.0

            queue_jobs = self.queue_service.lane_queue_job_infos_prefix(qt, lane, rank0)
            if not queue_jobs:
                return 0.0

            service_processing_times: Dict[str, float] = {}
            total_wait_time = 0.0
            base_current_service_type = None

            for job_info in queue_jobs:
                current_job_id = job_info["job_id"]
                current_service_type = job_info["service_type"]

                if current_service_type not in service_processing_times:
                    service_processing_times[current_service_type] = self._get_cached_service_processing_time(
                        current_service_type
                    )

                single_job_wait_time = service_processing_times[current_service_type]
                total_wait_time += single_job_wait_time

                if current_job_id == job_id:
                    base_current_service_type = current_service_type
                    break

            if base_current_service_type is None:
                self.logger.warning('Job %s not found in %s queue during wait time computation', job_id, qt)
                return float(self.default_eta_seconds)

            if qt == "waiting":
                processing_jobs = self.queue_service.get_jobs_in_queue(
                    limit=1000, queue_type="processing", lane=lane
                )
                for pinfo in processing_jobs:
                    cst = pinfo["service_type"]
                    if cst not in service_processing_times:
                        service_processing_times[cst] = self._get_cached_service_processing_time(cst)
                    total_wait_time += service_processing_times.get(cst, self.default_eta_seconds)

            return round(total_wait_time, 2)

        except Exception as e:
            self.logger.error('Error calculating wait time for job %s: %s', job_id, str(e))
            return float(self.default_eta_seconds)

    def synchronize_jobs(self) -> Dict[str, Any]:
        """
        Synchronize jobs between MongoDB and Redis queue.
        This ensures that all QUEUED and PROCESSING jobs in MongoDB
        are properly reflected in the Redis queue.

        Steps:
        1. Get all jobs with status PROCESSING or QUEUED from MongoDB
        2. Check if each job_id is in the Redis queue
        3. If not in Redis queue, add the job_id to Redis

        Returns:
            Dictionary with synchronization statistics including:
            - total_jobs: Total number of jobs checked
            - synchronized_count: Number of jobs added to Redis
            - already_in_queue_count: Number of jobs already in queue
            - failed_count: Number of jobs that failed to synchronize
            - synchronized_jobs: List of jobs that were synchronized
            - already_in_queue: List of jobs already in queue
            - failed_jobs: List of jobs that failed to synchronize
        """
        try:
            # Step 1: Get all jobs with status QUEUED or PROCESSING
            queued_jobs = self.get_jobs_by_status(Status.QUEUED.value)
            processing_jobs = self.get_jobs_by_status(Status.PROCESSING.value)

            all_jobs = queued_jobs + processing_jobs

            if not all_jobs:
                self.logger.info("No jobs found with QUEUED or PROCESSING status for synchronization")
                return {
                    "total_jobs": 0,
                    "synchronized_count": 0,
                    "already_in_queue_count": 0,
                    "failed_count": 0,
                    "synchronized_jobs": [],
                    "already_in_queue": [],
                    "failed_jobs": []
                }

            # Step 2 & 3: Check each job in Redis queue and add if missing
            synchronized_jobs = []
            already_in_queue = []
            failed_jobs = []

            for job in all_jobs:
                job_id = job.get("job_id")
                service_type = job.get("service_type")
                priority = job.get("priority", JobPriority.NORMAL.value)
                status = job.get("status")
                is_locked = job.get("locked", False)

                try:
                    # Only synchronize QUEUED jobs that are not locked
                    if status == Status.QUEUED.value and not is_locked:
                        # Check if job exists in Redis queue
                        queue_position = self.queue_service.get_queue_position(job_id)

                        if queue_position is None:
                            # Job not in queue, add it
                            self.queue_service.add_job_to_queue(
                                job_id=job_id,
                                priority=JobPriority(priority),
                                service_type=service_type,
                                user_id=str(job.get("user_id") or ""),
                            )

                            new_position = self.queue_service.get_queue_position(job_id)

                            synchronized_jobs.append({
                                "job_id": job_id,
                                "service_type": service_type,
                                "priority": priority,
                                "status": status,
                                "queue_position": new_position,
                                "created_at": job.get("created_at")
                            })

                            self.logger.info('Synchronized job %s to Redis queue at position %s', job_id, new_position)
                        else:
                            # Job already in queue
                            already_in_queue.append({
                                "job_id": job_id,
                                "service_type": service_type,
                                "status": status,
                                "queue_position": queue_position
                            })
                    elif status == Status.QUEUED.value and is_locked:
                        # Locked jobs should not be in the queue
                        already_in_queue.append({
                            "job_id": job_id,
                            "service_type": service_type,
                            "status": status,
                            "locked": True
                        })
                    elif status == Status.PROCESSING.value:
                        # Processing jobs are already being handled
                        already_in_queue.append({
                            "job_id": job_id,
                            "service_type": service_type,
                            "status": status
                        })

                except Exception as e:
                    self.logger.error('Error synchronizing job %s: %s', job_id, str(e))
                    failed_jobs.append({
                        "job_id": job_id,
                        "error": str(e)
                    })

            self.logger.info('Job synchronization completed: %s synchronized, %s already in queue, %s failed', len(synchronized_jobs), len(already_in_queue), len(failed_jobs))

            return {
                "total_jobs": len(all_jobs),
                "synchronized_count": len(synchronized_jobs),
                "already_in_queue_count": len(already_in_queue),
                "failed_count": len(failed_jobs),
                "synchronized_jobs": synchronized_jobs,
                "already_in_queue": already_in_queue,
                "failed_jobs": failed_jobs
            }

        except Exception as e:
            self.logger.error('Error during job synchronization: %s', str(e))
            raise

    def synchronize_jobs_async(self) -> Dict[str, Any]:
        """
        Execute job synchronization in a background thread.

        Returns:
            Dictionary describing whether a new synchronization was started
            and metadata about any running thread.
        """

        def _run():
            thread_name = threading.current_thread().name
            try:
                result = self.synchronize_jobs()
                with self._synchronize_jobs_lock:
                    self._last_synchronize_jobs_result = result
                    self._last_synchronize_jobs_error = None
                self.logger.info('Job synchronization thread %s finished successfully: %s synchronized, %s already in queue, %s failed', thread_name, result.get('synchronized_count', 0), result.get('already_in_queue_count', 0), result.get('failed_count', 0))
            except Exception as exc:  # pragma: no cover - defensive
                with self._synchronize_jobs_lock:
                    self._last_synchronize_jobs_result = None
                    self._last_synchronize_jobs_error = str(exc)
                self.logger.error('Job synchronization thread %s failed: %s', thread_name, exc)
            finally:
                with self._synchronize_jobs_lock:
                    self._synchronize_jobs_thread = None

        with self._synchronize_jobs_lock:
            if self._synchronize_jobs_thread and self._synchronize_jobs_thread.is_alive():
                return {
                    "started": False,
                    "running": True,
                    "thread_name": self._synchronize_jobs_thread.name,
                    "last_result": self._last_synchronize_jobs_result,
                    "error": self._last_synchronize_jobs_error
                }

            thread_name = f"job-sync-{uuid.uuid4().hex[:8]}"
            thread = threading.Thread(target=_run, name=thread_name, daemon=True)
            self._synchronize_jobs_thread = thread
            thread.start()

            return {
                "started": True,
                "running": True,
                "thread_name": thread_name
            }
