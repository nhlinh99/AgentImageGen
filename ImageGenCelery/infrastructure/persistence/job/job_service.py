"""
Job Service Module
Handles job creation, status management, and lifecycle operations
"""

import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

from common_lib.base_services import BaseServiceSingleton
from infrastructure.persistence.mongo.mongo_service import MongoService
from infrastructure.persistence.mongo.mongo_domains import MongoJobs
from infrastructure.cache.redis.redis import RedisClient
from infrastructure.cache.redis.utils import delete_keys_with_prefix
from infrastructure.persistence.job.queue_service import QueueService
from infrastructure.persistence.job.queue_lane import lane_for_service_type
from domain.schema.job_models import Status
from config.config import Config
import pytz

class JobService(BaseServiceSingleton):
    """
    Service for managing job lifecycle and operations
    """
    
    def __init__(self, config: Config = Config()):
        super(JobService, self).__init__(config)
        self.config = config
        self.mongo_service = MongoService(self.config)
        self.mongo_jobs = MongoJobs(self.mongo_service, self.config)
        self.redis_client = RedisClient(self.config)
        self.queue_service = QueueService(self.config)

        # MongoDB collections
        self.jobs_collection = "jobs"
        self.jobs_database = config.mongo.database_name
        
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
            self.logger.warning(f"Could not delete legacy Redis task hash for job {job_id}: {e}")
        try:
            delete_keys_with_prefix("*", job_id)
        except Exception as e:
            self.logger.warning(f"Could not delete Redis pipeline keys for job {job_id}: {e}")

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
            self.logger.error(f"Error getting job status: {str(e)}")
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
                                
            self.logger.info(f"Job {job_id} status updated to {status.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating job status: {str(e)}")
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
                                
            self.logger.info(f"celery_task_id: {celery_task_id} updated to {status}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating celery_task_id: {str(e)}")
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
            self.logger.error(f"Error cancelling job: {str(e)}")
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
            collection = self.mongo_jobs.collection
            
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
            self.logger.error(f"Error getting user jobs: {str(e)}")
            return []
    
    
    def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get job statistics for monitoring
        
        Returns:
            Job statistics
        """
        try:
            collection = self.mongo_jobs.collection
            
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
            self.logger.error(f"Error getting job statistics: {str(e)}")
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
            self.logger.error(f"Error getting active jobs count: {str(e)}")
            return 0


    
    
    def _get_job_from_mongodb(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job document from MongoDB"""
        try:
            collection = self.mongo_jobs.collection
            job_doc = collection.find_one({"job_id": job_id})
            if job_doc:
                job_doc["_id"] = str(job_doc["_id"])
            return job_doc
        except Exception as e:
            self.logger.error(f"Error getting job from MongoDB: {str(e)}")
            return None
    
    def _update_job_in_mongodb(self, job_id: str, update_data: Dict[str, Any]) -> bool:
        """Update job document in MongoDB"""
        try:
            collection = self.mongo_jobs.collection
            result = collection.update_one(
                {"job_id": job_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
        except Exception as e:
            self.logger.error(f"Error updating job in MongoDB: {str(e)}")
            return False
    
    


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
                self.logger.info(f"Created {len(tasks)} tasks for job {job_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error creating job tasks: {str(e)}")
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
            collection = self.mongo_jobs.collection
            
            # First, check if the job exists
            job_doc = collection.find_one({"job_id": job_id})
            if not job_doc:
                self.logger.warning(f"Job {job_id} not found")
                return False
            
            tasks = job_doc.get("tasks", [])
            if not tasks:
                self.logger.info(f"No tasks on job {job_id}")
                return False

            task_index, already_at_target = self._resolve_task_index_for_status_update(
                tasks, task_name, status, job_task_order
            )

            if task_index is None:
                self.logger.info(
                    f"Task {task_name} not found as QUEUED in job {job_id}"
                    + (f" (job_task_order={job_task_order})" if job_task_order is not None else "")
                )
                return False

            if already_at_target:
                self.logger.info(
                    f"Task {task_name} already {status} in job {job_id}"
                    + (f" (job_task_order={job_task_order})" if job_task_order is not None else "")
                    + "; skipping duplicate status update"
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
                self.logger.info(f"Updated task {task_name} status to {status} for job {job_id}")
                if time_process is not None:
                    self.logger.info(f"Updated task {task_name} time_process to {time_process}s for job {job_id}")
                
                return True
            else:
                self.logger.warning(f"No task {task_name} found to update in job {job_id}")
                return False
            
        except Exception as e:
            self.logger.error(f"Error updating task status: {str(e)}")
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
            self.logger.error(f"Error getting job progress for {job_id}: {str(e)}")
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
            self.logger.error(f"Error getting job total time process for {job_id}: {str(e)}")
            return 0.0
    

    
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
            self.logger.error(f"Error checking lock status for job {job_id}: {str(e)}")
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
            self.logger.error(f"Error getting processing job order for {job_id}: {str(e)}")
            return None

    
    

    


    

