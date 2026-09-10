"""
Celery chain/group orchestrator.

A thread (see orchestrator.py at repo root) polls the job queue that
ImageGenBackend's JobSubmitService.run_pipeline() writes to (same Mongo/Redis,
via JobService/QueueService), resolves the right celery_process.*Processor for
the job's service_type, builds the chain/group signature, and dispatches it
with apply_async(). GPU-bound task execution itself happens in the workers
(workers/celery_worker.py, celery_ksampler.py, celery_support.py).
"""
import time
from typing import Any, Dict, Optional

from config.config import Config

from domain.schema import (
    CeleryInferenceRequest,
    Status,
    ServiceType,
)

from common_lib.base_services import BaseServiceSingleton

from infrastructure.persistence.job.job_service import JobService
from infrastructure.persistence.job.queue_service import QueueService

from celery_process.inference import CeleryInferenceProcessor
from celery_process.base.base_processor import BaseProcessor


class PipelineOrchestrator(BaseServiceSingleton):
    def __init__(self, config: Config = Config()):
        super(PipelineOrchestrator, self).__init__(config)
        self.config = config if config is not None else Config()

        self.job_service = JobService(self.config)
        self.queue_service = QueueService(self.config)
        self._stop_processor = False

        # Initialize pipelines with service types
        self.pipeline_inference: BaseProcessor = CeleryInferenceProcessor(config)

        # Initialize pipeline mapping for quick lookup
        self.pipeline_map: Dict[str, BaseProcessor] = {
            ServiceType.INFERENCE.value: self.pipeline_inference,
        }

        self.data_model_map = {
            ServiceType.INFERENCE.value: CeleryInferenceRequest,
        }

    def run(self):
        """Block and continuously dispatch queued jobs. Entry point for this process."""
        self.logger.info("Pipeline orchestrator started")
        self._job_processing_loop()

    def stop(self):
        self._stop_processor = True

    def _process_job(self, job_id: str, service_type: str, job_info: Dict[str, Any]):
        """
        Build the chain/group for a job and dispatch it with apply_async()

        :param job_id: Job identifier
        :param service_type: Type of AI service
        :param job_info: Job information from database
        """
        try:
            self.logger.info(f"Starting job {job_id} processing for {service_type}")

            # Update job status to PROCESSING
            self.job_service.update_job_status(
                job_id=job_id,
                status=Status.PROCESSING
            )

            # Remove job from queue
            self.queue_service.remove_job_from_queue(job_id)

            # Get the appropriate pipeline
            pipeline = self._get_pipeline_by_service_type(service_type)
            if not pipeline:
                raise ValueError(f"Unsupported service type: {service_type}")

            # Get data model class for this service type
            data_model_cls = self._get_data_model_by_service_type(service_type)
            if data_model_cls is None:
                raise ValueError(f"No data model for service type: {service_type}")

            # Get request data from job
            request_data = job_info.get("request_data", {})
            processed_request_data = data_model_cls(**request_data)

            # Build the chain/group and dispatch it
            process_out = pipeline.process(
                processed_request_data,
                user_id=job_info.get("user_id"),
            )
            full_pipeline = process_out["chain_task"]
            task_names = process_out["task_names"]
            if task_names:
                self.logger.info(f"Extracted {len(task_names)} tasks from pipeline for job {job_id}: {task_names}")
                self.job_service.create_job_tasks(job_id, task_names)

            celery_task_id = full_pipeline.apply_async().id
            self.job_service.update_task_id(job_id=job_id, celery_task_id=celery_task_id)

        except Exception as e:
            error_message = str(e)

            self.logger.error(f"Job {job_id} failed: {error_message}")

            # Update job status to FAILED
            self.job_service.update_job_status(
                job_id=job_id,
                status=Status.FAILED,
                error_message=error_message
            )

    def load_model(self, service_type: str):
        pipeline = self._get_pipeline_by_service_type(service_type)
        pipeline.processor.load_model()

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job

        :param job_id: Job identifier
        :param user_id: User identifier for authorization
        :return: Success status
        """
        return self.job_service.cancel_job(job_id)

    def get_user_jobs(self, user_id: str, limit: int = 50, offset: int = 0) -> list:
        """
        Get jobs for a specific user

        :param user_id: User identifier
        :param limit: Maximum number of jobs to return
        :param offset: Number of jobs to skip
        :return: List of job information
        """
        return self.job_service.get_user_jobs(user_id, limit, offset)

    def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get system-wide job statistics

        :return: Job statistics
        """
        return self.job_service.get_job_statistics()

    def _job_processing_loop(self):
        """
        Main job processing loop
        """
        while not self._stop_processor:
            try:
                # Process next job (any service type)
                job_processed = self._process_next_job()
                # If no jobs were processed (either no jobs in queue or concurrency limit reached),
                # sleep for a short interval to check again soon
                if not job_processed:
                    time.sleep(1)  # Check every 1 seconds when idle or at limit

            except Exception as e:
                self.logger.error(f"Error in job processing loop: {str(e)}")
                time.sleep(5)  # Wait 5 seconds before retrying

        self.logger.info("Job processing loop stopped")

    def _process_next_job(self) -> bool:
        """
        Process the next job from the queue (any service type)

        :return: True if a job was processed, False otherwise
        """
        try:
            # Multi-lane admission: fast / medium / slow
            next_job = self.queue_service.get_next_job_for_scheduler(
                self.config.job_limits.fast_job,
                self.config.job_limits.medium_job,
                self.config.job_limits.slow_job,
            )
            if not next_job:
                return False

            job_id, position, job_service_type, _lane = next_job

            # Get job details from database
            job_info = self.job_service.get_job_status(job_id)
            if not job_info:
                self.logger.error(f"Job {job_id} not found in database")
                self.queue_service.remove_job_from_queue(job_id)
                return False

            self._process_job(job_id, job_service_type, job_info)

            return True

        except Exception as e:
            self.logger.error(f"Error processing next job: {str(e)}")
            return False

    def _get_pipeline_by_service_type(self, service_type: str) -> Optional[BaseProcessor]:
        """
        Get the appropriate pipeline based on service type

        :param service_type: Type of AI service
        :return: Pipeline interface instance or None if not found
        """
        return self.pipeline_map.get(service_type)

    def _get_data_model_by_service_type(self, service_type: str) -> Optional[type]:
        """
        Get the appropriate data model class based on service type

        :param service_type: Type of AI service
        :return: Data model class or None if not found
        """
        return self.data_model_map.get(service_type)

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current queue status and metrics

        :return: Queue status information
        """
        try:
            queue_stats = self.queue_service.get_queue_statistics()

            # Add concurrency information
            active_jobs_count = self.queue_service.get_processing_jobs_count()
            total_cap = (
                self.config.job_limits.fast_job
                + self.config.job_limits.medium_job
                + self.config.job_limits.slow_job
            )
            queue_stats.update({
                "active_jobs_count": active_jobs_count,
                "limit_total_concurrent_jobs": total_cap,
                "limit_fast_job": self.config.job_limits.fast_job,
                "limit_medium_job": self.config.job_limits.medium_job,
                "limit_slow_job": self.config.job_limits.slow_job,
                "processing_fast": self.queue_service.get_processing_jobs_count("fast"),
                "processing_medium": self.queue_service.get_processing_jobs_count("medium"),
                "processing_slow": self.queue_service.get_processing_jobs_count("slow"),
                "concurrency_utilization": round((active_jobs_count / total_cap) * 100, 2) if total_cap > 0 else 0,
                "can_start_new_job": (
                    self.queue_service.get_processing_jobs_count("fast") < self.config.job_limits.fast_job
                    or self.queue_service.get_processing_jobs_count("medium") < self.config.job_limits.medium_job
                    or self.queue_service.get_processing_jobs_count("slow") < self.config.job_limits.slow_job
                ),
            })

            return queue_stats
        except Exception as e:
            self.logger.error(f"Failed to get queue status: {str(e)}")
            return {"error": str(e)}

    def get_job_queue_position(self, job_id: str) -> Dict[str, Any]:
        """
        Get job's position in the queue

        :param job_id: Job identifier
        :return: Queue position information
        """
        try:
            position = self.queue_service.get_queue_position(job_id)
            if position is not None:
                return {"job_id": job_id, "position": position}
            else:
                return {"job_id": job_id, "position": None, "status": "not_in_queue"}
        except Exception as e:
            self.logger.error(f"Failed to get queue position for job {job_id}: {str(e)}")
            return {"error": str(e)}
