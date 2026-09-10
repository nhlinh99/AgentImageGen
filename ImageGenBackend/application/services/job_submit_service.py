from typing import Dict, Any, Optional
from config.settings import Settings

from common.base_services import BaseServiceSingleton
from application.services.image_url_conversion import convert_image_to_url

from infrastructure.persistence.job.job_service import JobService
from domain.interfaces.job_repository import IJobRepository
from api.schemas.model import Status


class JobSubmitService(BaseServiceSingleton):
    """
    API-side job submitter. Creates and enqueues a job, then returns immediately.

    Chain/group building and dispatch (celery_process processors + apply_async)
    happens out-of-process in ImageGenCelery's PipelineOrchestrator, which polls
    the same job queue (Mongo/Redis, via JobService/QueueService) this class writes to.
    """

    def __init__(self, config: Settings = Settings()):
        super(JobSubmitService, self).__init__(config)
        self.config = config if config is not None else Settings()
        self.job_service: IJobRepository = JobService(self.config)

    def run_pipeline(self,
                    request,
                    service_type: str,
                    user_id: Optional[str],
                    priority: str) -> Dict[str, Any]:
        """
        Internal method to run pipeline with job management

        :param request: Request data
        :param service_type: Type of AI service
        :param user_id: User identifier
        :param priority: Job priority
        :param pipeline: Pipeline interface instance
        :return: Job information and async result
        """
        try:
            # Convert request to dict for job storage
            if hasattr(request, "dict"):
                request_data = request.dict()
            elif hasattr(request, "__dict__"):
                request_data = request.__dict__
            else:
                request_data = str(request)

            # Create job if user_id is provided
            job_id = None
            job_info = None

            # Convert request_data image (url, numpy, PIL) into local URLs (only when request is dict-like)
            if isinstance(request_data, dict):
                processed_request_data = convert_image_to_url(request_data)
            else:
                processed_request_data = None

            if user_id and processed_request_data is not None:
                try:
                    job_info = self.job_service.create_job(
                        user_id=user_id,
                        service_type=service_type,
                        request_data=processed_request_data,
                        priority=priority,
                    )
                    job_id = job_info["job_id"]
                    self.logger.info('Created job %s for %s pipeline', job_id, service_type)
                except Exception as e:
                    self.logger.error('Failed to create job: %s', str(e))
                    # Continue without job management if job creation fails

            # Return job information (pipeline execution handled by the orchestrator)
            result = {
                "job_created": job_id is not None,
                "pipeline_execution": "deferred_to_job_processor"
            }

            if job_info:
                result["job_info"] = job_info

            return result

        except Exception as e:
            error_message = f"Error in pipeline execution: {str(e)}"
            self.logger.error(error_message)

            # Update job status to failed if job exists
            if job_id:
                try:
                    self.job_service.update_job_status(
                        job_id=job_id,
                        status=Status.FAILED,
                        error_message=error_message
                    )
                except Exception as update_error:
                    self.logger.error('Failed to update job status: %s', str(update_error))

            raise ValueError(error_message)
