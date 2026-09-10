import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.schemas.data.job_models import Status

# Global imports to avoid circular imports
from application.services.job_submit_service import JobSubmitService
from common.logging_format import module_logger
from config.settings import Settings

router = APIRouter(prefix="/result", tags=['Result'])
logger = module_logger("results_routes")

# Re-fetch job status this many times (with delay) before cancelling a stuck PROCESSING job.
_STUCK_JOB_STATUS_RETRIES = 3
_STUCK_JOB_RETRY_DELAY_SEC = 1.0


def _job_is_processing_all_tasks_completed(job_doc: dict) -> bool:
    tasks = job_doc.get("tasks") or []
    return (
        job_doc.get("status") == Status.PROCESSING.value
        and bool(tasks)
        and all(t.get("status") == Status.COMPLETED.value for t in tasks)
    )


# Global variables
config = Settings()
pipeline = JobSubmitService(config)


@router.get("/{job_id}", description="Get job result and status from MongoDB")
async def get_result_by_job_id(job_id: str):
    try:
        logger.info('Get job_id result request for job_id: %s', job_id)
        job_info = await asyncio.to_thread(pipeline.job_service.get_job_status, job_id=job_id)

        if _job_is_processing_all_tasks_completed(job_info):
            for _ in range(_STUCK_JOB_STATUS_RETRIES):
                await asyncio.sleep(_STUCK_JOB_RETRY_DELAY_SEC)
                job_info = await asyncio.to_thread(
                    pipeline.job_service.get_job_status, job_id=job_id
                )
                if not _job_is_processing_all_tasks_completed(job_info):
                    break
            if _job_is_processing_all_tasks_completed(job_info):
                stuck_msg = "Task is cancelled for waiting too long..."
                logger.warning('Job %s still PROCESSING with all tasks completed after %s retries; cancelling with: %s', job_id, _STUCK_JOB_STATUS_RETRIES, stuck_msg)
                await asyncio.to_thread(
                    pipeline.job_service.update_job_status,
                    job_id,
                    Status.CANCELLED,
                    error_message=stuck_msg,
                )
                job_info = await asyncio.to_thread(
                    pipeline.job_service.get_job_status, job_id=job_id
                )

        prog = await asyncio.to_thread(pipeline.job_service.get_job_progress, job_id)
        if prog:
            job_info["progress_percentage"] = prog.get("progress_percentage")
            job_info["steps_finished_percentage"] = prog.get("steps_finished_percentage")
            job_info["job_progress"] = prog

        return JSONResponse(content=job_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error getting job_id result: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/{job_id}", description="Delete job from MongoDB and remove from queues")
async def delete_result_by_job_id(job_id: str):
    try:
        logger.info('Delete job_id result request for job_id: %s', job_id)
        user_id = "anonymous"
        job_info = await asyncio.to_thread(
            pipeline.job_service.delete_job, job_id=job_id, user_id=user_id
        )
        return JSONResponse(content=job_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error getting job_id result: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
