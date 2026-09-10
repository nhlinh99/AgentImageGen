import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.schemas.data.job_models import ServiceTypeLockRequest, Status
from common.logging_format import module_logger
from config.settings import Settings
from domain.interfaces.job_repository import IJobRepository
from infrastructure.persistence.job.job_service import JobService

router = APIRouter(prefix="/jobs", tags=['Job Management'])
logger = module_logger("job_routes")

# Initialize services
config = Settings()
job_service: IJobRepository = JobService(config)


@router.get("/queue/lanes", description="Per-user lane: waiting, processing, estimated_wait_seconds (max ETA among your jobs in lane)")
async def get_queue_lane_snapshot():
    """
    For each lane: Redis ``waiting`` and ``processing`` counts for your ZSET members, plus
    ``estimated_wait_seconds`` (max of ``calculate_wait_time_for_job`` over your queued/processing jobs in that lane).
    Clients typically show queue as ``waiting + processing``.
    """
    try:
        user_id = "anonymous"
        lanes = await asyncio.to_thread(job_service.get_user_queue_lanes_snapshot, str(user_id))
        return JSONResponse(content={"lanes": lanes, "scope": "user"})
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Error getting queue lane snapshot: %s', str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{job_id}/status", description="Get job status including lock information")
async def get_job_status(job_id: str):
    """
    Get detailed job status including lock information

    Args:
        job_id: Job identifier

    Returns:
        Job status with lock information
    """
    try:
        logger.info('Get job status request for job_id: %s', job_id)

        # Get job status
        job_status = job_service.get_job_status(job_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "job_status": job_status,
                "processing_order": job_status.get("processing_order")
            }
        )

    except ValueError as e:
        logger.error('Validation error getting job status %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error getting job status %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.get("/{job_id}/queue-position", description="Get queue position for a specific job")
async def get_job_queue_position(job_id: str):
    """
    Get queue position for a specific job

    Args:
        job_id: Job identifier

    Returns:
        Queue position information
    """
    try:
        logger.info('Get queue position request for job_id: %s', job_id)

        # Get job status to extract queue position
        job_status = job_service.get_job_status(job_id)

        # Extract queue position and related information
        queue_position = job_status.get("queue_position")
        processing_order = job_status.get("processing_order")
        job_status_value = job_status.get("status")
        is_locked = job_status.get("is_locked", False)
        service_type = job_status.get("service_type")

        # Determine if job is in queue
        in_queue = (job_status_value == Status.QUEUED.value and not is_locked and queue_position is not None)

        # Generate specific message based on job status
        if job_status_value == Status.QUEUED.value:
            if is_locked:
                message = f"Job is locked and cannot be processed (position: {queue_position})"
            elif queue_position is not None:
                message = f"Job is queued at position {queue_position} and waiting to be processed"
            else:
                message = "Job is queued but position information is not available"
        elif job_status_value == Status.PROCESSING.value:
            if processing_order is not None:
                message = f"Job is currently being processed (processing order: {processing_order})"
            else:
                message = "Job is currently being processed"
        elif job_status_value == Status.COMPLETED.value:
            message = "Job has been completed successfully"
        elif job_status_value == Status.FAILED.value:
            message = "Job has failed"
        else:
            message = f"Job status: {job_status_value}"

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "job_id": job_id,
                "queue_position": queue_position,
                "processing_order": processing_order,
                "in_queue": in_queue,
                "job_status": job_status_value,
                "service_type": service_type,
                "is_locked": is_locked,
                "message": message
            }
        )

    except ValueError as e:
        logger.error('Validation error getting queue position for job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error getting queue position for job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.delete("/{job_id}/cancel", description="Cancel a job")
async def cancel_job(job_id: str):
    """
    Cancel a job if it's still queued or pending

    Args:
        job_id: Job identifier

    Returns:
        Success response with cancellation information
    """
    try:
        user_id = "anonymous"
        logger.info('Cancel job request for job_id: %s', job_id)

        # Cancel the job
        success = job_service.cancel_job(job_id, user_id)

        if not success:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": f"Failed to cancel job."
                }
            )

        # Get updated job status
        job_status = job_service.get_job_status(job_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Job {job_id} cancelled successfully",
                "job_id": job_id,
                "cancelled": True,
                "cancelled_by": user_id,
                "cancelled_at": job_status.get("updated_at"),
                "job_status": job_status
            }
        )

    except ValueError as e:
        logger.error('Validation error cancelling job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error cancelling job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.post("/lock-by-service", description="Lock all jobs of a specific service type")
async def lock_jobs_by_service_type(request: ServiceTypeLockRequest):
    """
    Lock all jobs of a specific service type for the current user

    Args:
        request: Service type lock request containing service_type and lock_duration

    Returns:
        Success response with lock information
    """
    try:
        user_id = "anonymous"
        logger.info('Lock jobs by service type request for service_type: %s', request.service_type)

        # Get jobs for the user with the specified service type and status (only queued jobs can be locked)
        target_jobs = job_service.get_user_jobs(
            user_id=user_id,
            service_type=request.service_type,
            status=Status.QUEUED.value,
            locked=False
        )

        if not target_jobs:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"No {Status.QUEUED.value} jobs found for service type '{request.service_type}'",
                    "service_type": request.service_type,
                    "locked_count": 0,
                    "locked_jobs": []
                }
            )

        # Lock each job
        locked_jobs = []
        failed_locks = []

        for job in target_jobs:
            job_id = job.get("job_id")
            if not job_id:
                failed_locks.append({
                    "job_id": None,
                    "error": "Job ID is missing"
                })
                continue
            try:
                success = job_service.lock_job(
                    job_id=job_id,
                    user_id=user_id,
                    lock_duration=request.lock_duration
                )

                if success:
                    locked_jobs.append({
                        "job_id": job_id,
                        "service_type": request.service_type,
                        "created_at": job.get("created_at")
                    })
                else:
                    failed_locks.append({
                        "job_id": job_id,
                        "error": "Failed to lock job"
                    })

            except Exception as e:
                failed_locks.append({
                    "job_id": job_id,
                    "error": str(e)
                })

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Locked {len(locked_jobs)} jobs for service type '{request.service_type}'",
                "service_type": request.service_type,
                "locked_count": len(locked_jobs),
                "failed_count": len(failed_locks),
                "lock_duration": request.lock_duration,
                "locked_jobs": locked_jobs,
                "failed_locks": failed_locks
            }
        )

    except ValueError as e:
        logger.error('Validation error locking jobs by service type: %s', str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error locking jobs by service type: %s', str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.post("/retry-fail-jobs", description="Retry all failed jobs")
async def retry_fail_jobs(limit: int = 0):
    """
    Retry all failed jobs by resetting their state and re-queueing them.

    Args:
        limit: Optional limit on the number of failed jobs to retry (0 = no limit).
    """
    try:
        logger.info('Retry failed jobs request (limit=%s)', limit)

        retry_limit = limit if limit and limit > 0 else None
        result = job_service.retry_failed_jobs(retry_limit)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Retry failed jobs completed",
                **result
            }
        )
    except Exception as e:
        logger.error('Error retrying failed jobs: %s', str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.post("/retry-fail-job/{job_id}", description="Retry a specific failed job")
async def retry_fail_job(job_id: str):
    """
    Retry a single failed job.
    """
    try:
        logger.info('Retry failed job request for job_id: %s', job_id)

        result = job_service.retry_failed_job(job_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Job {job_id} retried successfully",
                **result
            }
        )
    except ValueError as e:
        logger.error('Validation error retrying job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error retrying job %s: %s', job_id, str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )


@router.post("/synchronize", description="Synchronize jobs between MongoDB and Redis queue")
async def synchronize_jobs():
    """
    Synchronize jobs between MongoDB and Redis queue.
    This endpoint ensures that all QUEUED and PROCESSING jobs in MongoDB
    are properly reflected in the Redis queue.

    Steps:
    1. Get all jobs with status PROCESSING or QUEUED from MongoDB
    2. Check if each job_id is in the Redis queue
    3. If not in Redis queue, add the job_id to Redis

    Returns:
        Success response with synchronization statistics
    """
    try:
        logger.info('Synchronize jobs request')

        # Launch synchronization in a background thread
        thread_info = job_service.synchronize_jobs_async()

        if thread_info.get("started"):
            return JSONResponse(
                status_code=202,
                content={
                    "success": True,
                    "message": "Job synchronization started",
                    "thread_name": thread_info.get("thread_name")
                }
            )

        response_content = {
            "success": False,
            "message": "Job synchronization is already running",
            "thread_name": thread_info.get("thread_name"),
            "running": thread_info.get("running", False)
        }

        if thread_info.get("last_result") is not None:
            response_content["last_result"] = thread_info["last_result"]
        if thread_info.get("error") is not None:
            response_content["error"] = thread_info["error"]

        return JSONResponse(
            status_code=409,
            content=response_content
        )

    except ValueError as e:
        logger.error('Validation error synchronizing jobs: %s', str(e))
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(e)
            }
        )
    except Exception as e:
        logger.error('Error synchronizing jobs: %s', str(e))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error"
            }
        )