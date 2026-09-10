"""Inference demo endpoint.

Submits a job for ImageGenCelery's CeleryInferenceProcessor (inference.process,
which just prints "Hello World"). Returns the job envelope; poll
GET /v1/jobs/{job_id}/status to see it complete.
"""

import asyncio
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from api.schemas.model import JobPriority, ServiceType
from application.services.job_submit_service import JobSubmitService
from common.logging_format import module_logger
from config.settings import Settings

logger = module_logger("inference_routes")
router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])

config = Settings()
pipeline = JobSubmitService(config)


class InferenceBody(BaseModel):
    job_id: Optional[str] = ""
    priority: Optional[JobPriority] = None
    # Raw image bytes -- job_submit_service's convert_image_to_url() persists it to
    # storage and swaps it for the resulting URL before the job is queued.
    image: Optional[bytes] = Field(default=None)


async def run_pipeline_async(*args, **kwargs):
    return await asyncio.to_thread(pipeline.run_pipeline, *args, **kwargs)


@router.post("/process", description="Send task to inference (demo: prints Hello World)")
async def inference_process(
    image: Optional[UploadFile] = File(default=None),
    job_id: Optional[str] = Form(default=""),
    priority: Optional[JobPriority] = Form(default=None),
):
    try:
        user_id = "anonymous"
        image_bytes = await image.read() if image is not None else None
        body = InferenceBody(job_id=job_id or str(uuid4()), priority=priority, image=image_bytes)

        logger.info("inference request by user: %s, job_id: %s", user_id, body.job_id)

        task_result = await run_pipeline_async(
            body,
            ServiceType.INFERENCE.value,
            user_id,
            (body.priority or JobPriority.HIGH).value,
        )

        return JSONResponse(task_result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("inference error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e
