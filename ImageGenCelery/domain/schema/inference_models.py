from typing import Any, Optional

from pydantic import BaseModel, Field

from .job_models import JobPriority


class CeleryInferenceRequest(BaseModel):
    """Request for the inference.process demo task (prints Hello World)."""

    job_id: Optional[str] = Field(default="")
    priority: Optional[JobPriority] = Field(default=None)
    # Already a stored URL by the time this reaches Celery -- ImageGenBackend's
    # job_submit_service.convert_image_to_url() normalizes it before the job is queued.
    image: Optional[Any] = Field(default=None)
