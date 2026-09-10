"""
Backward compatibility module for model imports.
This file maintains compatibility with existing code that imports from api.schemas.model.

For new code, it's recommended to import directly from the specific model files:
- from api.schemas.data.base_models import HealthCheckResponse, TaskStatus
- from api.schemas.data.job_models import Status, JobPriority, ServiceType

Or use the convenience import:
- from api.schemas import HealthCheckResponse, TaskStatus
"""

# Import all models for backward compatibility
from .data.base_models import (
    HealthCheckResponse,
    MetaData,
    TextRequest,
    ImageRequest,
    ControlnetRequest,
    KSamplerRequest,
    StyleModelApplySimpleRequest,
    InpaintModelConditioningRequest,
    VAEDecodeRequest,
    DeleteKeysRedisRequest,
    TaskStatus,
    DiffusionRequestInfo,
    SignInRequest
)

from .data.job_models import *

# Export all models for backward compatibility
__all__ = [
    # Base models
    "HealthCheckResponse",
    "MetaData",
    "TextRequest",
    "ImageRequest",
    "ControlnetRequest",
    "KSamplerRequest",
    "StyleModelApplySimpleRequest",
    "InpaintModelConditioningRequest",
    "VAEDecodeRequest",
    "DeleteKeysRedisRequest",
    "TaskStatus",
    "DiffusionRequestInfo",
    "SignInRequest",
]
