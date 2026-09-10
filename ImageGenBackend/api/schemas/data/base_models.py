from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict


class HealthCheckResponse(BaseModel):
    status: str


class MetaData(BaseModel):
    status_code: int
    processing_time: str
    info: Optional[Any] = None


class TextRequest(BaseModel):
    text: str = Field(default="")


class ImageRequest(BaseModel):
    image: Any = Field(default="")


class ControlnetRequest(BaseModel):
    image: Any = Field(default="")


class KSamplerRequest(BaseModel):
    seed: int = Field(default=42)
    steps: int = Field(default=20)
    cfg: float = Field(default=7.5)
    sampler_name: str = Field(default="euler")
    scheduler: str = Field(default="normal")
    denoise: float = Field(default=1.0)
    positive: List[Any] = Field(default=[])
    negative: List[Any] = Field(default=[])
    controlnet: Any = Field(default=None)
    latent_image: Dict = Field(default={})
    use_teacache: bool = Field(default=True)


class StyleModelApplySimpleRequest(BaseModel):
    cliptextencode_emb: List[Any] = Field(default=[])
    clipvisionencode_emb: Dict[str, Any] = Field(default={})


class InpaintModelConditioningRequest(BaseModel):
    noise_mask: bool = Field(default=True)
    positive: List[Any] = Field(default=[])
    negative: List[Any] = Field(default=[])
    pixels: Any = Field(default="")
    mask: Any = Field(default="")


class VAEDecodeRequest(BaseModel):
    samples: Dict = Field(default={})


class DeleteKeysRedisRequest(BaseModel):
    service_name: str
    job_id: str


class TaskStatus:
    NEW = "NEW"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    STARTED = "STARTED"
    FAILURE = "FAILURE"
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    REVOKED = "REVOKED"
    RETRY = "RETRY"


class DiffusionRequestInfo(BaseModel):
    id: str = Field(default="", alias="_id")
    job_id: str = ""
    status: str = TaskStatus.QUEUED
    service_name: str = ""
    seq: int = 0
    task_id: str = ""
    tasks: list = []
    timestamp: Any


class SignInRequest(BaseModel):
    phone_number: str
    password: str
