"""Application settings — loaded from environment (pydantic-settings).

Grouped by integration, mirroring ImageGenBackend/config/settings.py. Each group
reads its own env prefix; a few fields keep their original (pre-existing) env
var name via an explicit alias where it doesn't match the group's prefix, so
no .env / deployment env var needs to change.
"""
from __future__ import annotations

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common_lib.config import MongoSettings, RedisSettings, RabbitMQSettings

# Export `.env` -> os.environ so anything reading os.environ directly sees the same
# values pydantic-settings loads. No-op if there is no `.env` (prod uses real env vars).
load_dotenv()


class JobLimitSettings(BaseSettings):
    """Job processing — multi-lane Redis concurrency (set each lane independently)."""
    model_config = SettingsConfigDict(env_prefix="LIMIT_", env_file=".env", extra="ignore")
    fast_job: int = 10
    medium_job: int = 10
    slow_job: int = 10

    @property
    def total_concurrent_jobs(self) -> int:
        return self.fast_job + self.medium_job + self.slow_job


class Flux2KleinModelPaths(BaseModel):
    # ponytail: every field but flux2_klein_9b_nunchaku was for other flux2_klein
    # loaders (model/clip/vae/lora_faceswap/remove_object_embed), all removed along
    # with their pipelines. Restore them here if those loaders come back.
    flux2_klein_9b_nunchaku: str = "cuda:0,flux2_klein_9b_nunchaku"


class ModelPathsSettings(BaseModel):
    """ComfyUI model registry keys grouped by pipeline: 'device,relative_path'. Not env-driven."""

    flux2_klein: Flux2KleinModelPaths = Field(default_factory=Flux2KleinModelPaths)


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    name_app: str = "image_gen"
    model_path: str = "/TMTAI/AI_MemBer/nhlinh/StableDiffusion/DiffusionService/checkpoints/model"
    celery_general_queue: str = "image_gen_task"
    storage_path: str = "storage/"

    # 'local' (default) or 'remote' -- read by common_lib.storage.providers.resolve_storage_mode.
    # Celery has no FILE_SERVER_* credential fields (removed with the pipelines that used them),
    # so 'remote' isn't actually usable here; the flag exists only so the shared StorageManager
    # never has to special-case which app it's running in.
    file_server_mode: str = Field(default="local", validation_alias="FILE_SERVER_MODE")

    # Route comfy's attention through SageAttention (libs/sageattention-*.whl) instead of
    # torch SDPA. Applied by infrastructure.model_management.comfy_runtime before the first
    # comfy import; falls back to pytorch attention if the package is missing.
    #
    # PROCESS-GLOBAL: comfy picks one attention implementation for everything in the
    # worker, so this changes flux2_klein, z_image, wan and seedvr2 output as well as
    # qwen -- SageAttention quantizes Q/K, so results shift slightly everywhere. Set
    # USE_SAGE_ATTENTION=false to go back to torch SDPA.
    use_sage_attention: bool = Field(default=True, validation_alias="USE_SAGE_ATTENTION")

    # Shared with ImageGenBackend via CommonLib (same env vars, same defaults).
    mongo: MongoSettings = Field(default_factory=MongoSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rabbitmq: RabbitMQSettings = Field(default_factory=RabbitMQSettings)

    job_limits: JobLimitSettings = Field(default_factory=JobLimitSettings)
    model_paths: ModelPathsSettings = Field(default_factory=ModelPathsSettings)

    @property
    def limit_total_concurrent_jobs(self) -> int:
        """Total concurrent job slots across fast + medium + slow lanes."""
        return self.job_limits.total_concurrent_jobs  # pylint: disable=no-member
