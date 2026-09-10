"""Application settings — loaded from environment (pydantic-settings).

Grouped by integration, mirroring BrandStudio/backend/app/config.py. Each group reads
its own env prefix; a few fields keep their original (pre-existing) env var name via
an explicit alias where it doesn't match the group's prefix, so no .env / deployment
env var needs to change.
"""
from __future__ import annotations

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from common_lib.config import MongoSettings, RedisSettings

# Export `.env` -> os.environ so anything reading os.environ directly sees the same
# values pydantic-settings loads. No-op if there is no `.env` (prod uses real env vars).
load_dotenv()


class StorageSettings(BaseSettings):
    """Remote file storage (upload) — TMT-style file server."""
    model_config = SettingsConfigDict(env_prefix="FILE_SERVER_", env_file=".env", extra="ignore")
    # 'local' (default, no external dependency) or 'remote'. Gates every remote-only
    # storage operation (common_lib.storage.manager.StorageManager) -- in local mode
    # those return a StorageResult(success=False, error=<announcement>) instead of
    # trying to call this remote file server.
    mode: str = "local"
    service_name: str = ""
    app_id: str = ""       # secret
    secret: str = ""       # secret
    app_folder: str = ""
    url: str = Field(default="", validation_alias="FILE_SERVER_BASE_URL")
    tenant_id: str = ""
    root_directory_name: str = ""


class JobLimitsSettings(BaseSettings):
    """Job processing — multi-lane Redis concurrency (set each lane independently)."""
    model_config = SettingsConfigDict(env_prefix="LIMIT_", env_file=".env", extra="ignore")
    fast_job: int = 10
    medium_job: int = 10
    slow_job: int = 10

    @property
    def total_concurrent_jobs(self) -> int:
        """Total concurrent job slots across fast + medium + slow lanes."""
        return self.fast_job + self.medium_job + self.slow_job


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    name_app: str = "image"
    storage_path: str = Field(default="storage/", validation_alias="STORAGE_PATH")
    # This API's own externally-reachable root, e.g. "http://localhost:8000" -- used to
    # build absolute URLs for locally-stored files (see StorageService.store_local,
    # api/routes/storage.py's GET /storage/{folder}/{filename}). No trailing slash.
    base_url: str = Field(default="http://localhost:8000", validation_alias="BASE_URL")

    mongo: MongoSettings = Field(default_factory=MongoSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    job_limits: JobLimitsSettings = Field(default_factory=JobLimitsSettings)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton."""
    return Settings()
