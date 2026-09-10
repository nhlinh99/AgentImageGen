"""FastAPI app factory — mirrors BrandStudio/backend/app/main.py's create_app() + lifespan shape.

Run via uvicorn (module:app), not an inline uvicorn.run() call:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from common.logging_format import module_logger
from config.settings import Settings
from api.routes.main import main_router

logger = module_logger("celery_client")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Initialize services and start background threads on application startup."""
    settings: Settings = fastapi_app.state.settings

    from infrastructure.persistence.job.job_service import JobService
    from infrastructure.storage.manager import StorageManager

    job_service = JobService(settings)
    job_service.start_processing_times_refresh_thread(interval_seconds=7200)
    job_service.refresh_service_processing_times(initial_run=True)
    logger.info("Started service processing times refresh thread")

    storage_manager = StorageManager(settings)
    storage_manager.start_cleanup_thread()

    yield


OPENAPI_TAGS = [
    {"name": "Inference", "description": "Demo inference endpoint (prints Hello World)."},
]


def create_app() -> FastAPI:
    settings = Settings()
    fastapi_app = FastAPI(
        title="ImageGenBackend API",
        version="2.0.0",
        lifespan=lifespan,
        openapi_tags=OPENAPI_TAGS,
    )
    fastapi_app.state.settings = settings

    # CORS middleware
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    fastapi_app.include_router(main_router)

    return fastapi_app


app = create_app()
