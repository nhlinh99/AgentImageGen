from fastapi import APIRouter

from api.routes import health, inference, jobs, results, storage

# Create main router
main_router = APIRouter()

# Include all route modules
main_router.include_router(inference.router)
main_router.include_router(results.router)
main_router.include_router(jobs.router)
main_router.include_router(storage.router)
main_router.include_router(health.router)
