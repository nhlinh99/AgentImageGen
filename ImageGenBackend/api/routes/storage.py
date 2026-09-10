"""Serve locally-stored images (StorageService.store_local output) back as raw bytes."""

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from application.services.storage_service import StorageService
from config.settings import Settings

router = APIRouter(prefix="/storage", tags=["Storage"])

config = Settings()
storage_service = StorageService(config)


def _safe_path_segment(value: str) -> bool:
    """Reject anything that could escape the storage root via os.path.join (traversal, separators)."""
    return bool(value) and value not in (".", "..") and "/" not in value and os.sep not in value


@router.get("/{folder}/{filename}", description="Return a stored image's raw bytes")
async def get_image(folder: str, filename: str):
    if not _safe_path_segment(folder) or not _safe_path_segment(filename):
        raise HTTPException(status_code=400, detail="Invalid path")

    if not storage_service.image_storage.image_exists_local(filename, folder):
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = storage_service.image_storage.get_image_path(filename, folder)
    return FileResponse(file_path)
