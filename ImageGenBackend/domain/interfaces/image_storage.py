"""Port for image/video storage. Only the methods real callers use are here —
StorageManager (the adapter) has more (local-only ops, pickle cache, TTL
cleanup); those are infra-internal, not part of this port.
"""
from __future__ import annotations

from typing import Protocol, Union, runtime_checkable

from PIL import Image
import numpy as np

from domain.value_objects.storage_result import ImageFormat, StorageResult


@runtime_checkable
class IImageStorage(Protocol):
    """Save images/videos to local and/or remote storage."""

    def save_image_local(
        self,
        image: Union[Image.Image, np.ndarray],
        filename: str,
        folder: str = "",
        format: ImageFormat = ImageFormat.JPEG,
    ) -> StorageResult:
        """Save an image to local storage."""

    def save_image_remote(
        self,
        image: Union[Image.Image, np.ndarray],
        filename: str,
        folder: str = "",
        format: ImageFormat = ImageFormat.JPEG,
        max_attempts: int = 3,
    ) -> StorageResult:
        """Save an image to remote storage, retrying transient failures."""

    def save_video_remote(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
        max_attempts: int = 3,
    ) -> StorageResult:
        """Upload raw video bytes to remote storage."""
