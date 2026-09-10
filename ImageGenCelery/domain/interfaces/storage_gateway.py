"""Port for local/remote file storage (StorageManager)."""
from __future__ import annotations

from typing import Any, Optional, Protocol, Union, runtime_checkable

from PIL import Image
import numpy as np

from infrastructure.storage.base import ImageFormat, StorageResult


@runtime_checkable
class IStorageManager(Protocol):
    """Unified local + remote storage surface, backed by StorageManager."""

    def save_image_remote(
        self,
        image: Union[Image.Image, np.ndarray],
        filename: str,
        folder: str = "",
        format: ImageFormat = ImageFormat.JPEG,
        max_attempts: int = 3,
    ) -> StorageResult:
        """Save an image to remote storage."""

    def save_video_remote(
        self,
        file_data: bytes,
        filename: str,
        folder: str = "",
        content_type: str = "video/mp4",
        max_attempts: int = 3,
    ) -> StorageResult:
        """Upload raw video bytes to remote file storage."""

    def save_pickle_local(
        self,
        data: Any,
        filename: str,
        folder: str = "",
        timeout: Optional[int] = None,
    ) -> StorageResult:
        """Pickle and save `data` to local storage, optionally TTL-expiring."""

    def load_pickle_local(self, filename: str, folder: str = "") -> StorageResult:
        """Load and unpickle data previously saved with `save_pickle_local`."""

    def load_image_remote(self, url: str, method: str = "opencv") -> StorageResult:
        """Download and decode an image from a remote URL."""

    def get_image_path(self, filename: str, folder: str = "") -> str:
        """Resolve the local filesystem path for a stored image."""
