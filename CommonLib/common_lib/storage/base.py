"""
Base classes and interfaces for the ImageStorage system.
Provides abstract base classes and type definitions for all storage components.

Canonical, shared between ImageGenBackend and ImageGenCelery — each app's own
``infrastructure/storage/base.py`` re-exports from here so existing import
paths keep working.
"""

from abc import abstractmethod
from typing import Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from PIL import Image
import cv2
import numpy as np

from common_lib.base_services import BaseServiceSingleton
from common_lib.logging_format import module_logger

__all__ = [
    "StorageType", "ImageFormat", "StorageResult", "StorageError",
    "FileNotFoundError", "FileTooLargeError", "UnsupportedFormatError",
    "RemoteStorageDisabledError", "BaseStorageProvider", "StorageValidator",
]


class StorageType(Enum):
    """Enumeration of available storage types."""
    LOCAL = "local"
    REMOTE = "remote"


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


@dataclass
class StorageResult:
    """Result container for storage operations."""
    success: bool
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None
    image: Optional[Union[Image.Image, np.ndarray]] = None


class StorageError(Exception):
    """Base exception for storage operations."""


class FileNotFoundError(StorageError):
    """Exception raised when a file is not found."""


class FileTooLargeError(StorageError):
    """Exception raised when a file is too large."""


class UnsupportedFormatError(StorageError):
    """Exception raised when an unsupported format is used."""


class RemoteStorageDisabledError(StorageError):
    """Raised when a remote-only storage operation is attempted while storage
    mode is 'local' (FILE_SERVER_MODE=local, the default)."""


class BaseStorageProvider(BaseServiceSingleton):
    """Abstract base class for all storage providers."""

    def __init__(self, config: Any):
        super(BaseStorageProvider, self).__init__(config)
        self.config = config
        self.logger = module_logger(self.__class__.__name__)

    @abstractmethod
    def get_storage_type(self) -> StorageType:
        """Return the storage type."""

    @abstractmethod
    def save_image(self, image: Union[Image.Image, np.ndarray],
                   filename: str,
                   folder: str = "",
                   format: ImageFormat = ImageFormat.JPEG) -> StorageResult:
        """
        Save an image to storage.

        Args:
            image: PIL Image or numpy array
            filename: Name of the file to save
            folder: Subfolder within storage
            format: Image format to save as

        Returns:
            StorageResult with operation details
        """

    def validate_filename(self, filename: str) -> bool:
        """
        Validate a filename for security and compatibility.

        Args:
            filename: Filename to validate

        Returns:
            True if valid, False otherwise
        """
        if not filename or len(filename) > 255:
            return False

        # Check for invalid characters
        invalid_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
        return not any(char in filename for char in invalid_chars)

    def validate_image(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Validate that an image is valid.

        Args:
            image: Image to validate

        Returns:
            True if valid, False otherwise
        """
        if image is None:
            return False

        if isinstance(image, np.ndarray):
            return len(image.shape) >= 2
        elif isinstance(image, Image.Image):
            return image.size[0] > 0 and image.size[1] > 0

        return False

    def convert_to_pil_image(self, image: Union[Image.Image, np.ndarray]) -> Image.Image:
        """
        Convert various image formats to PIL Image.

        Args:
            image: Image in various formats

        Returns:
            PIL Image object
        """
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Convert BGR to RGB if needed
                if image.dtype == np.uint8:
                    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                else:
                    return Image.fromarray(image)
            else:
                return Image.fromarray(image)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 4:
                # Convert BGR to RGB if needed
                if image.dtype == np.uint8:
                    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
                else:
                    return Image.fromarray(image)
            else:
                return Image.fromarray(image)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")


class StorageValidator:
    """Utility class for storage validation."""

    @staticmethod
    def validate_file_size(file_size: int, max_size: Optional[int]) -> bool:
        """
        Validate file size against maximum allowed size.

        Args:
            file_size: Size of the file in bytes
            max_size: Maximum allowed size in bytes

        Returns:
            True if valid, False otherwise
        """
        if max_size is None:
            return True
        return file_size <= max_size
