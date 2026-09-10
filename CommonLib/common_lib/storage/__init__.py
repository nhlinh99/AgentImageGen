"""
ImageStorage module for handling local and remote image storage operations.
Provides a clean, type-safe interface for saving and uploading images.

Shared between ImageGenBackend and ImageGenCelery — each app's own
``infrastructure/storage`` package re-exports from here.
"""

from .base import (
    StorageType,
    ImageFormat,
    StorageResult,
    StorageError,
    FileNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
    BaseStorageProvider,
    StorageValidator,
)

from .providers import (
    LocalStorageProvider,
    RemoteStorageProvider,
    remote_video_preview_and_file_urls,
)

from .manager import StorageManager

__all__ = [
    # Base classes and types
    "StorageType",
    "ImageFormat",
    "StorageResult",
    "StorageError",
    "FileNotFoundError",
    "FileTooLargeError",
    "UnsupportedFormatError",
    "BaseStorageProvider",
    "StorageValidator",

    # Storage providers
    "LocalStorageProvider",
    "RemoteStorageProvider",
    "remote_video_preview_and_file_urls",

    # Main manager
    "StorageManager",
]
