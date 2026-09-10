"""
ImageStorage module for handling local and remote image storage operations.
Provides a clean, type-safe interface for saving and uploading images.
"""

from .base import (
    ImageFormat,
    StorageResult,
    StorageError,
    BaseStorageProvider,
)

from .providers import (
    LocalStorageProvider,
    RemoteStorageProvider
)

from .manager import StorageManager

# Export main classes
__all__ = [
    # Base classes and types
    "ImageFormat",
    "StorageResult",
    "StorageError",
    "BaseStorageProvider",

    # Storage providers
    "LocalStorageProvider",
    "RemoteStorageProvider",

    # Main manager
    "StorageManager",
]
