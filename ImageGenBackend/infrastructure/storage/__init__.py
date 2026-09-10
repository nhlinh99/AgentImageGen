"""
ImageStorage module for handling local and remote image storage operations.
Provides a clean, type-safe interface for saving and uploading images.
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
    StorageValidator
)

from .providers import (
    LocalStorageProvider,
    RemoteStorageProvider
)

from .manager import StorageManager

# Export main classes
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
    
    # Main manager
    "StorageManager",
]
