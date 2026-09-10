"""Re-exports the shared storage base classes from common_lib.

Canonical implementation: CommonLib/common_lib/storage/base.py (shared with
ImageGenBackend). Kept here so existing ``infrastructure.storage.base``
import paths keep working.
"""

from common_lib.storage.base import (
    ImageFormat,
    StorageResult,
    StorageError,
    BaseStorageProvider,
)

