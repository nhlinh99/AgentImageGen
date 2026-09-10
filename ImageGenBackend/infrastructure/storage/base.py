"""Re-exports the shared storage base classes from common_lib.

Canonical implementation: CommonLib/common_lib/storage/base.py (shared with
ImageGenCelery). ``ImageFormat``/``StorageResult`` also flow through
``domain.value_objects.storage_result`` (this app's Clean Architecture value
objects for the storage port) — kept here so existing
``infrastructure.storage.base`` import paths keep working.
"""

from domain.value_objects.storage_result import ImageFormat, StorageResult  # noqa: F401
from common_lib.storage.base import (  # noqa: F401
    StorageType,
    StorageError,
    FileNotFoundError,
    FileTooLargeError,
    UnsupportedFormatError,
    RemoteStorageDisabledError,
    BaseStorageProvider,
    StorageValidator,
)
