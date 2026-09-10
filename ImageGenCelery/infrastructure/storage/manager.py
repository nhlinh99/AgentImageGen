"""Re-exports the shared StorageManager from common_lib.

Canonical implementation: CommonLib/common_lib/storage/manager.py (shared
with ImageGenBackend). Kept here so existing ``infrastructure.storage.manager``
import paths keep working.
"""

from common_lib.storage.manager import StorageManager  # noqa: F401

