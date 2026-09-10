"""Re-exports the shared storage providers from common_lib.

Canonical implementation: CommonLib/common_lib/storage/providers.py (shared
with ImageGenBackend). Kept here so existing
``infrastructure.storage.providers`` import paths keep working.
"""

from common_lib.storage.providers import (  # noqa: F401
    LocalStorageProvider,
    RemoteStorageProvider,
)

