"""Value objects for the image-storage port: the output format and result shape.

Canonical definition now lives in CommonLib/common_lib/storage/base.py (shared
with ImageGenCelery) — re-exported here so this stays the domain layer's
import path for the storage port's data contract.

# ponytail: StorageResult.image is typed as PIL.Image/np.ndarray rather than
# a domain-only type. This backend's whole domain IS image data, so treating
# PIL/numpy as value types here (like another domain would treat Decimal) is
# the pragmatic call, not a framework/ORM/HTTP-client dependency.
"""
from __future__ import annotations

from common_lib.storage.base import ImageFormat, StorageResult  # noqa: F401
