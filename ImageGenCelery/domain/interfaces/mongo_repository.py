"""Port for Mongo access (shared connection layer lives in CommonLib)."""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class IMongoService(Protocol):
    """Raw collection access + upsert-by-hash write, backed by CommonLib's MongoService."""

    def fetch_collection(self, mongo_db: str, mongo_col: str) -> Any:
        """Return a pymongo-compatible collection handle for `mongo_db.mongo_col`."""

    def create(self, mongo_db: str, mongo_col: str, data: Dict[str, Any]) -> None:
        """Upsert `data` by `_id` (reuses `data['_id']` if set, else derives one from a hash of the dict)."""
