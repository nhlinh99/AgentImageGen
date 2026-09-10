"""Generic base for a single-collection Mongo repository.

Every collection-specific repo currently hand-rolls the same three lines
(``self.mongo_service.fetch_collection(db, col)``) plus its own pagination
math and ObjectId stripping. This base gives that boilerplate once; a
concrete repo just sets db/collection and adds its own query methods
(mirrors the "new table follows the base model" convention for SQL).

Not every collection needs a domain port (see
domain/interfaces/user_usage_repository.py's note) — this base is for the
concrete adapter regardless of whether a port sits in front of it.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from infrastructure.persistence.mongo.mongo_service import MongoService


class MongoRepository:
    """Bind one (database, collection) pair to a shared MongoService."""

    def __init__(self, mongo_service: MongoService, database: str, collection: str):
        self._mongo = mongo_service
        self._database = database
        self._collection_name = collection

    @property
    def collection(self):
        return self._mongo.fetch_collection(self._database, self._collection_name)

    def find_one(
        self, filters: Dict[str, Any], projection: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        return self.collection.find_one(filters, projection)

    def find_many(
        self,
        filters: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[Tuple[str, int]] = None,
        skip: int = 0,
        limit: Optional[int] = None,
        projection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        cursor = self.collection.find(filters or {}, projection)
        if sort is not None:
            cursor = cursor.sort(*sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit is not None:
            cursor = cursor.limit(limit)
        return list(cursor)

    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        return self.collection.count_documents(filters or {})

    def paginate(
        self,
        filters: Optional[Dict[str, Any]] = None,
        *,
        sort: Optional[Tuple[str, int]] = None,
        skip: int = 0,
        limit: int = 20,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Page + total in one call — the (docs, total) shape every list_* method needs."""
        total = self.count(filters)
        docs = self.find_many(filters, sort=sort, skip=skip, limit=limit)
        return docs, total

    def insert_one(self, document: Dict[str, Any]):
        return self.collection.insert_one(document)

    def update_one(
        self, filters: Dict[str, Any], update: Dict[str, Any], *, upsert: bool = False
    ):
        return self.collection.update_one(filters, update, upsert=upsert)

    def update_many(
        self, filters: Dict[str, Any], update: Dict[str, Any], *, upsert: bool = False
    ):
        return self.collection.update_many(filters, update, upsert=upsert)

    def delete_one(self, filters: Dict[str, Any]):
        return self.collection.delete_one(filters)

    def delete_many(self, filters: Dict[str, Any]):
        return self.collection.delete_many(filters)

    @staticmethod
    def strip_id(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Drop Mongo's ``_id`` before handing a doc to a response schema."""
        doc = dict(doc)
        doc.pop("_id", None)
        return doc
