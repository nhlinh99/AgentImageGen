"""Mongo connection layer — shared with ImageGenBackend via CommonLib. No Celery-specific
CRUD wrapper needed: `create`/`fetch_collection` on the base already cover every call site here.
"""
from common_lib.mongo_service import MongoService

__all__ = ["MongoService"]
