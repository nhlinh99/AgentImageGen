"""Mongo repo for the job document collection.

MongoJobs wraps the same "jobs" collection as ImageGenBackend's
infrastructure/persistence/mongo/repositories/diffusion_repository.py:MongoDiffusionJobs.
"""
from __future__ import annotations

from config.config import Config
from infrastructure.persistence.mongo.mongo_service import MongoService


class MongoJobs:
    def __init__(self, mongo_service: MongoService, config: Config):
        self._mongo = mongo_service
        self._database = config.mongo.database_name
        self._collection_name = config.mongo.collection_diffusion

    @property
    def collection(self):
        return self._mongo.fetch_collection(self._database, self._collection_name)
