"""Mongo repo for the diffusion job history collection.

MongoDiffusionJobs wraps the same "jobs" collection as job_service.py's queue/
lifecycle owner.
"""
from __future__ import annotations

from config.settings import Settings
from infrastructure.persistence.mongo.mongo_service import MongoService
from infrastructure.persistence.mongo.base_repository import MongoRepository


class MongoDiffusionJobs(MongoRepository):
    def __init__(self, mongo_service: MongoService, config: Settings):
        super().__init__(mongo_service, config.mongo.database_name, config.mongo.collection_diffusion)
