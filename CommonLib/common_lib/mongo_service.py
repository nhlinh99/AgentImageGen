"""Low-level async Mongo client wrapper.

Runs pymongo's async driver on a dedicated event-loop thread so the rest of
the codebase (Celery tasks, routes, services — all synchronous) can call it
directly without an event loop of their own.

Collection-specific query logic belongs in a repository/subclass built on top
of this — this class only hands out the raw collection handle (fetch_collection)
plus the couple of generic helpers still used directly (fetch_all_documents, create).
"""
import asyncio
import hashlib
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from bson import ObjectId
from pymongo import AsyncMongoClient

from common_lib.base_services import BaseServiceSingleton
from common_lib.config import MongoSettings


def _resolve_mongo_settings(config: Any) -> MongoSettings:
    """Accept either a full app config/settings (with a `.mongo` sub-object) or a MongoSettings directly."""
    if isinstance(config, MongoSettings):
        return config
    return getattr(config, "mongo", config)


class _AsyncMongoRunner:
    """
    Runs pymongo async operations on a dedicated asyncio loop thread.

    This keeps the existing codebase synchronous (Celery/tasks/routes call MongoService
    methods directly) while still using an async Mongo client underneath.
    """

    def __init__(self, mongo_settings: MongoSettings, logger):
        self._config = mongo_settings
        self._logger = logger
        self._loop_ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Optional[AsyncMongoClient] = None

        self._thread = threading.Thread(target=self._thread_main, daemon=True)
        self._thread.start()
        self._loop_ready.wait()

    def _thread_main(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._logger.info(' Connecting to %s...', self._config.uri)
        self._client = AsyncMongoClient(self._config.uri)

        self._loop_ready.set()
        self._loop.run_forever()

    def run(self, coro):
        if self._loop is None:
            raise RuntimeError("Async mongo runner loop not initialized.")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    @property
    def client(self) -> AsyncMongoClient:
        if self._client is None:
            raise RuntimeError("Async mongo runner client not initialized.")
        return self._client


class _MongoCursorSyncProxy:
    def __init__(
        self,
        runner: _AsyncMongoRunner,
        mongo_db: str,
        mongo_col: str,
        query_filter: Optional[Dict[str, Any]] = None,
        projection: Optional[Dict[str, Any]] = None,
    ):
        self._runner = runner
        self._mongo_db = mongo_db
        self._mongo_col = mongo_col
        self._query_filter = query_filter
        self._projection = projection

        self._sort: Optional[List[Tuple[str, int]]] = None
        self._skip: int = 0
        self._limit: Optional[int] = None

    def sort(
        self,
        sort_key: Union[str, List[Tuple[str, int]], Tuple[str, int]],
        sort_dir: Optional[int] = None,
    ):
        """PyMongo-compatible: ``sort(field, dir)`` or ``sort([(field, dir), ...])``."""
        if sort_dir is not None:
            self._sort = [(str(sort_key), int(sort_dir))]
        elif isinstance(sort_key, list):
            self._sort = [(str(k), int(d)) for k, d in sort_key]
        elif isinstance(sort_key, tuple):
            self._sort = [(str(sort_key[0]), int(sort_key[1]))]
        else:
            raise TypeError(f"Unsupported sort spec: {sort_key!r}")
        return self

    def skip(self, n: int):
        self._skip = n
        return self

    def limit(self, n: int):
        self._limit = n
        return self

    async def _fetch_all(self) -> List[Dict[str, Any]]:
        collection = self._runner.client[self._mongo_db][self._mongo_col]

        if self._query_filter is None and self._projection is None:
            cursor = collection.find()
        elif self._projection is None:
            cursor = collection.find(self._query_filter)
        elif self._query_filter is None:
            cursor = collection.find({}, self._projection)
        else:
            cursor = collection.find(self._query_filter, self._projection)

        if self._sort is not None:
            cursor = cursor.sort(self._sort)
        if self._skip:
            cursor = cursor.skip(self._skip)
        if self._limit is not None:
            cursor = cursor.limit(self._limit)

        docs: List[Dict[str, Any]] = []
        async for doc in cursor:
            docs.append(doc)
        return docs

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        docs = self._runner.run(self._fetch_all())
        return iter(docs)


class _MongoAggregateSyncProxy:
    def __init__(self, runner: _AsyncMongoRunner, mongo_db: str, mongo_col: str, pipeline: List[Dict[str, Any]]):
        self._runner = runner
        self._mongo_db = mongo_db
        self._mongo_col = mongo_col
        self._pipeline = pipeline

    async def _fetch_all(self) -> List[Dict[str, Any]]:
        collection = self._runner.client[self._mongo_db][self._mongo_col]
        cursor = await collection.aggregate(self._pipeline)
        docs: List[Dict[str, Any]] = []
        async for doc in cursor:
            docs.append(doc)
        return docs

    async def _fetch_one(self) -> Optional[Dict[str, Any]]:
        collection = self._runner.client[self._mongo_db][self._mongo_col]
        cursor = await collection.aggregate(self._pipeline)
        async for doc in cursor:
            return doc
        return None

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        docs = self._runner.run(self._fetch_all())
        return iter(docs)

    def next(self) -> Dict[str, Any]:
        doc = self._runner.run(self._fetch_one())
        if doc is None:
            raise StopIteration
        return doc


class _MongoCollectionSyncProxy:
    def __init__(self, runner: _AsyncMongoRunner, mongo_db: str, mongo_col: str):
        self._runner = runner
        self._mongo_db = mongo_db
        self._mongo_col = mongo_col

    def find(self, query_filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None):
        return _MongoCursorSyncProxy(
            runner=self._runner,
            mongo_db=self._mongo_db,
            mongo_col=self._mongo_col,
            query_filter=query_filter,
            projection=projection,
        )

    def aggregate(self, pipeline: List[Dict[str, Any]]):
        return _MongoAggregateSyncProxy(
            runner=self._runner,
            mongo_db=self._mongo_db,
            mongo_col=self._mongo_col,
            pipeline=pipeline,
        )

    def find_one(self, query_filter: Optional[Dict[str, Any]] = None, projection: Optional[Dict[str, Any]] = None):
        async def _find_one():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            if projection is None:
                return await collection.find_one(query_filter)
            return await collection.find_one(query_filter, projection)

        return self._runner.run(_find_one())

    def count_documents(self, query_filter: Dict[str, Any]):
        async def _count():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.count_documents(query_filter)

        return int(self._runner.run(_count()))

    def insert_one(self, document: Dict[str, Any]):
        async def _insert():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.insert_one(document)

        return self._runner.run(_insert())

    def update_one(self, query_filter: Dict[str, Any], update_data: Dict[str, Any], upsert: bool = False, **kwargs):
        async def _update_one():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.update_one(query_filter, update_data, upsert=upsert, **kwargs)

        return self._runner.run(_update_one())

    def update_many(self, query_filter: Dict[str, Any], update_data: Dict[str, Any], upsert: bool = False, **kwargs):
        async def _update_many():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.update_many(query_filter, update_data, upsert=upsert, **kwargs)

        return self._runner.run(_update_many())

    def delete_one(self, query_filter: Dict[str, Any]):
        async def _delete_one():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.delete_one(query_filter)

        return self._runner.run(_delete_one())

    def delete_many(self, query_filter: Dict[str, Any]):
        async def _delete_many():
            collection = self._runner.client[self._mongo_db][self._mongo_col]
            return await collection.delete_many(query_filter)

        return self._runner.run(_delete_many())


class MongoService(BaseServiceSingleton):
    """Base Mongo service: connection + generic collection access.

    `config` may be the full app config/settings object (read via its `.mongo`
    sub-object) or a `MongoSettings` instance directly.
    """

    def __init__(self, config: Any):
        super().__init__(config)

        self.mongo_settings: MongoSettings = _resolve_mongo_settings(config)
        self.mongo_client: Optional[_AsyncMongoRunner] = None
        self.connect_service_mongo()

    def connect_service_mongo(self) -> None:
        self.mongo_client = _AsyncMongoRunner(self.mongo_settings, self.logger)

    def fetch_collection(self, mongo_db: str, mongo_col: str) -> _MongoCollectionSyncProxy:
        return _MongoCollectionSyncProxy(self.mongo_client, mongo_db, mongo_col)

    def fetch_all_documents(self, mongo_db: str, mongo_col: str):
        documents = list(self.fetch_collection(mongo_db, mongo_col).find())
        return self._convert_objectid_to_string(documents)

    def create(self, mongo_db: str, mongo_col: str, data: dict) -> None:
        """Upsert by ``_id``: reuses data['_id'] if set, else derives one from a hash of the dict."""
        collection = self.fetch_collection(mongo_db, mongo_col)
        if not data.get("_id"):
            data["_id"] = hashlib.md5(str(data).encode("utf8")).hexdigest()
        collection.update_many({"_id": data["_id"]}, {"$set": data}, upsert=True)

    @staticmethod
    def _convert_objectid_to_string(obj):
        """
        Recursively convert ObjectId instances to strings in MongoDB documents
        """
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: MongoService._convert_objectid_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [MongoService._convert_objectid_to_string(item) for item in obj]
        else:
            return obj
