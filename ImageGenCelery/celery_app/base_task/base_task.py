import logging
import socket
import threading
from functools import wraps
from typing import Any, Callable, Optional

from celery.app.task import Task

from config.config import Config

_config = Config()

class BaseTask(Task):
    _image_storage = None
    _redis_client = None
    _logger = None
    _model_lock = threading.Lock()
    _mongo_service = None
    _model_manager = None
    _job_service = None

    @property
    def image_storage(self):
        if self.__class__._image_storage is None:
            from infrastructure.storage import StorageManager
            self.__class__._image_storage = StorageManager(config=_config)
        return self.__class__._image_storage
    
    @property
    def redis_client(self):
        if self.__class__._redis_client is None:
            from config.config import Config
            from infrastructure.cache.redis.redis import RedisClient
            
            redis_client = RedisClient(config=_config)
            self.__class__._redis_client = redis_client
            
        return self.__class__._redis_client
    
    @property
    def logger(self):
        """Logger riêng cho từng task."""
        if self.__class__._logger is None:
            logger_name = f"celery.task.{self.name or self.__class__.__name__}"
            logger = logging.getLogger(logger_name)
            if not logger.hasHandlers():  # tránh trùng handler khi reload
                handler = logging.StreamHandler()
                formatter = logging.Formatter('[%(levelname)s] [%(name)s] [%(asctime)s] %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            self.__class__._logger = logger
        return self.__class__._logger

    @property
    def mongo_service(self):
        if self.__class__._mongo_service is None:
            with self.__class__._model_lock:
                if self.__class__._mongo_service is None:
                    from infrastructure.persistence.mongo.mongo_service import MongoService
                
                    mongo = MongoService(config=_config)
                    self.__class__._mongo_service = mongo
                else:
                    return self.__class__._mongo_service
        return self.__class__._mongo_service

    @property
    def model_manager(self):
        if self.__class__._model_manager is None:
            from infrastructure.model_management.models import ModelManager

            model_manager = ModelManager(config=_config)
            self.__class__._model_manager = model_manager

        return self.__class__._model_manager

    @property
    def job_service(self):
        if self.__class__._job_service is None:
            from config.config import Config
            from infrastructure.persistence.job.job_service import JobService 
        
            job_service = JobService(config=Config())
            self.__class__._job_service = job_service
            
        return self.__class__._job_service

    @property
    def config(self):
        return _config

    def acquire_lock(self, lock_key: str, timeout: int = 300, blocking: bool = False, blocking_timeout: int = 5) -> Optional[Any]:
        """
        Acquire a distributed lock using Redis.
        
        Args:
            lock_key: Unique identifier for the lock
            timeout: Lock expiration time in seconds (default: 300s)
            blocking: Whether to wait for lock acquisition (default: False)
            blocking_timeout: Max time to wait for lock if blocking=True (default: 5s)
            
        Returns:
            Lock object if acquired, None otherwise
        """
        try:
            lock = self.redis_client.client.lock(
                name=f"celery:lock:{lock_key}",
                timeout=timeout,
                blocking=blocking,
                blocking_timeout=blocking_timeout
            )
            
            if lock.acquire(blocking=blocking, blocking_timeout=blocking_timeout):
                self.logger.info(f"Lock acquired: {lock_key}")
                return lock
            else:
                self.logger.warning(f"Failed to acquire lock: {lock_key}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error acquiring lock {lock_key}: {str(e)}")
            return None
    
    def release_lock(self, lock) -> bool:
        """
        Release a distributed lock.
        
        Args:
            lock: Lock object to release
            
        Returns:
            True if released successfully, False otherwise
        """
        try:
            if lock:
                lock.release()
                self.logger.info(f"Lock released successfully")
                return True
        except Exception as e:
            self.logger.error(f"Error releasing lock: {str(e)}")
            return False
        return False
    
    def with_lock(self, lock_key_fn: Optional[Callable] = None, timeout: int = 300, 
                  skip_on_locked: bool = True, retry_on_locked: bool = False,
                  blocking: bool = False, blocking_timeout: int = None):
        """
        Decorator for task methods to ensure single execution using distributed lock.
        
        Args:
            lock_key_fn: Function to generate lock key from task args (default: uses task name)
            timeout: Lock timeout in seconds (how long lock is held)
            skip_on_locked: If True, skip task when locked; if False, raise error (ignored if blocking=True)
            retry_on_locked: If True, retry the task when locked (ignored if blocking=True)
            blocking: If True, wait for lock to be available instead of skipping/retrying
            blocking_timeout: Max time to wait for lock (None = wait indefinitely)
            
        Usage:
            # Wait for lock (blocking)
            @celery_app.task(base=BaseTask, bind=True)
            def my_task(self, user_id):
                @self.with_lock(
                    lock_key_fn=lambda: f"my_task:{user_id}",
                    timeout=600,
                    blocking=True,
                    blocking_timeout=300  # Wait up to 5 minutes
                )
                def execute():
                    # Task logic here
                    return result
                return execute()
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate lock key
                if lock_key_fn:
                    lock_key = lock_key_fn()
                else:
                    lock_key = f"{self.name}:{self.request.id}"
                
                # Try to acquire lock
                if blocking:
                    # Blocking mode: wait for lock
                    self.logger.info(f"Waiting for lock: {lock_key} (timeout: {blocking_timeout}s)")
                    lock = self.acquire_lock(
                        lock_key, 
                        timeout=timeout, 
                        blocking=True, 
                        blocking_timeout=blocking_timeout if blocking_timeout else 0
                    )
                    if lock is None:
                        raise RuntimeError(f"Could not acquire lock after waiting: {lock_key}")
                else:
                    # Non-blocking mode: immediate check
                    lock = self.acquire_lock(lock_key, timeout=timeout, blocking=False)
                    
                    if lock is None:
                        if retry_on_locked:
                            self.logger.info(f"Task locked, retrying: {lock_key}")
                            raise self.retry(countdown=5, max_retries=3)
                        elif skip_on_locked:
                            self.logger.warning(f"Task already running, skipping: {lock_key}")
                            return {
                                "status": "skipped",
                                "reason": "Task is already running",
                                "lock_key": lock_key
                            }
                        else:
                            raise RuntimeError(f"Task is already running with lock: {lock_key}")
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.release_lock(lock)
                    
            return wrapper
        return decorator


class LockedTask(BaseTask):
    """
    Base task class with automatic distributed locking.
    
    Override generate_lock_key() to customize lock key generation.
    Override on_lock_failed() to customize behavior when lock cannot be acquired.
    """
    
    # Default lock configuration (can be overridden per task)
    lock_timeout = 120  # seconds - how long lock is held
    skip_on_locked = False  # skip if already locked
    retry_on_locked = False  # retry if already locked
    wait_for_lock = True  # wait for lock to be available (blocking mode)
    wait_timeout = 120  # max time to wait for lock (None = wait indefinitely)
    
    def __call__(self, *args, **kwargs):
        """Override __call__ to add lock logic before task execution."""
        # Generate lock key
        lock_key = self.generate_lock_key(*args, **kwargs)
        
        # Try to acquire lock
        if self.wait_for_lock:
            # Blocking mode: wait for lock to be available
            self.logger.info(f"Waiting for lock: {lock_key} (max wait: {self.wait_timeout * (_config.limit_total_concurrent_jobs)}s)")
            lock = self.acquire_lock(
                lock_key, 
                timeout=self.lock_timeout * (_config.limit_total_concurrent_jobs + 2), 
                blocking=True, 
                blocking_timeout=self.wait_timeout * (_config.limit_total_concurrent_jobs + 2) if self.wait_timeout else 0
            )
            if lock is None:
                raise RuntimeError(f"Could not acquire lock after waiting: {lock_key}")
        else:
            # Non-blocking mode: immediate check
            lock = self.acquire_lock(lock_key, timeout=self.lock_timeout, blocking=False)
            if lock is None:
                return self.on_lock_failed(lock_key, *args, **kwargs)
        
        try:
            # Execute the actual task
            return self.run(*args, **kwargs)
        finally:
            self.release_lock(lock)
    
    def generate_lock_key(self, *args, **kwargs) -> str:
        """
        Generate lock key from task arguments and worker identity.
        Override this method to customize lock key generation.
        """
        worker_identity = self.get_worker_identity()
        return f"image_gen:{self.name}:{worker_identity}"

    def get_worker_identity(self) -> str:
        """
        Resolve a stable identifier for the current Celery worker so each worker
        has its own lock namespace. Falls back gracefully when the worker
        hostname is not available.
        """

        config_worker_name = getattr(_config, "worker_name", None)
        if config_worker_name:
            return str(config_worker_name)
            
        worker_hostname = getattr(self.request, "hostname", None)
        if worker_hostname:
            return worker_hostname.replace(":", "_")

        return socket.gethostname().replace(":", "_")
    
    def on_lock_failed(self, lock_key: str, *args, **kwargs):
        """
        Handle lock acquisition failure.
        Override this method to customize behavior.
        """
        if self.retry_on_locked:
            self.logger.info(f"Task locked, retrying: {lock_key}")
            raise self.retry(countdown=5, max_retries=3)
        elif self.skip_on_locked:
            self.logger.warning(f"Task already running, skipping: {lock_key}")
            return {
                "status": "skipped",
                "reason": "Task is already running",
                "lock_key": lock_key
            }
        else:
            raise RuntimeError(f"Task is already running with lock: {lock_key}")