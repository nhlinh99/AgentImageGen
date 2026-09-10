"""Celery task decorators for caching task results/inputs in Redis via BaseTask.redis_client."""
from __future__ import annotations

import inspect
import pickle
import traceback
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from celery_app.base_task.base_task import BaseTask


def save_redis(service_name: str = "test", task_name: str = "test", expire_time: int = 1600):
    """Cache a task's return value under key (prefix, service_name, job_id, redis_task_name).

    Only writes when the task is called with allow_save_redis=True. The label defaults
    to task_name but can be overridden per-call via redis_task_name=<unique slot>, so the
    same task can be invoked multiple times per pipeline without each call clobbering the
    previous one's cached value.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self: "BaseTask", *args, **kwargs):
            try:
                result = func(self, *args, **kwargs)

                allow_save_redis = kwargs.get("allow_save_redis", False)
                job_id = kwargs.get("job_id", None)
                redis_task_name = kwargs.get("redis_task_name", task_name)

                if allow_save_redis:
                    key = self.redis_client.create_redis_key(self.config.redis.prefix_key, service_name, job_id, redis_task_name)
                    self.redis_client.set(key, pickle.dumps(result), expire=expire_time)

                return result

            except Exception as e:
                self.logger.error(f"[{service_name}]:[{task_name}] failed with error: {e}")
                self.logger.debug(traceback.format_exc())
                raise e  # để Celery biết task thất bại

        return wrapper
    return decorator

def get_redis(service_name: str = "test", task_name: str = "test"):
    """Inject values cached by save_redis into kwargs before the task runs.

    For each kwarg_name -> ref pair in dict_redis_task_name, fetches the value saved under
    key (prefix, ref_service_name, job_id, sub_task_name) and sets kwargs[kwarg_name].
    ref may be a bare label string (looked up under this task's own service_name, for
    same-module reads) or a (service_name, label) tuple (for reading a value saved by a
    task in a different module/service).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self: "BaseTask", *args, **kwargs):
            try:
                dict_redis_task_name = kwargs.get("dict_redis_task_name", {})
                job_id = kwargs.get("job_id", None)

                if dict_redis_task_name:
                    loaded_from_redis = []
                    for key, ref in dict_redis_task_name.items():
                        ref_service_name, sub_task_name = ref if isinstance(ref, tuple) else (service_name, ref)
                        redis_key = self.redis_client.create_redis_key(self.config.redis.prefix_key, ref_service_name, job_id, sub_task_name)
                        redis_value = self.redis_client.get_data_by_key(redis_key)
                        if redis_value:
                            redis_value = pickle.loads(redis_value)
                            kwargs[key] = redis_value
                            loaded_from_redis.append(key)

                    # Chained tasks may still forward the parent return as a positional arg
                    # even when the same parameter is loaded from Redis above.
                    if loaded_from_redis and args:
                        sig = inspect.signature(func)
                        pos_params = [
                            name for name, param in sig.parameters.items()
                            if name != "self"
                            and param.kind in (
                                inspect.Parameter.POSITIONAL_ONLY,
                                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            )
                        ]
                        skip = set(loaded_from_redis)
                        args = tuple(
                            arg for i, arg in enumerate(args)
                            if i >= len(pos_params) or pos_params[i] not in skip
                        )

                result = func(self, *args, **kwargs)

                return result

            except Exception as e:
                self.logger.error(f"[{service_name}]:[{task_name}] failed with error: {e}")
                self.logger.debug(traceback.format_exc())
                raise e  # để Celery biết task thất bại

        return wrapper
    return decorator
