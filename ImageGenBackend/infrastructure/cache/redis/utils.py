import redis
from typing import Optional

from config.settings import Settings
from infrastructure.cache.redis.redis import RedisClient


_config = Settings()
_redis_client = RedisClient(_config).client


def get_redis_client():
    """
    Return the shared Redis client instance.
    """
    return _redis_client


def delete_keys_by_pattern(pattern: str, client: Optional["redis.Redis"] = None) -> int:
    """
    Delete all keys matching a Redis scan pattern.

    Args:
        pattern: The pattern to match, e.g. 'prefix:*'.
        client: Optional redis client to use; defaults to the shared client.

    Returns:
        Number of keys deleted.
    """
    redis_client = client or _redis_client
    deleted = 0
    for key in redis_client.scan_iter(pattern):
        redis_client.delete(key)
        deleted += 1
    return deleted


def delete_keys_with_prefix(service_name: str, job_id: str, client: Optional["redis.Redis"] = None) -> int:
    """
    Delete keys using the convention '<service_name>*:<job_id>:*'.

    Args:
        service_name: Service/prefix name.
        job_id: Job identifier segment.
        client: Optional redis client to use; defaults to the shared client.

    Returns:
        Number of keys deleted.
    """
    pattern = f"{service_name}*:{job_id}:*"
    return delete_keys_by_pattern(pattern, client)

