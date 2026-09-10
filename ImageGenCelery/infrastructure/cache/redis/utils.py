from typing import Optional

import redis

from config.config import Config
from infrastructure.cache.redis.redis import RedisClient

_config = Config()
_redis_client = RedisClient(_config).client


def delete_keys_by_pattern(
    pattern: str,
    client: Optional["redis.Redis"] = None,
    batch_size: int = 1000,
) -> int:
    """
    Delete all keys matching a Redis scan pattern.

    SCAN duyệt toàn bộ keyspace chứ không chỉ key khớp pattern, nên chi phí phụ
    thuộc vào số round-trip: scan_iter mặc định COUNT=10 và xoá từng key một sẽ
    tốn ~keyspace/10 + n_matched round-trip (hàng chục giây trên Redis lớn).
    Dùng COUNT lớn và DELETE theo lô để cắt số round-trip đi ~100 lần.

    Args:
        pattern: The pattern to match, e.g. 'prefix:*'.
        client: Optional redis client to use; defaults to the shared client.
        batch_size: SCAN COUNT hint và số key tối đa mỗi lệnh DELETE.

    Returns:
        Number of keys deleted.
    """
    redis_client = client or _redis_client
    deleted = 0
    batch = []
    for key in redis_client.scan_iter(pattern, count=batch_size):
        batch.append(key)
        if len(batch) >= batch_size:
            deleted += redis_client.delete(*batch)
            batch = []
    if batch:
        deleted += redis_client.delete(*batch)
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

