import time
import logging
import json
from typing import Any

import redis

from common_lib.base_services import BaseServiceSingleton
from common_lib.config import RedisSettings


def _resolve_redis_settings(config: Any) -> RedisSettings:
    """Accept either a full app config/settings (with a `.redis` sub-object) or a RedisSettings directly."""
    if isinstance(config, RedisSettings):
        return config
    return getattr(config, "redis", config)


class RedisClient(BaseServiceSingleton):
    def __init__(self, config: Any):
        self.config = config
        redis_settings = _resolve_redis_settings(config)
        self.client = redis.Redis(host=redis_settings.host,
                                  port=redis_settings.port,
                                  db=redis_settings.db,
                                  password=redis_settings.password,
                                #   decode_responses=True
                                  )
        self.max_try = 50
        self.time_sleep = 0.01
        self.logger = logging.getLogger(name=self.__class__.__name__)

    def get_data_by_key(self, key: str):
        n_try = 0
        result_bytes = self.client.get(key)
        if result_bytes is None:
            self.logger.info("Redis Waiting... keys:", key)
            while result_bytes is None:
                time.sleep(self.time_sleep)
                result_bytes = self.client.get(key)
                if result_bytes is not None:
                    break

                n_try +=1
                if n_try >= self.max_try:
                    self.logger.info('Key %s not existed!', key)
                    break
                self.logger.info('Retry get keys: %s, %s times', key, n_try)

        return result_bytes

    def set(self, key, value, expire=None):
        """Set key to hold the string value. Optional: expire time in seconds."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        return self.client.set(name=key, value=value, ex=expire)

    def get(self, key):
        """Get the value of key. Try to parse JSON."""
        value = self.client.get(name=key)
        if value:
            try:
                return value
            except json.JSONDecodeError:
                return value
        return None

    def delete(self, key):
        """Delete a key."""
        return self.client.delete(key)

    def exists(self, key):
        """Check if a key exists."""
        return self.client.exists(key) == 1

    def incr(self, key, amount=1):
        """Increment a key's value (must be an integer)."""
        return self.client.incr(name=key, amount=amount)

    def decr(self, key, amount=1):
        """Decrement a key's value (must be an integer)."""
        return self.client.decr(name=key, amount=amount)

    def expire(self, key, seconds):
        """Set a timeout on a key."""
        return self.client.expire(name=key, time=seconds)

    def ttl(self, key):
        """Get the time to live for a key."""
        return self.client.ttl(name=key)

    def keys(self, pattern='*'):
        """Get all keys matching pattern."""
        return self.client.keys(pattern)

    def flushdb(self):
        """Delete all keys in the current database."""
        return self.client.flushdb()

    def hset(self, name, key, value):
        """Set a field in a hash."""
        return self.client.hset(name, key, value)

    def hget(self, name, key):
        """Get a field from a hash."""
        return self.client.hget(name, key)

    def hgetall(self, name):
        """Get all fields from a hash."""
        return self.client.hgetall(name)

    def hdel(self, name, *keys):
        """Delete one or more fields from a hash."""
        return self.client.hdel(name, *keys)

    def hexists(self, name, key):
        """Check if a field exists in a hash."""
        return self.client.hexists(name, key)

    def hkeys(self, name):
        """Get all field names in a hash."""
        return self.client.hkeys(name)

    def zadd(self, name, mapping, nx=False, xx=False, ch=False, incr=False, gt=False, lt=False):
        """Add one or more members to a sorted set, or update its score if it already exists."""
        return self.client.zadd(name, mapping, nx=nx, xx=xx, ch=ch, incr=incr, gt=gt, lt=lt)

    def zrange(self, name, start, end, desc=False, withscores=False, score_cast_func=float):
        """Return a range of members from sorted set."""
        return self.client.zrange(name, start, end, desc=desc, withscores=withscores, score_cast_func=score_cast_func)

    def zrank(self, name, value):
        """Determine the index of a member in a sorted set."""
        return self.client.zrank(name, value)

    def zrem(self, name, *values):
        """Remove one or more members from a sorted set."""
        return self.client.zrem(name, *values)

    def zcard(self, name):
        """Get the number of members in a sorted set."""
        return self.client.zcard(name)

    def pipeline(self, transaction: bool = True):
        """Redis pipeline for batching commands (fewer round trips)."""
        return self.client.pipeline(transaction=transaction)

    def lpush(self, name, *values):
        """Push one or more values onto the left side of the list."""
        return self.client.lpush(name, *values)

    def lrange(self, name, start, end):
        """Get a range of elements from a list."""
        return self.client.lrange(name, start, end)

    def ltrim(self, name, start, end):
        """Trim the list to the specified range."""
        return self.client.ltrim(name, start, end)

    def ping(self):
        """Check Redis connection."""
        return self.client.ping()

    @staticmethod
    def create_redis_key(*args):
        return ":".join(args)
