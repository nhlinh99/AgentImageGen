"""Port for the Redis client (connection lives in CommonLib, extended in
infrastructure/cache/redis for Celery's byte<->tensor/numpy/image helpers)."""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class IRedisClient(Protocol):
    """Thin wrapper over redis-py's command surface actually used by this service."""

    def get_data_by_key(self, key: str) -> Any:
        """Blocking get with retry/backoff until the key appears (or max_try is hit)."""

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> Any:
        """Set key to hold the string value (dict/list values are JSON-encoded). Optional TTL in seconds."""

    def get(self, key: str) -> Any:
        """Get the value of key."""

    def delete(self, key: str) -> Any:
        """Delete a key."""

    def exists(self, key: str) -> bool:
        """Check if a key exists."""

    def incr(self, key: str, amount: int = 1) -> Any:
        """Increment a key's integer value."""

    def decr(self, key: str, amount: int = 1) -> Any:
        """Decrement a key's integer value."""

    def expire(self, key: str, seconds: int) -> Any:
        """Set a timeout on a key."""

    def ttl(self, key: str) -> Any:
        """Get the time to live for a key."""

    def keys(self, pattern: str = "*") -> Any:
        """Get all keys matching pattern."""

    def flushdb(self) -> Any:
        """Delete all keys in the current database."""

    def hset(self, name: str, key: str, value: Any) -> Any:
        """Set a field in a hash."""

    def hget(self, name: str, key: str) -> Any:
        """Get a field from a hash."""

    def hgetall(self, name: str) -> Any:
        """Get all fields from a hash."""

    def hdel(self, name: str, *keys: str) -> Any:
        """Delete one or more fields from a hash."""

    def hexists(self, name: str, key: str) -> Any:
        """Check if a field exists in a hash."""

    def hkeys(self, name: str) -> Any:
        """Get all field names in a hash."""

    def zadd(self, name: str, mapping: dict, nx: bool = False, xx: bool = False,
              ch: bool = False, incr: bool = False, gt: bool = False, lt: bool = False) -> Any:
        """Add one or more members to a sorted set, or update its score if it already exists."""

    def zrange(self, name: str, start: int, end: int, desc: bool = False,
               withscores: bool = False, score_cast_func: Any = float) -> Any:
        """Return a range of members from a sorted set."""

    def zrank(self, name: str, value: Any) -> Any:
        """Determine the index of a member in a sorted set."""

    def zrem(self, name: str, *values: Any) -> Any:
        """Remove one or more members from a sorted set."""

    def zcard(self, name: str) -> Any:
        """Get the number of members in a sorted set."""

    def pipeline(self, transaction: bool = True) -> Any:
        """Redis pipeline for batching commands (fewer round trips)."""

    def lpush(self, name: str, *values: Any) -> Any:
        """Push one or more values onto the left side of a list."""

    def lrange(self, name: str, start: int, end: int) -> Any:
        """Get a range of elements from a list."""

    def ltrim(self, name: str, start: int, end: int) -> Any:
        """Trim a list to the specified range."""

    def ping(self) -> Any:
        """Check the Redis connection."""
