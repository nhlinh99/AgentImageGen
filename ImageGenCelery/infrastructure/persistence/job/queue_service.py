"""
Queue Service Module
Handles job queue management using Redis sorted sets (multi-lane: fast / medium / slow).
"""

import json
import time
from typing import Dict, Any, Optional, List, Tuple


from common_lib.base_services import BaseServiceSingleton
from infrastructure.cache.redis.redis import RedisClient
from config.config import Config
from domain.schema import JobPriority, ServiceType
from infrastructure.persistence.job.queue_lane import (
    lane_for_service_type,
    LANE_FAST,
    LANE_MEDIUM,
    LANE_SLOW,
    ALL_LANES,
    QueueLane,
)


def _decode_redis_str(raw: Any) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def format_queue_zset_member(job_id: str, service_type: str, user_id: str) -> str:
    """Redis ZSET member for waiting/processing queues: job_id:service_type:user_id."""
    uid = user_id if user_id is not None else ""
    return f"{job_id}:{service_type}:{uid}"


def queue_member_lookup_variants(job_id: str, service_type: str, user_id: str) -> List[str]:
    """Possible Redis ZSET member strings for ``zrank`` (new format + legacy two-part)."""
    seen = set()
    out: List[str] = []
    uid = user_id if user_id is not None else ""
    for m in (
        format_queue_zset_member(job_id, service_type, uid),
        format_queue_zset_member(job_id, service_type, ""),
        f"{job_id}:{service_type}",
    ):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def parse_queue_zset_member(member: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse queue member. Supports legacy ``job_id:service_type`` (returns user_id None).
    """
    parts = member.split(":", 2)
    if len(parts) == 3:
        return parts[0], parts[1], parts[2] if parts[2] != "" else None
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 1:
        return parts[0], "", None
    return "", "", None


class QueueService(BaseServiceSingleton):
    """
    Three parallel lanes (fast / medium / slow): separate Redis waiting + processing ZSETs per lane.
    Scheduler admits jobs per lane with independent concurrency limits; queue positions and ETA
    use one lane only (jobs do not wait behind other lanes).
    """

    def __init__(self, config: Config):
        super(QueueService, self).__init__(config)
        self.config = config

        self.redis_client = RedisClient(self.config)
        p = config.redis.prefix_key

        self.queue_key_fast = f"{p}:queue:waiting:fast"
        self.queue_key_medium = f"{p}:queue:waiting:medium"
        self.queue_key_slow = f"{p}:queue:waiting:slow"

        self.processing_jobs_key_fast = f"{p}:queue:processing:fast"
        self.processing_jobs_key_medium = f"{p}:queue:processing:medium"
        self.processing_jobs_key_slow = f"{p}:queue:processing:slow"

        self.locked_jobs_key = f"{p}:queue:locked_jobs"

        # Service types surfaced in stats + processing-time cache (see JobService.refresh_service_processing_times).
        # Video jobs use the slow lane (lane_for_service_type / queue_lane.py).
        self.service_queues = [
            ServiceType.INFERENCE.value,
        ]

    def _waiting_key(self, lane: QueueLane) -> str:
        if lane == LANE_FAST:
            return self.queue_key_fast
        if lane == LANE_MEDIUM:
            return self.queue_key_medium
        return self.queue_key_slow

    def _processing_key(self, lane: QueueLane) -> str:
        if lane == LANE_FAST:
            return self.processing_jobs_key_fast
        if lane == LANE_MEDIUM:
            return self.processing_jobs_key_medium
        return self.processing_jobs_key_slow

    def resolve_job_queue_state(
        self,
        job_id: str,
        service_type: str,
        user_id: str,
    ) -> Tuple[Optional[str], Optional[QueueLane], Optional[int]]:
        """
        Locate job in Redis via ZRANK (no full-queue ZRANGE scan).
        Returns (queue_type, lane, rank0) with queue_type \"waiting\"|\"processing\", rank0 0-based.
        """
        try:
            expected = lane_for_service_type(service_type)
            lanes_order = (expected,) + tuple(ln for ln in ALL_LANES if ln != expected)
            members = queue_member_lookup_variants(job_id, service_type, user_id)
            for lane in lanes_order:
                wkey = self._waiting_key(lane)
                for m in members:
                    r = self.redis_client.zrank(wkey, m)
                    if r is not None:
                        return ("waiting", lane, int(r))
                pkey = self._processing_key(lane)
                for m in members:
                    r = self.redis_client.zrank(pkey, m)
                    if r is not None:
                        return ("processing", lane, int(r))
            return (None, None, None)
        except Exception as e:
            self.logger.error(f"Error resolving queue state for job {job_id}: {str(e)}")
            return (None, None, None)

    def lane_queue_job_infos_prefix(
        self,
        queue_type: str,
        lane: QueueLane,
        end_rank_inclusive: int,
    ) -> List[Dict[str, Any]]:
        """
        Entries from rank 0 through end_rank_inclusive in one lane's waiting or processing ZSET.
        Compatible with calculate_wait_time_for_job iteration (job_id, service_type).
        """
        if end_rank_inclusive < 0:
            return []
        try:
            key = self._waiting_key(lane) if queue_type == "waiting" else self._processing_key(lane)
            chunk = self.redis_client.zrange(key, 0, end_rank_inclusive, withscores=True)
            current_time = time.time()
            jobs: List[Dict[str, Any]] = []
            if queue_type == "waiting":
                for job_data, score in chunk:
                    job_data_str = _decode_redis_str(job_data)
                    jid, job_service_type, uid = parse_queue_zset_member(job_data_str)
                    priority = self._score_to_priority(score)
                    jobs.append(
                        {
                            "job_id": jid,
                            "service_type": job_service_type,
                            "user_id": uid,
                            "score": score,
                            "queue_type": "waiting",
                            "lane": lane,
                            "priority": priority.value if priority else None,
                        }
                    )
            else:
                for rank, (job_data, score) in enumerate(chunk):
                    job_data_str = _decode_redis_str(job_data)
                    jid, job_service_type, uid = parse_queue_zset_member(job_data_str)
                    jobs.append(
                        {
                            "job_id": jid,
                            "service_type": job_service_type,
                            "user_id": uid,
                            "position": rank,
                            "score": score,
                            "queue_type": "processing",
                            "lane": lane,
                            "processing_start_time": score,
                            "processing_duration": current_time - score if current_time and score else 0,
                        }
                    )
            return jobs
        except Exception as e:
            self.logger.error(
                f"Error reading lane queue prefix ({queue_type}, {lane}, end={end_rank_inclusive}): {str(e)}"
            )
            return []

    def count_user_jobs_all_lanes(self, user_id: str) -> Dict[QueueLane, Dict[str, int]]:
        """Waiting/processing counts per lane for one user (single Redis pipeline round trip)."""
        uid_s = str(user_id)
        empty = {"waiting": 0, "processing": 0}
        out: Dict[QueueLane, Dict[str, int]] = {
            LANE_FAST: dict(empty),
            LANE_MEDIUM: dict(empty),
            LANE_SLOW: dict(empty),
        }
        try:
            pipe = self.redis_client.pipeline(transaction=False)
            for lane in ALL_LANES:
                pipe.zrange(self._waiting_key(lane), 0, -1, withscores=True)
                pipe.zrange(self._processing_key(lane), 0, -1, withscores=True)
            chunks = pipe.execute()
            idx = 0
            for lane in ALL_LANES:
                wchunk = chunks[idx]
                idx += 1
                pchunk = chunks[idx]
                idx += 1
                for job_data, _s in wchunk:
                    _jid, _st, uid = parse_queue_zset_member(_decode_redis_str(job_data))
                    if uid is not None and uid == uid_s:
                        out[lane]["waiting"] += 1
                for job_data, _s in pchunk:
                    _jid, _st, uid = parse_queue_zset_member(_decode_redis_str(job_data))
                    if uid is not None and uid == uid_s:
                        out[lane]["processing"] += 1
            return out
        except Exception as e:
            self.logger.error(f"Error counting user jobs all lanes: {str(e)}")
            return out

    def add_job_to_queue(
        self,
        job_id: str,
        priority: JobPriority,
        service_type: str,
        user_id: str,
        lane: Optional[QueueLane] = None,
    ) -> int:
        try:
            resolved_lane = lane if lane is not None else lane_for_service_type(service_type)
            timestamp = time.time()
            score = self._calculate_score(timestamp, priority)
            job_data = format_queue_zset_member(job_id, service_type, user_id)
            queue_key = self._waiting_key(resolved_lane)

            self.redis_client.zadd(queue_key, {job_data: score})

            queue_position = self.get_queue_position(job_id)
            if queue_position is None:
                return 0

            self.logger.info(
                f"Job {job_id} added to {resolved_lane} waiting queue at lane position {queue_position}"
            )
            return queue_position

        except Exception as e:
            self.logger.error(f"Error adding job to queue: {str(e)}")
            raise

    def get_next_job_for_scheduler(
        self,
        limit_fast: int,
        limit_medium: int,
        limit_slow: int,
    ) -> Optional[Tuple[str, int, str, QueueLane]]:
        """
        Admit the next job: try fast, then medium, then slow when each lane has capacity.
        """
        try:
            n_fast = self.get_processing_jobs_count(LANE_FAST)
            n_medium = self.get_processing_jobs_count(LANE_MEDIUM)
            n_slow = self.get_processing_jobs_count(LANE_SLOW)

            if n_fast < limit_fast:
                peek = self._peek_first_waiting(LANE_FAST)
                if peek:
                    job_id, position, service_type = peek
                    return (job_id, position, service_type, LANE_FAST)

            if n_medium < limit_medium:
                peek = self._peek_first_waiting(LANE_MEDIUM)
                if peek:
                    job_id, position, service_type = peek
                    return (job_id, position, service_type, LANE_MEDIUM)

            if n_slow < limit_slow:
                peek = self._peek_first_waiting(LANE_SLOW)
                if peek:
                    job_id, position, service_type = peek
                    return (job_id, position, service_type, LANE_SLOW)

            return None

        except Exception as e:
            self.logger.error(f"Error in get_next_job_for_scheduler: {str(e)}")
            return None

    def _peek_first_waiting(self, lane: QueueLane) -> Optional[Tuple[str, int, str]]:
        key = self._waiting_key(lane)
        result = self.redis_client.zrange(key, 0, 0, withscores=True)
        if not result:
            return None
        job_data, _score = result[0]
        job_data_str = _decode_redis_str(job_data)
        job_id, job_service_type, _u = parse_queue_zset_member(job_data_str)
        position = self.get_queue_position(job_id)
        pos = position if position is not None else 0
        return (job_id, pos, job_service_type)

    def get_queue_position(self, job_id: str) -> Optional[int]:
        """1-based position within this job's waiting lane only (lanes run in parallel)."""
        try:
            for lane in ALL_LANES:
                key = self._waiting_key(lane)
                chunk = self.redis_client.zrange(key, 0, -1, withscores=True)
                for rank, (job_data, _score) in enumerate(chunk):
                    job_data_str = _decode_redis_str(job_data)
                    jid = parse_queue_zset_member(job_data_str)[0]
                    if jid == job_id:
                        return rank + 1
            return None
        except Exception as e:
            self.logger.error(f"Error getting queue position: {str(e)}")
            return None

    def remove_job_from_queue(self, job_id: str) -> bool:
        try:
            for lane in ALL_LANES:
                key = self._waiting_key(lane)
                jobs_with_scores = self.redis_client.zrange(key, 0, -1, withscores=True)
                for job_data, _score in jobs_with_scores:
                    job_data_str = _decode_redis_str(job_data)
                    if job_data_str.startswith(f"{job_id}:"):
                        removed = self.redis_client.zrem(key, job_data)
                        if removed:
                            self.logger.info(f"Job {job_id} removed from {lane} waiting queue")
                            return True
            return False
        except Exception as e:
            self.logger.error(f"Error removing job from queue: {str(e)}")
            return False


    def get_queue_statistics(self) -> Dict[str, Any]:
        try:
            stats = {
                "total_jobs": 0,
                "lanes": {
                    LANE_FAST: {"waiting": 0, "processing": 0},
                    LANE_MEDIUM: {"waiting": 0, "processing": 0},
                    LANE_SLOW: {"waiting": 0, "processing": 0},
                },
                "service_queues": {},
                "priority_distribution": {},
                "average_wait_time": 0,
            }

            for lane in ALL_LANES:
                wkey = self._waiting_key(lane)
                pkey = self._processing_key(lane)
                stats["lanes"][lane]["waiting"] = self.redis_client.zcard(wkey)
                stats["lanes"][lane]["processing"] = self.redis_client.zcard(pkey)

            jobs_with_scores: List[Tuple[bytes, float]] = []
            for lane in ALL_LANES:
                wkey = self._waiting_key(lane)
                jobs_with_scores.extend(self.redis_client.zrange(wkey, 0, -1, withscores=True))

            service_jobs: Dict[str, List[Tuple[str, JobPriority]]] = {}
            for job_data, score in jobs_with_scores:
                job_data_str = _decode_redis_str(job_data)
                jid, st, _u = parse_queue_zset_member(job_data_str)
                priority = self._score_to_priority(score)
                service_jobs.setdefault(st, []).append((jid, priority))
                stats["priority_distribution"][priority.value] = (
                    stats["priority_distribution"].get(priority.value, 0) + 1
                )

            for service_name in self.service_queues:
                service_job_list = service_jobs.get(service_name, [])
                service_stats = {
                    "length": len(service_job_list),
                    "count": len(service_job_list),
                    "priority_distribution": {},
                }
                for _jid, priority in service_job_list:
                    service_stats["priority_distribution"][priority.value] = (
                        service_stats["priority_distribution"].get(priority.value, 0) + 1
                    )
                stats["service_queues"][service_name] = service_stats
                stats["total_jobs"] += service_stats["length"]

            stats["total_jobs"] = (
                stats["lanes"][LANE_FAST]["waiting"]
                + stats["lanes"][LANE_MEDIUM]["waiting"]
                + stats["lanes"][LANE_SLOW]["waiting"]
            )
            stats["average_wait_time"] = 0.0
            return stats
        except Exception as e:
            self.logger.error(f"Error getting queue statistics: {str(e)}")
            return {}


    def get_jobs_in_queue(
        self,
        service_type: Optional[str] = None,
        limit: int = 100,
        queue_type: str = "waiting",
        lane: Optional[QueueLane] = None,
    ) -> List[Dict[str, Any]]:
        try:
            if limit is None or limit <= 0:
                return []

            if queue_type == "processing":
                return self._get_jobs_in_processing_merged(service_type, limit, lane)
            return self._get_jobs_in_waiting_merged(service_type, limit, lane)
        except Exception as e:
            self.logger.error(f"Error getting jobs in queue: {str(e)}")
            return []

    def _get_jobs_in_waiting_merged(
        self,
        service_type: Optional[str],
        limit: int,
        lane: Optional[QueueLane] = None,
    ) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        position = 0
        lanes_iter: Tuple[QueueLane, ...] = (
            (lane,) if lane is not None and lane in ALL_LANES else ALL_LANES
        )
        for ln in lanes_iter:
            key = self._waiting_key(ln)
            chunk = self.redis_client.zrange(key, 0, -1, withscores=True)
            for job_data, score in chunk:
                job_data_str = _decode_redis_str(job_data)
                job_id, job_service_type, user_id = parse_queue_zset_member(job_data_str)
                if service_type and job_service_type != service_type:
                    continue
                position += 1
                priority = self._score_to_priority(score)
                jobs.append(
                    {
                        "job_id": job_id,
                        "service_type": job_service_type,
                        "user_id": user_id,
                        "position": position,
                        "score": score,
                        "queue_type": "waiting",
                        "lane": ln,
                        "priority": priority.value if priority else None,
                        "added_at": self._score_to_timestamp(score),
                    }
                )
                if len(jobs) >= limit:
                    return jobs
        return jobs

    def _get_jobs_in_processing_merged(
        self,
        service_type: Optional[str],
        limit: int,
        lane: Optional[QueueLane] = None,
    ) -> List[Dict[str, Any]]:
        current_time = time.time()

        if lane is not None and lane in ALL_LANES:
            jobs: List[Dict[str, Any]] = []
            pkey = self._processing_key(lane)
            chunk = self.redis_client.zrange(pkey, 0, -1, withscores=True)
            for rank, (job_data, score) in enumerate(chunk):
                job_data_str = _decode_redis_str(job_data)
                job_id, job_service_type, user_id = parse_queue_zset_member(job_data_str)
                if service_type and job_service_type != service_type:
                    continue
                jobs.append(
                    {
                        "job_id": job_id,
                        "service_type": job_service_type,
                        "user_id": user_id,
                        "position": rank,
                        "score": score,
                        "queue_type": "processing",
                        "lane": lane,
                        "processing_start_time": score,
                        "processing_duration": current_time - score if current_time and score else 0,
                    }
                )
                if len(jobs) >= limit:
                    break
            return jobs

        merged: List[Tuple[str, str, Optional[str], float, QueueLane]] = []
        for ln in ALL_LANES:
            pkey = self._processing_key(ln)
            chunk = self.redis_client.zrange(pkey, 0, -1, withscores=True)
            for job_data, score in chunk:
                job_data_str = _decode_redis_str(job_data)
                job_id, job_service_type, user_id = parse_queue_zset_member(job_data_str)
                merged.append((job_id, job_service_type, user_id, score, ln))
        merged.sort(key=lambda x: x[3])
        jobs = []
        for rank, (job_id, job_service_type, user_id, score, ln) in enumerate(merged):
            if service_type and job_service_type != service_type:
                continue
            jobs.append(
                {
                    "job_id": job_id,
                    "service_type": job_service_type,
                    "user_id": user_id,
                    "position": rank,
                    "score": score,
                    "queue_type": "processing",
                    "lane": ln,
                    "processing_start_time": score,
                    "processing_duration": current_time - score if current_time and score else 0,
                }
            )
            if len(jobs) >= limit:
                break
        return jobs

    def _calculate_score(self, timestamp: float, priority: JobPriority) -> float:
        priority_offset = {
            JobPriority.HIGH: 0,
            JobPriority.NORMAL: 100000,
            JobPriority.LOW: 200000,
        }
        return timestamp + priority_offset.get(priority, 0)

    def _score_to_priority(self, score: float) -> JobPriority:
        timestamp = int(score)
        offset = score - timestamp
        if offset < 500:
            return JobPriority.HIGH
        elif offset < 1500:
            return JobPriority.NORMAL
        return JobPriority.LOW

    def _score_to_timestamp(self, score: float) -> float:
        return int(score)

    def lock_job(self, job_id: str, priority: JobPriority, service_type: str, lock_duration: int = 3600) -> bool:
        try:
            if self.is_job_locked(job_id):
                self.logger.info(f"Job {job_id} is already locked")
                return False

            for lane in ALL_LANES:
                key = self._waiting_key(lane)
                jobs_with_scores = self.redis_client.zrange(key, 0, -1, withscores=True)
                for job_data, score in jobs_with_scores:
                    job_data_str = _decode_redis_str(job_data)
                    if job_data_str.startswith(f"{job_id}:"):
                        removed = self.redis_client.zrem(key, job_data)
                        if removed:
                            _jid, parsed_st, parsed_uid = parse_queue_zset_member(job_data_str)
                            payload = {
                                "job_id": job_id,
                                "priority": priority.value,
                                "service_type": parsed_st or service_type,
                                "user_id": parsed_uid or "",
                                "lane": lane,
                                "original_score": score,
                                "locked_at": int(time.time()),
                            }
                            self.redis_client.hset(self.locked_jobs_key, job_id, json.dumps(payload))
                            self.redis_client.expire(self.locked_jobs_key, lock_duration)
                            self.logger.info(f"Job {job_id} locked ({lane}) for {lock_duration}s")
                            return True

            self.logger.warning(f"Job {job_id} not found in queue to lock")
            return False
        except Exception as e:
            self.logger.error(f"Error locking job {job_id}: {str(e)}")
            return False

    def unlock_job(self, job_id: str) -> bool:
        try:
            if not self.is_job_locked(job_id):
                self.logger.info(f"Job {job_id} is not locked")
                return False

            locked_raw = self.redis_client.hget(self.locked_jobs_key, job_id)
            if not locked_raw:
                self.logger.warning(f"No locked job info found for {job_id}")
                return False

            raw = locked_raw.decode("utf-8") if isinstance(locked_raw, bytes) else locked_raw
            self.redis_client.hdel(self.locked_jobs_key, job_id)

            lock_user_id = ""
            try:
                info = json.loads(raw)
                priority = JobPriority(info.get("priority", JobPriority.NORMAL.value))
                service_type = info.get("service_type", "")
                lane = info.get("lane", LANE_MEDIUM)
                lock_user_id = info.get("user_id") or ""
                if lane not in ALL_LANES:
                    lane = lane_for_service_type(service_type)
                elif lane == LANE_SLOW and lane_for_service_type(service_type) != LANE_SLOW:
                    # Legacy two-lane: "slow" held all non-fast jobs; no service type is
                    # slow-lane anymore, so always remap to medium.
                    lane = LANE_MEDIUM
            except (json.JSONDecodeError, ValueError, TypeError):
                priority = JobPriority.NORMAL
                service_type = ServiceType.INFERENCE.value
                lane = LANE_MEDIUM
                lock_user_id = ""

            timestamp = time.time()
            score = self._calculate_score(timestamp, priority)
            job_data = format_queue_zset_member(job_id, service_type, lock_user_id)
            queue_key = self._waiting_key(lane)
            self.redis_client.zadd(queue_key, {job_data: score})
            queue_position = self.redis_client.zrank(queue_key, job_data)
            pos = (queue_position + 1) if queue_position is not None else 0
            self.logger.info(f"Job {job_id} unlocked to {lane} queue at position {pos}")
            return True
        except Exception as e:
            self.logger.error(f"Error unlocking job {job_id}: {str(e)}")
            return False

    def is_job_locked(self, job_id: str) -> bool:
        try:
            return self.redis_client.hexists(self.locked_jobs_key, job_id)
        except Exception as e:
            self.logger.error(f"Error checking lock status for job {job_id}: {str(e)}")
            return False

    def add_job_to_processing_queue(self, job_id: str, service_type: str, lane: QueueLane, user_id: str) -> bool:
        try:
            timestamp = time.time()
            job_data = format_queue_zset_member(job_id, service_type, user_id)
            pkey = self._processing_key(lane)
            self.redis_client.zadd(pkey, {job_data: timestamp})
            self.logger.info(f"Job {job_id} ({service_type}) added to {lane} processing queue")
            return True
        except Exception as e:
            self.logger.error(f"Error adding job to processing queue: {str(e)}")
            return False

    def remove_job_from_processing_queue(self, job_id: str, lane: Optional[QueueLane] = None) -> bool:
        try:
            lanes_to_search = [lane] if lane in ALL_LANES else list(ALL_LANES)
            for ln in lanes_to_search:
                pkey = self._processing_key(ln)
                jobs_with_scores = self.redis_client.zrange(pkey, 0, -1, withscores=True)
                for job_data, _score in jobs_with_scores:
                    job_data_str = _decode_redis_str(job_data)
                    if job_data_str.startswith(f"{job_id}:"):
                        removed = self.redis_client.zrem(pkey, job_data)
                        if removed:
                            self.logger.info(f"Job {job_id} removed from {ln} processing queue")
                            return True
            self.logger.warning(f"Job {job_id} not found in processing queues")
            return False
        except Exception as e:
            self.logger.error(f"Error removing job from processing queue: {str(e)}")
            return False


    def get_processing_job_order(self, job_id: str) -> Optional[int]:
        """0-based rank within this job's processing lane only (lanes run in parallel)."""
        try:
            for lane in ALL_LANES:
                pkey = self._processing_key(lane)
                chunk = self.redis_client.zrange(pkey, 0, -1, withscores=True)
                for rank, (job_data, _score) in enumerate(chunk):
                    job_data_str = _decode_redis_str(job_data)
                    jid = parse_queue_zset_member(job_data_str)[0]
                    if jid == job_id:
                        return rank
            return None
        except Exception as e:
            self.logger.error(f"Error getting processing job order: {str(e)}")
            return None

    def get_processing_jobs_count(self, lane: Optional[QueueLane] = None) -> int:
        try:
            if lane == LANE_FAST:
                return self.redis_client.zcard(self.processing_jobs_key_fast)
            if lane == LANE_MEDIUM:
                return self.redis_client.zcard(self.processing_jobs_key_medium)
            if lane == LANE_SLOW:
                return self.redis_client.zcard(self.processing_jobs_key_slow)
            return (
                self.redis_client.zcard(self.processing_jobs_key_fast)
                + self.redis_client.zcard(self.processing_jobs_key_medium)
                + self.redis_client.zcard(self.processing_jobs_key_slow)
            )
        except Exception as e:
            self.logger.error(f"Error getting processing jobs count: {str(e)}")
            return 0

    def get_processing_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        try:
            merged_raw = self._get_jobs_in_processing_merged(None, limit)
            out = []
            for j in merged_raw:
                out.append(
                    {
                        "job_id": j["job_id"],
                        "service_type": j["service_type"],
                        "user_id": j.get("user_id"),
                        "position": j["position"],
                        "processing_start_time": j["processing_start_time"],
                        "lane": j.get("lane"),
                    }
                )
            return out
        except Exception as e:
            self.logger.error(f"Error getting processing jobs: {str(e)}")
            return []
