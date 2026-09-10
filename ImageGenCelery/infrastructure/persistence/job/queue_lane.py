"""
Multi-lane queue: classify service types into fast / medium / slow Redis lanes.
"""

from typing import Literal, Tuple

QueueLane = Literal["fast", "medium", "slow"]

LANE_FAST: QueueLane = "fast"
LANE_MEDIUM: QueueLane = "medium"
LANE_SLOW: QueueLane = "slow"

ALL_LANES: Tuple[QueueLane, ...] = (LANE_FAST, LANE_MEDIUM, LANE_SLOW)

# ponytail: only one service type exists (ServiceType.INFERENCE) and it has no
# special-cased lane, so fast/slow sets are empty and everything falls to medium.
_FAST_SERVICE_TYPES = frozenset()
_SLOW_SERVICE_TYPES = frozenset()


def lane_for_service_type(service_type: str) -> QueueLane:
    if service_type in _FAST_SERVICE_TYPES:
        return LANE_FAST
    if service_type in _SLOW_SERVICE_TYPES:
        return LANE_SLOW
    return LANE_MEDIUM
