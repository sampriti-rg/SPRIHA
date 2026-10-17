"""
request.py
----------
Defines the Request data class representing a user request
for a specific service, with an associated deadline.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


_request_counter = 0


def _next_id() -> int:
    global _request_counter
    _request_counter += 1
    return _request_counter


@dataclass
class Request:
    """
    A user request for a service.

    Timing model (Strategy B - Cumulative Latency)
    -----------------------------------------------
    arrival_time       : simulated clock when request was GENERATED at the
                         origin leaf node.  This field is IMMUTABLE - it never
                         changes regardless of how many hops the request makes.

    remaining_deadline : starts equal to deadline_ms and is DECREMENTED by
                         each forwarding hop latency.  It represents the
                         remaining time budget the request has left.

    node_arrival_time  : simulated clock when the request entered the CURRENT
                         (or last visited) node, derived as:
                             arrival_time + (deadline_ms - remaining_deadline)
                         Updated at EVERY node entry (not just at execution),
                         so it correctly reflects the arrival time even for
                         dropped requests.

    hop_path           : ordered list of (node_id, arrival_time_at_node)
                         tuples, one entry appended each time the request
                         enters a new node.  Provides a full audit trail of
                         the request's journey through the topology.

    node_log           : ordered list of (node_id, event) tuples recording
                         the routing decision taken at each visited node.
                         Events: 'cache_hit' | 'cache_miss_peer' |
                                 'cache_miss_parent' | 'cache_miss_cloud' |
                                 'dropped' | 'executed'

    Attributes
    ----------
    request_id      : unique auto-incrementing id
    service_id      : which service is requested
    origin_node_id  : leaf node where the request was generated
    deadline_ms     : total latency budget available (ms)
    arrival_time    : simulated clock at generation - NEVER mutated
    remaining_deadline : remaining budget after all hops so far
    node_arrival_time  : simulated clock when request arrived at current node
    hop_path           : [(node_id, arrival_time_at_node), ...] timing log
    node_log           : [(node_id, event), ...] routing decision log
    executed_at_node   : node_id that finally executed the request
    status          : 'pending' | 'executed' | 'dropped'
    """
    service_id        : int
    origin_node_id    : str
    deadline_ms       : float
    arrival_time      : float
    request_id        : int                       = field(default_factory=_next_id)
    remaining_deadline: float                     = field(init=False)
    node_arrival_time : float                     = 0.0
    hop_path          : List[Tuple[str, float]]   = field(default_factory=list)
    node_log          : List[Tuple[str, str]]     = field(default_factory=list)
    executed_at_node  : Optional[str]             = None
    status            : str                       = "pending"

    def __post_init__(self):
        self.remaining_deadline = self.deadline_ms
        self.node_arrival_time  = self.arrival_time

    def subtract_hop_latency(self, latency_ms: float) -> bool:
        """
        Consume `latency_ms` of deadline budget for one forwarding hop.
        Returns True if the deadline is still valid, False if expired.
        """
        self.remaining_deadline -= latency_ms
        self.node_arrival_time = self.arrival_time + (
            self.deadline_ms - self.remaining_deadline
        )
        return self.remaining_deadline > 0

    def is_deadline_valid(self) -> bool:
        return self.remaining_deadline > 0

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, svc={self.service_id}, "
            f"origin={self.origin_node_id}, "
            f"node_arrival={self.node_arrival_time:.1f}ms, "
            f"remaining={self.remaining_deadline:.1f}ms, "
            f"status={self.status}, hops={len(self.hop_path)})"
        )
