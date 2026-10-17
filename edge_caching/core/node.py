"""
node.py
-------
EdgeNode: A single node in the N-Ary hierarchical edge network.
"""

import heapq
from typing import Optional, Dict, List, Tuple
from edge_caching.core.request import Request
from edge_caching.core.service import Service
from edge_caching.config import (
    HOP_LATENCY_MS, TRANSFER_TIME_PER_MB,
    NODE_CACHE_CAP, NODE_CPU_CAPACITY, FORWARD_WEIGHT,
)

ACTION_INSERT = 0
ACTION_DELETE = 1
ACTION_KEEP   = 2


class EdgeNode:
    def __init__(
        self,
        node_id      : str,
        level        : int,
        cache_cap    : int                  = NODE_CACHE_CAP,
        parent       : Optional["EdgeNode"] = None,
        cpu_capacity : float                = NODE_CPU_CAPACITY,
    ):
        self.node_id      = node_id
        self.level        = level
        self.cache_cap    = cache_cap
        self.parent       = parent
        self.cpu_capacity = cpu_capacity

        self.children : List["EdgeNode"] = []
        self.peers    : List["EdgeNode"] = []

        self.cache       : set            = set()
        self.cache_sizes : Dict[int, int] = {}

        self.forwarded_count    : Dict[int, int] = {}
        self.forwarded_to_peer  : Dict[int, int] = {}
        self.forwarded_to_parent: Dict[int, int] = {}
        self.executed_count     : Dict[int, int] = {}
        self.served_count       : Dict[int, int] = {}

        self._queue : List[Tuple] = []
        self.total_revenue : float = 0.0

    @property
    def used_capacity(self) -> int:
        return sum(self.cache_sizes.values())

    @property
    def free_capacity(self) -> int:
        return self.cache_cap - self.used_capacity

    def cache_utilization(self) -> float:
        if self.cache_cap == 0:
            return 0.0
        return self.used_capacity / self.cache_cap

    def _is_service_cached(self, service_id: int) -> bool:
        return service_id in self.cache

    def has_service(self, service_id: int) -> bool:
        return self._is_service_cached(service_id)

    def _handle_cache_hit(self, request: Request, catalogue: Dict[int, Service]) -> str:
        svc_id = request.service_id
        self.served_count[svc_id] = self.served_count.get(svc_id, 0) + 1
        request.node_log.append((self.node_id, "cache_hit"))
        return self._execute_from_queue(request, catalogue)

    def _execute_from_queue(self, request: Request, catalogue: Dict[int, Service]) -> str:
        heapq.heappush(
            self._queue,
            (request.remaining_deadline, request.request_id, request),
        )
        _, _, next_req = heapq.heappop(self._queue)

        if not next_req.is_deadline_valid():
            next_req.status = "dropped"
            next_req.node_log.append((self.node_id, "dropped"))
            catalogue[next_req.service_id].record_violation()
            return "dropped"

        next_req.status           = "executed"
        next_req.executed_at_node = self.node_id
        next_req.node_log.append((self.node_id, "executed"))

        svc     = catalogue[next_req.service_id]
        revenue = svc.revenue * svc.omega
        self.executed_count[svc.service_id] = (
            self.executed_count.get(svc.service_id, 0) + 1
        )
        self.total_revenue += revenue
        return "hit"

    def _handle_cache_miss(self, request: Request, catalogue: Dict[int, Service]) -> str:
        svc_id = request.service_id
        self.forwarded_count[svc_id] = self.forwarded_count.get(svc_id, 0) + 1

        for peer in self.peers:
            if peer._is_service_cached(svc_id):
                request.node_log.append((self.node_id, "cache_miss_peer"))
                catalogue[svc_id].record_forward()
                self.forwarded_to_peer[svc_id] = (
                    self.forwarded_to_peer.get(svc_id, 0) + 1
                )
                ok = request.subtract_hop_latency(HOP_LATENCY_MS["peer"])
                if not ok:
                    request.status = "dropped"
                    request.node_log.append((self.node_id, "dropped"))
                    catalogue[svc_id].record_violation()
                    return "dropped"
                return peer.receive_request(request, catalogue)

        if self.parent is not None:
            request.node_log.append((self.node_id, "cache_miss_parent"))
            catalogue[svc_id].record_forward()
            self.forwarded_to_parent[svc_id] = (
                self.forwarded_to_parent.get(svc_id, 0) + 1
            )
            ok = request.subtract_hop_latency(HOP_LATENCY_MS["parent"])
            if not ok:
                request.status = "dropped"
                request.node_log.append((self.node_id, "dropped"))
                catalogue[svc_id].record_violation()
                return "dropped"
            return self.parent.receive_request(request, catalogue)

        request.node_log.append((self.node_id, "cache_miss_cloud"))
        return self._execute_from_queue(request, catalogue)

    def receive_request(self, request: Request, catalogue: Dict[int, Service]) -> str:
        request.hop_path.append((self.node_id, request.node_arrival_time))
        svc_id = request.service_id

        if self.node_id == request.origin_node_id:
            catalogue[svc_id].record_demand()

        if self._is_service_cached(svc_id):
            return self._handle_cache_hit(request, catalogue)
        else:
            return self._handle_cache_miss(request, catalogue)

    def apply_action(
        self,
        action             : int,
        service            : Service,
        actual_interval_ms : float = 1.0,
    ) -> float:
        if action == ACTION_INSERT:
            return self._insert_service(service, actual_interval_ms)
        elif action == ACTION_DELETE:
            self._delete_service(service.service_id)
            return 0.0
        else:
            return 0.0

    def _compute_placement_cost(self, service: Service, actual_interval_ms: float) -> float:
        path = "cloud" if self.level <= 1 else "parent"

        T_transfer = service.size * TRANSFER_TIME_PER_MB[path]
        T_init     = min(
            service.init_cpu_cycles / max(self.cpu_capacity, 1.0),
            100.0,
        )
        T_deploy = T_transfer + T_init

        peer_fwd   = self.forwarded_to_peer.get(service.service_id, 0)
        parent_fwd = self.forwarded_to_parent.get(service.service_id, 0)
        weighted_fwd = (
            FORWARD_WEIGHT["peer"]   * peer_fwd +
            FORWARD_WEIGHT["parent"] * parent_fwd
        )

        miss_rate  = weighted_fwd / max(actual_interval_ms, 1.0)
        floor_rate = service.popularity / max(actual_interval_ms, 1.0)
        miss_rate  = max(miss_rate, floor_rate)

        revenue_rate = miss_rate * service.revenue
        return revenue_rate * T_deploy

    def _insert_service(self, service: Service, actual_interval_ms: float) -> float:
        if self._is_service_cached(service.service_id):
            return 0.0

        while self.free_capacity < service.size and self.cache:
            self._evict_one()

        if self.free_capacity < service.size:
            return 0.0

        self.cache.add(service.service_id)
        self.cache_sizes[service.service_id] = service.size
        return self._compute_placement_cost(service, actual_interval_ms)

    def _delete_service(self, service_id: int) -> None:
        self.cache.discard(service_id)
        self.cache_sizes.pop(service_id, None)

    def _evict_one(self) -> None:
        if not self.cache:
            return
        victim = min(
            self.cache,
            key=lambda sid: (
                self.forwarded_count.get(sid, 0) +
                self.executed_count.get(sid, 0)
            ),
        )
        self._delete_service(victim)

    def get_stats_snapshot(self) -> dict:
        return {
            "node_id"           : self.node_id,
            "cache_utilization" : self.cache_utilization(),
            "free_capacity"     : self.free_capacity,
            "cpu_capacity"      : self.cpu_capacity,
            "forwarded_count"   : dict(self.forwarded_count),
            "forwarded_to_peer" : dict(self.forwarded_to_peer),
            "forwarded_to_parent": dict(self.forwarded_to_parent),
            "executed_count"    : dict(self.executed_count),
            "served_count"      : dict(self.served_count),
            "cached_services"   : set(self.cache),
        }

    def reset_interval_stats(self) -> None:
        self.forwarded_count.clear()
        self.forwarded_to_peer.clear()
        self.forwarded_to_parent.clear()
        self.executed_count.clear()
        self.served_count.clear()

    def __repr__(self) -> str:
        return (
            f"EdgeNode(id={self.node_id}, level={self.level}, "
            f"cache={sorted(self.cache)}, util={self.cache_utilization():.2f})"
        )
