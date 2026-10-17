"""
node.py
-------
EdgeNode: A single node in the N-Ary hierarchical edge network.

Request processing flow
-----------------------
Every incoming request passes through receive_request(), which acts as the
sole entry point and dispatcher.  It delegates to one of two clearly separated
paths depending on whether the requested service is cached locally:

    receive_request()
        |
        |-- _is_service_cached()  -->  True  --> _handle_cache_hit()
        |                                              |
        |                                              +--> EDF queue insert
        |                                              +--> _execute_from_queue()
        |                                                        |
        |                                                        +--> revenue
        |                                                        +--> 'hit'
        |
        `-- _is_service_cached()  --> False --> _handle_cache_miss()
                                                     |
                                                     +--> try peer forward
                                                     +--> try parent forward
                                                     +--> cloud fallback
                                                     +--> 'dropped' if expired

Cache management
----------------
The DDQN orchestrator calls apply_action(INSERT | DELETE | KEEP) once per
provisioning interval (slow process).  The node's cache state then guides
receive_request() decisions in the fast process.

Statistics
----------
forwarded_count  : per-service miss counter, reset each provisioning interval
executed_count   : per-service successful delivery counter, reset each provisioning interval
served_count     : per-service EDF queue entry counter (cache hit confirmed, regardless
                   of whether EDF later drops or executes the request), reset each interval.
                   This is the DDQN placement-quality signal: it fires exactly when
                   _is_service_cached() returns True, at the boundary of the agent's
                   responsibility, independent of scheduling outcomes.
total_revenue    : cumulative revenue, reset each episode by orchestrator
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
    """
    A single node in the hierarchical edge network.

    Parameters
    ----------
    node_id   : unique identifier, e.g. 'L1_0', 'L2_3', 'CLOUD'
    level     : depth from cloud root  (0 = cloud, 1 = top edge, ...)
    cache_cap : total cache slot budget for this node
    parent    : reference to parent EdgeNode, None for the cloud root
    """

    def __init__(
        self,
        node_id      : str,
        level        : int,
        cache_cap    : int                = NODE_CACHE_CAP,
        parent       : Optional["EdgeNode"] = None,
        cpu_capacity : float              = NODE_CPU_CAPACITY,
    ):
        self.node_id      = node_id
        self.level        = level
        self.cache_cap    = cache_cap
        self.parent       = parent
        self.cpu_capacity = cpu_capacity  # cycles / ms available at this node

        self.children : List["EdgeNode"] = []
        self.peers    : List["EdgeNode"] = []   # same-level siblings

        # ── Cache state ───────────────────────────────────────────────────
        self.cache       : set            = set()   # cached service_ids
        self.cache_sizes : Dict[int, int] = {}      # service_id -> size

        # ── Per-interval statistics (reset by orchestrator) ───────────────
        # forwarded_count     : total cache-miss forwards (all paths)
        # forwarded_to_peer   : misses resolved by a peer node
        # forwarded_to_parent : misses forwarded to parent / cloud
        # executed_count      : successful local deliveries
        # served_count        : EDF queue entries (cache hit confirmed,
        #                       independent of EDF scheduling outcome)
        self.forwarded_count    : Dict[int, int] = {}
        self.forwarded_to_peer  : Dict[int, int] = {}
        self.forwarded_to_parent: Dict[int, int] = {}
        self.executed_count     : Dict[int, int] = {}
        self.served_count       : Dict[int, int] = {}

        # ── EDF request queue (min-heap on remaining_deadline) ────────────
        self._queue : List[Tuple] = []

        # ── Cumulative revenue (reset by orchestrator each episode) ───────
        self.total_revenue : float = 0.0

    # =========================================================================
    # Cache capacity properties
    # =========================================================================

    @property
    def used_capacity(self) -> int:
        """Total slot units currently occupied."""
        return sum(self.cache_sizes.values())

    @property
    def free_capacity(self) -> int:
        """Remaining slot units available for new services."""
        return self.cache_cap - self.used_capacity

    def cache_utilization(self) -> float:
        """Fraction of cache capacity in use  [0.0, 1.0]."""
        if self.cache_cap == 0:
            return 0.0
        return self.used_capacity / self.cache_cap

    # =========================================================================
    # Cache presence check  (Step 1 of request routing)
    # =========================================================================

    def _is_service_cached(self, service_id: int) -> bool:
        """
        Step 1 - Dispatcher predicate.

        Returns True if the requested service is available in this node's
        local cache.  This is the sole decision gate used by receive_request()
        to choose between the cache-hit path and the cache-miss path.
        """
        return service_id in self.cache

    # kept as a public alias for external callers (tests, orchestrator)
    def has_service(self, service_id: int) -> bool:
        return self._is_service_cached(service_id)

    # =========================================================================
    # Cache-HIT path  (Step 2a)
    # =========================================================================

    def _handle_cache_hit(
        self,
        request   : Request,
        catalogue : Dict[int, Service],
    ) -> str:
        """
        Step 2a - Cache Hit handler.

        Called when _is_service_cached() returns True.  Inserts the request
        into the EDF (Earliest Deadline First) priority queue, then
        immediately executes the front-of-queue request and credits revenue.

        The EDF queue ensures that requests with the tightest remaining
        deadlines are always served first, minimising deadline violations
        when multiple requests are waiting at the same node.

        Returns
        -------
        'hit'     - the front-of-queue request was executed successfully
        'dropped' - the front-of-queue request had already expired
        """
        # Credit placement BEFORE EDF scheduling: the service was present when
        # the request arrived.  served_count fires at the boundary of the DDQN's
        # responsibility regardless of whether EDF later drops the request.
        svc_id = request.service_id
        self.served_count[svc_id] = self.served_count.get(svc_id, 0) + 1

        request.node_log.append((self.node_id, "cache_hit"))
        return self._execute_from_queue(request, catalogue)

    def _execute_from_queue(
        self,
        request   : Request,
        catalogue : Dict[int, Service],
    ) -> str:
        """
        EDF scheduler for the cache-hit path.

        Inserts `request` into the min-heap keyed on remaining_deadline,
        then pops and executes the request with the earliest deadline.
        Revenue is earned only when execution occurs at this node.

        Returns
        -------
        'hit'     - execution completed, revenue credited
        'dropped' - popped request had an expired deadline
        """
        # Insert into EDF min-heap: smallest remaining_deadline served first
        heapq.heappush(
            self._queue,
            (request.remaining_deadline, request.request_id, request),
        )

        # Pop the highest-priority (earliest-deadline) request
        _, _, next_req = heapq.heappop(self._queue)

        # Final deadline check before execution
        if not next_req.is_deadline_valid():
            next_req.status = "dropped"
            next_req.node_log.append((self.node_id, "dropped"))
            catalogue[next_req.service_id].record_violation()
            return "dropped"

        # Execute: mark status, record executing node, credit revenue
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

    # =========================================================================
    # Cache-MISS path  (Step 2b)
    # =========================================================================

    def _handle_cache_miss(
        self,
        request   : Request,
        catalogue : Dict[int, Service],
    ) -> str:
        """
        Step 2b - Cache Miss handler.

        Called when _is_service_cached() returns False.  Counts the miss,
        then attempts to forward the request in priority order:

            1. Peer node (same level, peer hop latency = 5 ms)
               A peer is tried only if it holds the service in its cache.

            2. Parent node (one level up, parent hop latency = 10 ms)
               The request climbs the tree toward the cloud.

            3. Cloud root (level 0)
               Always has all services; falls through to _handle_cache_hit()
               at the cloud node.

        Each forwarding step deducts the appropriate hop latency from the
        request's remaining_deadline via subtract_hop_latency().  If the
        deadline is exhausted during any hop the request is dropped
        immediately and never forwarded further.

        Returns
        -------
        'hit'      - ultimately executed at a peer, ancestor, or cloud
        'dropped'  - deadline expired during forwarding
        """
        svc_id = request.service_id

        # Total miss counter (all forwarding paths)
        self.forwarded_count[svc_id] = self.forwarded_count.get(svc_id, 0) + 1

        # ── 1. Try a peer that holds the service ──────────────────────────
        for peer in self.peers:
            if peer._is_service_cached(svc_id):
                request.node_log.append((self.node_id, "cache_miss_peer"))
                catalogue[svc_id].record_forward()
                # Peer-path severity counter (weak INSERT signal)
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

        # ── 2. Forward to parent (or cloud) ──────────────────────────────
        if self.parent is not None:
            request.node_log.append((self.node_id, "cache_miss_parent"))
            catalogue[svc_id].record_forward()
            # Parent/cloud-path severity counter (strong INSERT signal)
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

        # ── 3. We are the cloud root — always executes ────────────────────
        request.node_log.append((self.node_id, "cache_miss_cloud"))
        return self._execute_from_queue(request, catalogue)

    # =========================================================================
    # Main entry point  (dispatcher)
    # =========================================================================

    def receive_request(
        self,
        request   : Request,
        catalogue : Dict[int, Service],
    ) -> str:
        """
        Entry point for every request arriving at this node.

        Responsibilities (in order):
            1. Stamp arrival time and node_id into hop_path (timing audit).
            2. Call _is_service_cached() to decide the routing path.
            3. Dispatch to _handle_cache_hit() or _handle_cache_miss().

        Timing contract
        ---------------
        When this method is called, request.node_arrival_time already holds
        the correct simulated clock for THIS node:
          - Set to arrival_time in __post_init__ for the origin node.
          - Re-derived by subtract_hop_latency() for every subsequent hop.

        Parameters
        ----------
        request   : the incoming Request object
        catalogue : {service_id -> Service} full service catalogue

        Returns
        -------
        'hit'     - request executed locally or at a reachable ancestor
        'dropped' - deadline expired before execution
        """
        # Step 1: stamp timing audit BEFORE any routing decision
        request.hop_path.append((self.node_id, request.node_arrival_time))

        svc_id = request.service_id

        # Record demand at the origin (leaf) node — only the first hop
        # counts as a user-originated request for this service
        if self.node_id == request.origin_node_id:
            catalogue[svc_id].record_demand()

        # Step 2 + 3: dispatch based on cache presence
        if self._is_service_cached(svc_id):
            return self._handle_cache_hit(request, catalogue)
        else:
            return self._handle_cache_miss(request, catalogue)

    # =========================================================================
    # DDQN action interface  (slow process)
    # =========================================================================

    def apply_action(
        self,
        action             : int,
        service            : Service,
        actual_interval_ms : float = 1.0,
    ) -> float:
        """
        Apply a DDQN provisioning decision for one service.

        Parameters
        ----------
        action             : INSERT / DELETE / KEEP
        service            : service being provisioned
        actual_interval_ms : elapsed simulated ms since the previous
                             provision cycle, passed from the orchestrator.
                             Used to compute miss_rate inside
                             _compute_placement_cost() without any
                             external rate constants.

        Returns
        -------
        placement_cost : cost of delay incurred by INSERT (0 otherwise)
        """
        if action == ACTION_INSERT:
            return self._insert_service(service, actual_interval_ms)
        elif action == ACTION_DELETE:
            self._delete_service(service.service_id)
            return 0.0
        else:
            return 0.0

    def _compute_placement_cost(
        self,
        service            : Service,
        actual_interval_ms : float,
    ) -> float:
        """
        Placement cost as Cost of Delay.

        Theoretical basis
        -----------------
        Reinertsen (2009): C_delay = revenue_rate * T_deploy
        Poularakis et al. (2019) INFOCOM:
            C_deploy(s,n) = d(s,n) * lambda * revenue

        Formula
        -------
        Step 1 — Deployment window  [ms]

            T_transfer = service.size * TRANSFER_TIME_PER_MB[path]
                         [MB] * [ms/MB]  =  [ms]

            T_init     = phi_s_init / F_n
                         [cycles] / [cycles/ms]        =  [ms]

            T_deploy   = T_transfer + T_init            [ms]

        Step 2 — Observed miss rate  [requests/ms]

            weighted_fwd = W_peer   * peer_fwd
                         + W_parent * parent_fwd        [requests]

            miss_rate    = weighted_fwd / actual_interval_ms
                           [requests] / [ms]            =  [requests/ms]

            actual_interval_ms is the MEASURED elapsed simulated time
            between provision cycles — no config constant required.

        Step 3 — Floor when no forwarding observed

            miss_rate = max(miss_rate, popularity / actual_interval_ms)

            popularity / actual_interval_ms is the EMA-derived prior
            demand rate — not an external constant.

        Step 4 — Revenue rate  [cost/ms]

            revenue_rate = miss_rate * service.revenue
                           [requests/ms] * [cost/request]  =  [cost/ms]

        Step 5 — Placement cost  [cost]

            C_place = revenue_rate * T_deploy
                      [cost/ms] * [ms]                  =  [cost]  ✓

        Dimensional analysis — every step closes cleanly.
        No external tuning constants. All inputs known at INSERT time.
        """
        path       = "cloud" if self.level <= 1 else "parent"

        # ── Step 1: deployment window ─────────────────────────────────────
        # T_transfer uses actual service size in MB and link bandwidth [ms/MB]
        # DNS_Resolver (2 MB)  -> 0.02 ms vs BackupService_Cold (4500 MB) -> 45 ms
        T_transfer = service.size * TRANSFER_TIME_PER_MB[path]               # [ms]
        # Cap T_init at 100ms to prevent ML-heavy services (init=7.2M cycles)
        # from producing T_init=7200ms which would make INSERT cost 500x higher
        # than lightweight services, causing the DDQN to never INSERT them.
        T_init     = min(
            service.init_cpu_cycles / max(self.cpu_capacity, 1.0),
            100.0,
        )                                                                     # [ms]
        T_deploy   = T_transfer + T_init                                      # [ms]

        # ── Step 2: severity-weighted observed miss rate ──────────────────
        peer_fwd   = self.forwarded_to_peer.get(service.service_id, 0)
        parent_fwd = self.forwarded_to_parent.get(service.service_id, 0)
        weighted_fwd = (
            FORWARD_WEIGHT["peer"]   * peer_fwd +
            FORWARD_WEIGHT["parent"] * parent_fwd
        )                                                                     # [requests]

        miss_rate = weighted_fwd / max(actual_interval_ms, 1.0)              # [requests/ms]

        # ── Step 3: popularity-based floor (EMA prior, no external const) ─
        floor_rate = service.popularity / max(actual_interval_ms, 1.0)       # [requests/ms]
        miss_rate  = max(miss_rate, floor_rate)

        # ── Step 4: revenue rate ──────────────────────────────────────────
        revenue_rate = miss_rate * service.revenue                            # [cost/ms]

        # ── Step 5: cost of delay ─────────────────────────────────────────
        return revenue_rate * T_deploy                                        # [cost]

    def _insert_service(
        self,
        service            : Service,
        actual_interval_ms : float,
    ) -> float:
        """
        Insert service into cache, evicting least-demanded service if
        needed, then charge the cost-of-delay placement cost.
        """
        if self._is_service_cached(service.service_id):
            return 0.0      # already present — no cost

        # Evict until there is room (capacity tracked in MB)
        while self.free_capacity < service.size and self.cache:
            self._evict_one()

        if self.free_capacity < service.size:
            return 0.0      # still no room (service larger than cap)

        self.cache.add(service.service_id)
        self.cache_sizes[service.service_id] = service.size

        return self._compute_placement_cost(service, actual_interval_ms)

    def _delete_service(self, service_id: int) -> None:
        """Remove a service from the local cache."""
        self.cache.discard(service_id)
        self.cache_sizes.pop(service_id, None)

    def _evict_one(self) -> None:
        """
        Evict the cached service with the lowest combined demand signal
        (forwarded_count + executed_count).  Least-demanded service leaves
        the cache first so that popular services are retained.
        """
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

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats_snapshot(self) -> dict:
        """Return a snapshot of current node statistics for the orchestrator."""
        return {
            "node_id"              : self.node_id,
            "cache_utilization"    : self.cache_utilization(),
            "free_capacity"        : self.free_capacity,
            "cpu_capacity"         : self.cpu_capacity,
            "forwarded_count"      : dict(self.forwarded_count),
            "forwarded_to_peer"    : dict(self.forwarded_to_peer),
            "forwarded_to_parent"  : dict(self.forwarded_to_parent),
            "executed_count"       : dict(self.executed_count),
            "served_count"         : dict(self.served_count),
            "cached_services"      : set(self.cache),
        }

    def reset_interval_stats(self) -> None:
        """Reset per-interval counters.  Called by orchestrator each interval."""
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
