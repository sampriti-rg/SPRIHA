"""
cache_initialiser.py
--------------------
One-time cache initialisation strategies for the N-Ary edge network.

Initialisation runs ONCE before the simulation starts.  After that the
DDQN orchestrator takes over all provisioning decisions (INSERT/DELETE/KEEP).

Strategy: priority_init
-----------------------
Services are ranked by a composite priority score:

    score = omega * revenue

where:
    omega   = CRITICALITY_WEIGHTS[criticality]  (mission_critical=3 .. best_effort=0.5)
    revenue = profit earned per local execution

Placement rule (each service placed at most ONCE in the non-cloud network):
    1. Sort services by score descending.
    2. For each service (highest score first):
       a. Walk levels from the deepest (leaf) up to level 1.
       b. At each level, pick the node with the most free capacity.
       c. If that node has enough free space for the service, place it there
          and stop - the service is NOT replicated to any other node.
       d. If no node at any level can fit it, the service stays cloud-only
          (it will still be served, just with higher hop latency).
    3. Cloud always holds all services (via prime_cloud_cache).

Result
------
    - Highest-priority services end up closest to end users (leaf level).
    - Lower-priority services are pushed toward level 1 or remain cloud-only.
    - Every node's cache is filled as much as possible without overflow.
    - The DDQN starts from a meaningful warm state instead of empty caches.
"""

from typing import Dict, List, Tuple
from edge_caching.core.node import EdgeNode
from edge_caching.core.service import Service


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def priority_score(service: Service) -> float:
    """
    Composite score used to rank services for cache placement.

    score = omega * revenue

    Higher score  -> more valuable to cache close to users.
    """
    return service.omega * service.revenue


# ---------------------------------------------------------------------------
# Placement report
# ---------------------------------------------------------------------------

class PlacementReport:
    """
    Records every placement decision made during initialisation.

    Attributes
    ----------
    placements     : list of (service_id, score, node_id, level) tuples
    cloud_only     : service_ids that could not fit in any edge node
    level_summary  : {level: [service_id, ...]} grouped summary
    """

    def __init__(self) -> None:
        self.placements  : List[Tuple[int, float, str, int]] = []
        self.cloud_only  : List[int]                         = []

    def record(self, svc_id: int, score: float, node_id: str, level: int) -> None:
        self.placements.append((svc_id, score, node_id, level))

    def record_cloud_only(self, svc_id: int) -> None:
        self.cloud_only.append(svc_id)

    @property
    def level_summary(self) -> Dict[int, List[int]]:
        summary: Dict[int, List[int]] = {}
        for svc_id, _, _, level in self.placements:
            summary.setdefault(level, []).append(svc_id)
        return summary

    def print_report(self) -> None:
        """Print a human-readable placement report to stdout."""
        divider = "-" * 72
        print(divider)
        print("  Cache Initialisation Report  (priority_init strategy)")
        print(divider)

        # Header
        print(f"  {'Service ID':<12} {'Score':>8}  {'Placed At':<14} {'Level':>6}")
        print(divider)

        for svc_id, score, node_id, level in self.placements:
            print(f"  {svc_id:<12} {score:>8.3f}  {node_id:<14} {'L' + str(level):>6}")

        if self.cloud_only:
            print(divider)
            print(f"  Cloud-only (no edge node had room): {self.cloud_only}")

        print(divider)

        # Level summary
        print("  Placement by level:")
        for level, svcs in sorted(self.level_summary.items()):
            print(f"    Level {level}: services {svcs}")
        if self.cloud_only:
            print(f"    Cloud only: {self.cloud_only}")

        print(divider)
        total_placed = len(self.placements)
        total_cloud  = len(self.cloud_only)
        print(f"  Placed at edge: {total_placed}  |  Cloud-only: {total_cloud}")
        print(divider)


# ---------------------------------------------------------------------------
# Initialiser
# ---------------------------------------------------------------------------

class CacheInitialiser:
    """
    One-time cache seeding for the N-Ary edge topology.

    Usage
    -----
    CacheInitialiser.priority_init(all_nodes, catalogue, num_levels)

    This is called once before the simulation loop.  The DDQN takes over
    afterwards and can override any of these placements via INSERT/DELETE.
    """

    @staticmethod
    def priority_init(
        all_nodes   : List[EdgeNode],
        catalogue   : Dict[int, Service],
        num_levels  : int,
    ) -> PlacementReport:
        """
        Rank services by (omega * revenue) and greedily place each one
        at the deepest level that has enough free capacity.

        Each service is placed at MOST ONCE in the non-cloud network.

        Parameters
        ----------
        all_nodes  : all EdgeNodes excluding the cloud root
        catalogue  : {service_id -> Service}
        num_levels : depth of the tree (leaf level number)

        Returns
        -------
        PlacementReport with full placement audit trail
        """
        report = PlacementReport()

        # Build a level map: {level: [EdgeNode, ...]} deepest first
        level_map: Dict[int, List[EdgeNode]] = {}
        for node in all_nodes:
            level_map.setdefault(node.level, []).append(node)

        # Sort levels deepest -> shallowest (leaf nodes first)
        sorted_levels = sorted(level_map.keys(), reverse=True)

        # Rank services by composite score, highest first
        ranked_services = sorted(
            catalogue.values(),
            key=priority_score,
            reverse=True,
        )

        placed: set = set()  # service_ids already assigned to an edge node

        for svc in ranked_services:
            assigned = False

            for level in sorted_levels:
                # At this level, pick the node with the most free capacity
                nodes_at_level = sorted(
                    level_map[level],
                    key=lambda n: n.free_capacity,
                    reverse=True,
                )
                best_node = nodes_at_level[0]

                if best_node.free_capacity >= svc.size:
                    # Place the service on this node
                    best_node.cache.add(svc.service_id)
                    best_node.cache_sizes[svc.service_id] = svc.size

                    score = priority_score(svc)
                    report.record(svc.service_id, score, best_node.node_id, level)
                    placed.add(svc.service_id)
                    assigned = True
                    break   # placed - do NOT replicate to any other level

            if not assigned:
                report.record_cloud_only(svc.service_id)

        return report
