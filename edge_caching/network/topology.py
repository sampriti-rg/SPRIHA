"""
topology.py
-----------
Builds the N-Ary hierarchical edge network tree at runtime.

Structure
---------
  Level 0  : Cloud root (always has all services, unlimited cache)
  Level 1+ : Edge nodes with limited cache

The number of levels and children per node are runtime parameters
sourced from config.py (NUM_LEVELS, NUM_CHILDREN).
"""

from typing import List, Dict
from edge_caching.control.core.node import EdgeNode
from edge_caching.core.service import Service
from edge_caching.config import (
    NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP, NODE_CPU_CAPACITY,
    SCHEDULING_ALGORITHM
)
from edge_caching.network.cache_initialiser import CacheInitialiser, PlacementReport


class NetworkTopology:
    """
    Constructs and holds the N-Ary hierarchical edge network.

    Attributes
    ----------
    cloud_root   : EdgeNode - the cloud root (level 0)
    all_nodes    : list of all EdgeNodes (excluding cloud root)
    leaf_nodes   : list of leaf-level EdgeNodes (where users connect)
    node_map     : dict {node_id -> EdgeNode}
    """

    def __init__(
        self,
        num_levels   : int   = NUM_LEVELS,
        num_children : int   = NUM_CHILDREN,
        cache_cap    : int   = NODE_CACHE_CAP,
        cpu_capacity : float = NODE_CPU_CAPACITY,
        scheduling_algorithm : str = SCHEDULING_ALGORITHM,
    ):
        self.num_levels            = num_levels
        self.num_children          = num_children
        self.cache_cap             = cache_cap
        self.cpu_capacity          = cpu_capacity
        self.scheduling_algorithm  = scheduling_algorithm

        self.node_map  : Dict[str, EdgeNode] = {}
        self.all_nodes : List[EdgeNode]       = []
        self.leaf_nodes: List[EdgeNode]       = []

        # Build cloud root (level 0, unlimited cache)
        self.cloud_root = EdgeNode(
            node_id               = "CLOUD",
            level                 = 0,
            cache_cap             = 999,
            parent                = None,
            cpu_capacity          = cpu_capacity * 10,  # cloud has 10x edge CPU
            scheduling_algorithm  = scheduling_algorithm,
        )
        self.node_map["CLOUD"] = self.cloud_root

        # Recursively build the tree
        self._build_tree(self.cloud_root, current_level=1)

        # Wire peers at each level
        self._wire_peers()

    # ????????????????????????????????????????????????????????????????????????

    def _build_tree(self, parent: EdgeNode, current_level: int) -> None:
        """Recursively create child nodes up to num_levels."""
        if current_level > self.num_levels:
            return

        for i in range(self.num_children):
            node_id = f"L{current_level}_{len(self.all_nodes)}"
            # CPU capacity decreases with depth:
            # Level 1 = 100 %, Level 2 = 75 %, Level 3 = 50 %
            level_cpu = self.cpu_capacity * max(
                0.5, 1.0 - (current_level - 1) * 0.25
            )
            node = EdgeNode(
                node_id               = node_id,
                level                 = current_level,
                cache_cap             = self.cache_cap,
                parent                = parent,
                cpu_capacity          = level_cpu,
                scheduling_algorithm  = self.scheduling_algorithm,
            )
            parent.children.append(node)
            self.node_map[node_id] = node
            self.all_nodes.append(node)

            if current_level == self.num_levels:
                self.leaf_nodes.append(node)
            else:
                self._build_tree(node, current_level + 1)

    def _wire_peers(self) -> None:
        """
        For each internal node, its direct children become peers of each other.
        Peers allow lateral forwarding before going upward.
        """
        all_with_children = [self.cloud_root] + [
            n for n in self.all_nodes if n.children
        ]
        for parent_node in all_with_children:
            siblings = parent_node.children
            for node in siblings:
                node.peers = [s for s in siblings if s.node_id != node.node_id]

    def prime_cloud_cache(self, catalogue: Dict[int, Service]) -> None:
        """Seed the cloud root with all services (it always has everything)."""
        for svc in catalogue.values():
            self.cloud_root.cache.add(svc.service_id)
            self.cloud_root.cache_sizes[svc.service_id] = svc.size

    def priority_init(self, catalogue: Dict[int, Service]) -> PlacementReport:
        """
        One-time priority-based cache initialisation for all edge nodes.

        Services are ranked by (omega * revenue) and placed greedily from
        leaf level upward.  Each service is placed at MOST ONCE in the
        non-cloud network so that DDQN starts from a warm, sensible state
        instead of empty caches.

        Call this AFTER prime_cloud_cache() and BEFORE the simulation loop.
        The DDQN orchestrator then takes full control of all future
        INSERT / DELETE / KEEP decisions.

        Parameters
        ----------
        catalogue : {service_id -> Service}

        Returns
        -------
        PlacementReport  with a full audit trail of every placement decision
        """
        return CacheInitialiser.priority_init(
            all_nodes  = self.all_nodes,
            catalogue  = catalogue,
            num_levels = self.num_levels,
        )

    def summary(self) -> str:
        lines = ["=== Network Topology ==="]
        lines.append(f"  Cloud Root : {self.cloud_root.node_id}")
        lines.append(f"  Levels     : {self.num_levels}")
        lines.append(f"  Children   : {self.num_children}")
        lines.append(f"  Edge Nodes : {len(self.all_nodes)}")
        lines.append(f"  Leaf Nodes : {len(self.leaf_nodes)}")
        # Print tree level by level
        level_map: Dict[int, List[str]] = {}
        for node in self.all_nodes:
            level_map.setdefault(node.level, []).append(node.node_id)
        for lvl in sorted(level_map):
            lines.append(f"  Level {lvl}: {level_map[lvl]}")
        return "\n".join(lines)
