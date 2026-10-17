"""
lca_placement.py
----------------
LCA (Lowest Common Ancestor) Cache Placement Policy

Strategy
--------
Cache services at the LCA (Lowest Common Ancestor) of requesting nodes
to minimize average network distance. This is a classical hierarchical
caching strategy used in CDN and edge computing systems.

Algorithm
---------
For each service:
  1. Track which leaf nodes have requested it during this interval
  2. Find the LCA node in the tree that covers all requesters
  3. INSERT at LCA if not cached there
  4. DELETE from nodes that are not on the path from requesters to LCA
  5. KEEP at all other nodes

Example
-------
Tree:           CLOUD
                  |
               [Node A]
              /        \\
         [Node B]    [Node C]
         /      \\    /      \\
      Leaf1  Leaf2 Leaf3  Leaf4

If Service X is requested by Leaf1 and Leaf3:
  - LCA = Node A (root covering both subtrees)
  - INSERT at Node A
  - DELETE from Node B, Node C (not serving this service's requests)
  - Leaves don't cache (only internal nodes)

Benefits
--------
- Minimize average latency by caching at optimal tree position
- Reduce redundant caching across multiple nodes
- Naturally adapts to request distribution patterns

Limitations
-----------
- Requires global knowledge of request origins
- Does not consider cache capacity constraints directly
- May not optimize for revenue or SLA violations
"""

from typing import Dict, Set, List
from collections import defaultdict

from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service


class LCAPlacementPolicy:
    """
    LCA-based cache placement policy.
    
    Tracks request origins per service and caches at the lowest common
    ancestor in the tree hierarchy.
    """
    
    def __init__(self, topology):
        """
        Parameters
        ----------
        topology : NetworkTopology
            The network tree structure used to compute LCA
        """
        self.topology = topology
        self.node_map = topology.node_map

        # Track which leaf nodes requested each service this interval
        # {service_id -> set of leaf node IDs}
        self.service_requesters: Dict[int, Set[str]] = defaultdict(set)

        # Historical demand tracking (don't delete services with recent history)
        # {service_id -> intervals_since_last_request}
        self.service_history: Dict[int, int] = defaultdict(int)

        # Build parent mapping for LCA computation
        self._build_parent_map()

        # Cache for computed paths (memoization)
        self._path_cache: Dict[str, List[str]] = {}

        # Cache for LCA computations (avoid recomputing for same set of nodes)
        self._lca_cache: Dict[frozenset, str] = {}

        # Node depth mapping for efficient LCA
        self._node_depth: Dict[str, int] = {}
        self._compute_node_depths()
    
    def _build_parent_map(self):
        """Build mapping of node_id -> parent_node for LCA algorithm."""
        self.parent_map: Dict[str, str] = {}

        # BFS from cloud root to build parent relationships
        queue = [self.topology.cloud_root]
        while queue:
            node = queue.pop(0)
            for child in node.children:
                self.parent_map[child.node_id] = node.node_id
                queue.append(child)

    def _compute_node_depths(self):
        """Compute depth of each node from root for efficient LCA."""
        self._node_depth["CLOUD"] = 0
        queue = [(self.topology.cloud_root, 0)]

        while queue:
            node, depth = queue.pop(0)
            self._node_depth[node.node_id] = depth
            for child in node.children:
                queue.append((child, depth + 1))
    
    def track_request(self, service_id: int, origin_node_id: str):
        """
        Track that a service was requested from a specific leaf node.
        
        Called during the fast process (request routing).
        """
        self.service_requesters[service_id].add(origin_node_id)
    
    def reset_interval(self):
        """Clear tracked requests for new provisioning interval."""
        # Update history before clearing
        for svc_id in self.service_history:
            if svc_id not in self.service_requesters:
                self.service_history[svc_id] += 1
            else:
                self.service_history[svc_id] = 0  # Reset if requested

        # Remove services not requested for 5+ intervals
        stale_services = [sid for sid, count in self.service_history.items() if count > 5]
        for sid in stale_services:
            del self.service_history[sid]

        self.service_requesters.clear()
        # Clear LCA cache but keep path cache as tree structure doesn't change
        self._lca_cache.clear()
    
    def _get_path_to_root(self, node_id: str) -> List[str]:
        """Return path from node to cloud root with caching."""
        # Check cache first
        if node_id in self._path_cache:
            return self._path_cache[node_id]

        # Compute path
        path = [node_id]
        current = node_id
        while current in self.parent_map:
            current = self.parent_map[current]
            path.append(current)

        # Cache the result
        self._path_cache[node_id] = path
        return path
    
    def _find_lca(self, node_ids: Set[str]) -> str:
        """
        Find the lowest common ancestor of a set of nodes with caching.

        Returns
        -------
        node_id of the LCA, or "CLOUD" if no common internal node
        """
        if not node_ids:
            return "CLOUD"

        if len(node_ids) == 1:
            # Single requester: LCA is its parent (first internal node)
            node_id = list(node_ids)[0]
            return self.parent_map.get(node_id, "CLOUD")

        # Check cache using frozenset for hashability
        cache_key = frozenset(node_ids)
        if cache_key in self._lca_cache:
            return self._lca_cache[cache_key]

        # Optimized LCA using depth information
        # Convert to list and get paths
        node_list = list(node_ids)
        paths = [self._get_path_to_root(nid) for nid in node_list]

        # Reverse paths so they go root -> leaf
        for p in paths:
            p.reverse()

        # Find common prefix (all paths start at CLOUD)
        lca = "CLOUD"
        min_len = min(len(p) for p in paths)

        for i in range(min_len):
            # Check if all paths have same node at position i
            first_node = paths[0][i]
            if all(p[i] == first_node for p in paths):
                lca = first_node
            else:
                break

        # Cache the result
        self._lca_cache[cache_key] = lca
        return lca
    
    def _is_on_path(self, node_id: str, leaf_ids: Set[str]) -> bool:
        """Check if node is on path from any leaf to root (optimized)."""
        # Use cached paths for efficiency
        for leaf_id in leaf_ids:
            path = self._get_path_to_root(leaf_id)
            if node_id in path:
                return True
        return False
    
    def _compute_service_value(self, service, request_count, node_id=None):
        """
        Calculate the value of a service based on revenue, omega, popularity, request count,
        and potential latency reduction from caching at this node.

        This enhanced value calculation considers:
        - Revenue potential (service.revenue * service.omega)
        - Popularity (how frequently requested)
        - Request volume (actual demand)
        - Node position benefit (deeper nodes serve requests faster)

        Parameters
        ----------
        service : Service
            The service instance for which to compute the value.
        request_count : int
            The number of requests for this service.
        node_id : str, optional
            The node where caching is being considered (for position-based weighting).

        Returns
        -------
        float
            The computed value for the service.
        """
        base_value = service.revenue * service.omega * service.popularity * request_count

        # Add bonus for caching at deeper (edge) nodes - they reduce latency more
        if node_id and node_id in self._node_depth:
            depth = self._node_depth[node_id]
            # Higher depth = closer to edge = more valuable
            # Use exponential weighting to significantly favor edge caching
            position_multiplier = 1.0 + (depth * 0.2)
            base_value *= position_multiplier

        return base_value
    
    def get_action(self, node: EdgeNode, service: Service, snap: dict) -> int:
        """
        Decide INSERT/DELETE/KEEP action based on enhanced LCA strategy with profit optimization.

        Parameters
        ----------
        node    : EdgeNode being provisioned
        service : Service being considered
        snap    : Current node statistics snapshot

        Returns
        -------
        action : ACTION_INSERT (0), ACTION_DELETE (1), or ACTION_KEEP (2)
        """
        svc_id = service.service_id
        is_cached = svc_id in snap["cached_services"]

        # Get requesters for this service this interval
        requesters = self.service_requesters.get(svc_id, set())

        if not requesters:
            # No requests THIS interval, but check history and popularity
            if is_cached:
                # Don't delete if:
                # 1. Service has been requested recently (within 3 intervals)
                # 2. Service has high popularity (likely to be requested soon)
                intervals_idle = self.service_history.get(svc_id, 0)
                if intervals_idle < 3 or service.popularity > 0.5:
                    return ACTION_KEEP  # Keep for now
                # 3. Service is high-value (mission critical or high revenue)
                if service.criticality in ["mission_critical", "high"]:
                    return ACTION_KEEP
                # Only delete if truly idle and low value
                return ACTION_DELETE
            return ACTION_KEEP

        # Compute LCA for this service's requesters
        lca_node_id = self._find_lca(requesters)

        # Calculate service value for profit-aware decisions
        request_count = len(requesters)
        service_value = self._compute_service_value(service, request_count, node.node_id)

        # Enhanced decision logic with profit optimization
        if node.node_id == lca_node_id:
            # This node is the LCA - should cache high-value services
            if not is_cached:
                # Check if we have capacity or if this service is valuable enough
                free_capacity = snap.get("free_capacity", 0)

                # Always try to cache if capacity available
                if free_capacity >= service.size:
                    return ACTION_INSERT

                # If no capacity, only insert if service is high-value
                # (orchestrator will handle eviction of lower-value services)
                if service_value > 0:  # Any positive value is worth considering
                    return ACTION_INSERT
            return ACTION_KEEP

        elif self._is_on_path(node.node_id, requesters):
            # Node is on path from requesters to LCA
            # CHANGED: Also cache at intermediate nodes for better coverage
            node_depth = self._node_depth.get(node.node_id, 0)

            if not is_cached:
                # Cache at deeper (closer to leaves) nodes if valuable
                free_capacity = snap.get("free_capacity", 0)
                # Cache if: high demand (3+ requesters) OR high-value service OR good capacity
                if (len(requesters) >= 3 and free_capacity >= service.size) or \
                   (service.criticality in ["mission_critical", "high"] and free_capacity >= service.size) or \
                   (node_depth >= 2 and free_capacity >= service.size * 2):  # Plenty of space at edge
                    return ACTION_INSERT
            else:
                # Keep valuable services on path
                if service_value > service.revenue * 0.5:  # More lenient threshold
                    return ACTION_KEEP
                # Only delete if really low value AND low popularity
                if service.popularity < 0.2:
                    return ACTION_DELETE
            return ACTION_KEEP

        else:
            # Node is not relevant for this service
            if is_cached:
                # But don't delete if service is critical or very popular
                if service.criticality == "mission_critical" or service.popularity > 0.7:
                    return ACTION_KEEP  # Keep it just in case
                return ACTION_DELETE
            return ACTION_KEEP


# ---------------------------------------------------------------------------
# Policy factory function for compare_policies_others.py integration
# ---------------------------------------------------------------------------

def make_lca_policy(topology):
    """
    Create LCA policy action function.
    
    Returns
    -------
    Callable that takes (node, service, snap) and returns action
    """
    lca = LCAPlacementPolicy(topology)
    
    def _policy(node: EdgeNode, service: Service, snap: dict) -> int:
        return lca.get_action(node, service, snap)
    
    # Attach the LCA instance so we can call reset_interval
    _policy.lca_instance = lca
    
    return _policy
