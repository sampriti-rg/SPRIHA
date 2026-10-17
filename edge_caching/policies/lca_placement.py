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
              /        \
         [Node B]    [Node C]
         /      \    /      \
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
        
        # Build parent mapping for LCA computation
        self._build_parent_map()
    
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
    
    def track_request(self, service_id: int, origin_node_id: str):
        """
        Track that a service was requested from a specific leaf node.
        
        Called during the fast process (request routing).
        """
        self.service_requesters[service_id].add(origin_node_id)
    
    def reset_interval(self):
        """Clear tracked requests for new provisioning interval."""
        self.service_requesters.clear()
    
    def _get_path_to_root(self, node_id: str) -> List[str]:
        """Return path from node to cloud root."""
        path = [node_id]
        while node_id in self.parent_map:
            node_id = self.parent_map[node_id]
            path.append(node_id)
        return path
    
    def _find_lca(self, node_ids: Set[str]) -> str:
        """
        Find the lowest common ancestor of a set of nodes.
        
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
        
        # Get paths to root for all requesters
        paths = [self._get_path_to_root(nid) for nid in node_ids]
        
        # Reverse paths so they go root -> leaf
        for p in paths:
            p.reverse()
        
        # Find common prefix (all paths start at CLOUD)
        lca = "CLOUD"
        min_len = min(len(p) for p in paths)
        
        for i in range(min_len):
            # Check if all paths have same node at position i
            nodes_at_i = set(p[i] for p in paths)
            if len(nodes_at_i) == 1:
                lca = nodes_at_i.pop()
            else:
                break
        
        return lca
    
    def _is_on_path(self, node_id: str, leaf_ids: Set[str]) -> bool:
        """Check if node is on path from any leaf to root."""
        for leaf_id in leaf_ids:
            path = self._get_path_to_root(leaf_id)
            if node_id in path:
                return True
        return False
    
    def get_action(self, node: EdgeNode, service: Service, snap: dict) -> int:
        """
        Decide INSERT/DELETE/KEEP action based on LCA strategy.
        
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
            # No requests this interval
            if is_cached:
                # Delete unused services to free space
                return ACTION_DELETE
            return ACTION_KEEP
        
        # Compute LCA for this service's requesters
        lca_node_id = self._find_lca(requesters)
        
        # Decision logic
        if node.node_id == lca_node_id:
            # This node is the LCA - should cache
            if not is_cached:
                return ACTION_INSERT
            return ACTION_KEEP
        
        elif self._is_on_path(node.node_id, requesters):
            # Node is on path from requesters to LCA - keep if cached
            return ACTION_KEEP
        
        else:
            # Node is not relevant for this service - delete if cached
            if is_cached:
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
