"""
top_down_placement.py
---------------------
Top-Down Cache Placement Policy

Strategy
--------
Traverse the tree hierarchy from root to leaves, greedily placing
services at each level based on profit potential. Services are ranked
by their total expected profit (revenue x omega x request count) and
placed at the highest level possible to maximize tree-wide coverage.

Algorithm
---------
For each node (starting from cloud root, moving toward leaves):
  1. Collect all pending (unsatisfied) requests passing through this node
  2. Group requests by service and compute total profit per service
  3. Sort services by profit (descending)
  4. INSERT services greedily until cache capacity is exhausted
  5. Mark requests as satisfied if their service was cached

Services cached higher in the tree benefit more downstream nodes,
naturally maximizing utility across the entire subtree.

Example
-------
Tree:           CLOUD (root)
                  |
               [Node A]
              /        \\
         [Node B]    [Node C]
         /      \\    /      \\
      Leaf1  Leaf2 Leaf3  Leaf4

Processing order: CLOUD -> Node A -> Node B -> Node C -> Leaves
At each node, cache highest-profit services not yet cached upstream.

Benefits
--------
- Maximizes tree-wide service coverage
- Naturally leverages hierarchy for shared caching
- Greedy by profit: prioritizes high-value services

Limitations
-----------
- Requires global knowledge of request traces
- Greedy may miss optimal placements (NP-hard problem)
- Does not adapt dynamically during provisioning intervals
"""

from typing import Dict, Set, List
from collections import defaultdict

from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service


class TopDownPlacementPolicy:
    """
    Top-down cache placement policy.
    
    Tracks request origins and profits per service, then greedily places
    services starting from the root and moving down the tree.
    """
    
    def __init__(self, topology):
        """
        Parameters
        ----------
        topology : NetworkTopology
            The network tree structure
        """
        self.topology = topology
        self.node_map = topology.node_map
        
        # Track requests per service at each node this interval
        # {node_id: {service_id: count}}
        self.node_service_requests: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # Track which services have been cached upstream (already satisfied)
        self.services_satisfied: Set[int] = set()
        
        # Track processing order: root to leaves
        self.processing_order: List[str] = self._build_processing_order()
        
        # Cache decisions made this interval
        self.cache_decisions: Dict[str, Set[int]] = defaultdict(set)
    
    def _build_processing_order(self) -> List[str]:
        """Build top-down node processing order (BFS from cloud root)."""
        order = []
        queue = [self.topology.cloud_root]
        
        while queue:
            node = queue.pop(0)
            order.append(node.node_id)
            queue.extend(node.children)
        
        return order
    
    def track_request(self, service_id: int, origin_node_id: str):
        """
        Track that a service was requested from a specific leaf node.
        
        We increment counters for all nodes on the path from leaf to root
        since the request must traverse this path.
        """
        # Trace path from leaf to root
        current_id = origin_node_id
        parent_map = self._get_parent_map()
        
        while True:
            self.node_service_requests[current_id][service_id] += 1
            if current_id not in parent_map:
                break
            current_id = parent_map[current_id]
    
    def _get_parent_map(self) -> Dict[str, str]:
        """Build parent mapping for path traversal."""
        parent_map = {}
        queue = [self.topology.cloud_root]
        
        while queue:
            node = queue.pop(0)
            for child in node.children:
                parent_map[child.node_id] = node.node_id
                queue.append(child)
        
        return parent_map
    
    def reset_interval(self):
        """Clear tracked requests for new provisioning interval."""
        self.node_service_requests.clear()
        self.services_satisfied.clear()
        self.cache_decisions.clear()
    
    def compute_all_decisions(self, catalogue: Dict[int, Service]):
        """
        Compute cache decisions for all nodes in top-down order.
        
        This is called once per provisioning interval before applying actions.
        """
        self.cache_decisions.clear()
        self.services_satisfied.clear()
        
        for node_id in self.processing_order:
            node = self.node_map.get(node_id)
            if not node:
                continue
            
            # Get current cache state
            snap = node.get_stats_snapshot()
            cached_services = snap["cached_services"]
            cache_util = snap["cache_utilization"]
            
            # Collect profit for each service at this node
            service_profits = {}
            
            for svc_id, count in self.node_service_requests[node_id].items():
                # Skip if already satisfied upstream
                if svc_id in self.services_satisfied:
                    continue
                
                service = catalogue.get(svc_id)
                if not service:
                    continue
                
                # Profit = revenue x omega x request count
                profit = service.revenue * service.omega * count
                service_profits[svc_id] = profit
            
            # Sort services by profit (descending)
            sorted_services = sorted(
                service_profits.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Greedy placement: insert highest-profit services
            for svc_id, profit in sorted_services:
                service = catalogue[svc_id]

                # Check if we have capacity
                remaining_capacity = node.cache_cap - cache_util
                if remaining_capacity < service.size:
                    continue

                # Cache this service
                self.cache_decisions[node_id].add(svc_id)
                self.services_satisfied.add(svc_id)

                # Update virtual cache utilization
                cache_util += service.size
    
    def get_action(self, node: EdgeNode, service: Service, snap: dict) -> int:
        """
        Decide INSERT/DELETE/KEEP action based on pre-computed top-down decisions.
        
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
        
        # Check pre-computed decision
        should_cache = svc_id in self.cache_decisions.get(node.node_id, set())
        
        if should_cache and not is_cached:
            return ACTION_INSERT
        elif not should_cache and is_cached:
            return ACTION_DELETE
        else:
            return ACTION_KEEP


# ---------------------------------------------------------------------------
# Policy factory function for compare_policies_topDown.py integration
# ---------------------------------------------------------------------------

def make_top_down_policy(topology):
    """
    Create Top-Down policy action function.
    
    Returns
    -------
    Callable that takes (node, service, snap) and returns action
    """
    td = TopDownPlacementPolicy(topology)
    
    def _policy(node: EdgeNode, service: Service, snap: dict) -> int:
        return td.get_action(node, service, snap)
    
    # Attach the TopDown instance so we can call methods
    _policy.td_instance = td
    
    return _policy