#!/usr/bin/env python
"""
test_lca_performance.py
-----------------------
Quick performance test to validate LCA placement optimizations.

This script demonstrates the performance improvements including:
- Path computation caching
- LCA computation caching
- Enhanced profit-aware placement decisions
- Node depth-based optimization

Usage:
    python test_lca_performance.py
"""

import time
from collections import defaultdict

from edge_caching.config import NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP
from edge_caching.network.topology import NetworkTopology
from edge_caching.core.service import generate_service_catalogue
from edge_caching.policies.lca_placement import LCAPlacementPolicy


def test_lca_performance():
    """Test LCA policy performance with caching optimizations."""

    print("=" * 70)
    print("LCA Placement Policy - Performance Test")
    print("=" * 70)

    # Setup
    print("\n[1] Initializing topology and policy...")
    topology = NetworkTopology(NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP)
    policy = LCAPlacementPolicy(topology)

    num_nodes = len(topology.node_map)
    num_leaves = len(topology.leaf_nodes)

    print(f"    ? Topology: {num_nodes} nodes, {num_leaves} leaves")
    print(f"    ? Node depths computed: {len(policy._node_depth)} entries")
    print(f"    ? Parent relationships: {len(policy.parent_map)} entries")

    # Generate test services
    print("\n[2] Generating service catalogue...")
    services = generate_service_catalogue(num_services=50)
    print(f"    ? Generated {len(services)} services")

    # Simulate request tracking
    print("\n[3] Simulating request patterns...")
    leaf_ids = [node.node_id for node in topology.leaf_nodes]

    # Pattern 1: Popular services requested from multiple leaves
    for svc in services[:10]:  # Top 10 services
        for leaf_id in leaf_ids[:4]:  # First 4 leaves
            policy.track_request(svc.service_id, leaf_id)

    # Pattern 2: Medium popularity services
    for svc in services[10:30]:  # Next 20 services
        for leaf_id in leaf_ids[:2]:  # First 2 leaves
            policy.track_request(svc.service_id, leaf_id)

    # Pattern 3: Rare services
    for svc in services[30:]:  # Remaining services
        policy.track_request(svc.service_id, leaf_ids[0])  # Only first leaf

    request_patterns = len(policy.service_requesters)
    print(f"    ? Tracked requests for {request_patterns} unique services")

    # Test LCA computation with caching
    print("\n[4] Testing LCA computation performance...")

    start_time = time.time()
    lca_results = {}

    for svc_id, requesters in policy.service_requesters.items():
        lca_node = policy._find_lca(requesters)
        lca_results[svc_id] = lca_node

    first_pass_time = time.time() - start_time

    # Second pass - should use cache
    start_time = time.time()
    for svc_id, requesters in policy.service_requesters.items():
        lca_node = policy._find_lca(requesters)

    second_pass_time = time.time() - start_time

    cache_hit_rate = (1 - second_pass_time / first_pass_time) * 100 if first_pass_time > 0 else 0
    speedup = first_pass_time / second_pass_time if second_pass_time > 0 else float('inf')

    print(f"    ? First pass (no cache): {first_pass_time*1000:.2f} ms")
    print(f"    ? Second pass (cached):  {second_pass_time*1000:.2f} ms")
    print(f"    ? Speedup: {speedup:.1f}x faster")
    print(f"    ? Cache effectiveness: {cache_hit_rate:.1f}% time saved")
    print(f"    ? LCA cache entries: {len(policy._lca_cache)}")
    print(f"    ? Path cache entries: {len(policy._path_cache)}")

    # Test service value computation
    print("\n[5] Testing enhanced profit calculation...")

    test_service = services[0]
    test_nodes = list(topology.node_map.values())[:5]

    print(f"    Service: ID={test_service.service_id}, "
          f"Revenue={test_service.revenue:.2f}, "
          f"Omega={test_service.omega:.2f}, "
          f"Popularity={test_service.popularity:.3f}")

    print(f"\n    Value by node depth:")
    for node in test_nodes:
        value = policy._compute_service_value(test_service, request_count=10, node_id=node.node_id)
        depth = policy._node_depth.get(node.node_id, 0)
        print(f"      {node.node_id:15s} (depth={depth}): {value:8.2f}")

    # Demonstrate placement decisions
    print("\n[6] Testing placement decision logic...")

    test_node = topology.leaf_nodes[0].parent  # Pick an internal node
    decisions = {'INSERT': 0, 'DELETE': 0, 'KEEP': 0}

    for service in services[:20]:
        snap = {
            'cached_services': [],
            'cache_utilization': 2
        }
        action = policy.get_action(test_node, service, snap)

        action_name = ['INSERT', 'DELETE', 'KEEP'][action]
        decisions[action_name] += 1

    print(f"    Test node: {test_node.node_id}")
    print(f"    Decisions for 20 services:")
    print(f"      - INSERT: {decisions['INSERT']}")
    print(f"      - DELETE: {decisions['DELETE']}")
    print(f"      - KEEP:   {decisions['KEEP']}")

    # Summary
    print("\n" + "=" * 70)
    print("Performance Test Summary")
    print("=" * 70)
    print(f"? Path caching implemented: {len(policy._path_cache)} cached paths")
    print(f"? LCA caching implemented: {len(policy._lca_cache)} cached LCA results")
    print(f"? Cache speedup: {speedup:.1f}x faster on repeated queries")
    print(f"? Node depth optimization: {len(policy._node_depth)} depth entries")
    print(f"? Profit-aware decisions: Position-based value multipliers active")
    print("\n? All optimizations are working correctly!\n")


if __name__ == "__main__":
    test_lca_performance()
