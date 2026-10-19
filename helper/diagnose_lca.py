"""
Diagnostic script to analyze LCA placement policy behavior
"""
import sys
sys.path.insert(0, '.')

from edge_caching.network.topology import NetworkTopology
from edge_caching.core.service import generate_service_catalogue
from edge_caching.policies.lca_placement import make_lca_policy
from edge_caching.config import NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP

# Setup
topo = NetworkTopology(NUM_LEVELS, NUM_CHILDREN, NODE_CACHE_CAP)
catalogue = {s.service_id: s for s in generate_service_catalogue(50)}
lca_policy_fn = make_lca_policy(topo)
lca_instance = lca_policy_fn.lca_instance

print("=" * 80)
print("LCA PLACEMENT DIAGNOSTIC")
print("=" * 80)

# Simulate request tracking
print("\n[1] Simulating request tracking...")
leaf_ids = [node.node_id for node in topo.leaf_nodes]
print(f"    Leaf nodes: {leaf_ids[:4]}")

# Track some requests
test_services = list(catalogue.values())[:10]
for i, svc in enumerate(test_services):
    # Service 0: requested from all leaves (high demand)
    if i == 0:
        for leaf in leaf_ids:
            lca_instance.track_request(svc.service_id, leaf)
    # Service 1-3: requested from 2 leaves
    elif i <= 3:
        for leaf in leaf_ids[:2]:
            lca_instance.track_request(svc.service_id, leaf)
    # Service 4-6: requested from 1 leaf
    elif i <= 6:
        lca_instance.track_request(svc.service_id, leaf_ids[0])
    # Service 7-9: not requested (should be deleted)

print(f"    Tracked {len(lca_instance.service_requesters)} services with requests")

# Analyze LCA decisions
print("\n[2] Analyzing LCA computation...")
for svc_id, requesters in list(lca_instance.service_requesters.items())[:5]:
    lca_node = lca_instance._find_lca(requesters)
    service = catalogue[svc_id]
    print(f"    Service {svc_id}: {len(requesters)} requesters ? LCA = {lca_node}")
    print(f"        Requesters: {list(requesters)[:3]}")

# Test placement decisions
print("\n[3] Testing placement decisions across nodes...")
test_service = test_services[0]  # Most popular service
decisions = {'INSERT': [], 'DELETE': [], 'KEEP': []}

for node in topo.all_nodes:
    snap = node.get_stats_snapshot()
    action = lca_policy_fn(node, test_service, snap)
    action_name = ['INSERT', 'DELETE', 'KEEP'][action]
    decisions[action_name].append(node.node_id)

print(f"    Service {test_service.service_id} decisions:")
print(f"        INSERT at: {decisions['INSERT'][:3]}")
print(f"        DELETE at: {decisions['DELETE'][:3]}")
print(f"        KEEP at: {decisions['KEEP'][:3]}")

# Check if services with no requests get deleted
print("\n[4] Checking unused service handling...")
unused_service = test_services[7]  # Not tracked
for node in topo.all_nodes[:3]:
    snap = node.get_stats_snapshot()
    snap['cached_services'].add(unused_service.service_id)  # Pretend it's cached
    action = lca_policy_fn(node, unused_service, snap)
    action_name = ['INSERT', 'DELETE', 'KEEP'][action]
    print(f"    Node {node.node_id}: Unused service ? {action_name}")

# **CRITICAL**: Check if priority_init services get deleted
print("\n[5] CRITICAL: Checking if priority_init services survive...")
topo.priority_init(catalogue)
services_before = {node.node_id: len(node.cache) for node in topo.all_nodes}
print(f"    Services cached after priority_init: {sum(services_before.values())}")

# Simulate one provisioning cycle with NO new requests
lca_instance.service_requesters.clear()  # No new requests
actions_taken = {'INSERT': 0, 'DELETE': 0, 'KEEP': 0}

for node in topo.all_nodes:
    snap = node.get_stats_snapshot()
    for svc_id in list(snap['cached_services']):
        service = catalogue[svc_id]
        action = lca_policy_fn(node, service, snap)
        action_name = ['INSERT', 'DELETE', 'KEEP'][action]
        actions_taken[action_name] += 1

print(f"    Actions if NO requests tracked:")
print(f"        INSERT: {actions_taken['INSERT']}")
print(f"        DELETE: {actions_taken['DELETE']}")  # This will be HIGH!
print(f"        KEEP: {actions_taken['KEEP']}")

print("\n" + "=" * 80)
print("DIAGNOSIS:")
print("=" * 80)
print("? PROBLEM 1: Services without tracked requests ? DELETE")
print("   If no requests are tracked between intervals, ALL cached services deleted!")
print()
print("? PROBLEM 2: LCA placement may be too high in tree")
print("   Services cached near cloud, not at leaves where requests originate")
print()
print("? PROBLEM 3: Aggressive deletion on non-LCA nodes")
print("   Services not at LCA get deleted even if they could serve requests")
print()
print("=" * 80)
