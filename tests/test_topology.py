"""
test_topology.py
----------------
Unit tests for edge_caching.network.topology (NetworkTopology)

Covers
------
- Cloud root node properties
- Total node count formula: sum(num_children^l, l=1..num_levels)
- Leaf node count equals num_children^num_levels
- Parent-child linkage
- Peer wiring (siblings only, self excluded)
- prime_cloud_cache populates cloud root cache
- summary() string contains key labels
"""

import unittest

from edge_caching.network.topology import NetworkTopology
from edge_caching.core.service import generate_service_catalogue


class TestTopologyCloudRoot(unittest.TestCase):

    def setUp(self):
        self.topo = NetworkTopology(num_levels=2, num_children=2, cache_cap=5)

    def test_cloud_root_id(self):
        self.assertEqual(self.topo.cloud_root.node_id, "CLOUD")

    def test_cloud_root_level_zero(self):
        self.assertEqual(self.topo.cloud_root.level, 0)

    def test_cloud_root_has_no_parent(self):
        self.assertIsNone(self.topo.cloud_root.parent)

    def test_cloud_root_large_cache(self):
        self.assertGreater(self.topo.cloud_root.cache_cap, 100)

    def test_cloud_root_in_node_map(self):
        self.assertIn("CLOUD", self.topo.node_map)


class TestTopologyNodeCounts(unittest.TestCase):

    def _expected_total(self, levels: int, children: int) -> int:
        return sum(children ** l for l in range(1, levels + 1))

    def _expected_leaves(self, levels: int, children: int) -> int:
        return children ** levels

    def test_binary_tree_2_levels(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        self.assertEqual(len(topo.all_nodes), self._expected_total(2, 2))

    def test_binary_tree_3_levels(self):
        topo = NetworkTopology(num_levels=3, num_children=2)
        self.assertEqual(len(topo.all_nodes), self._expected_total(3, 2))

    def test_ternary_tree_2_levels(self):
        topo = NetworkTopology(num_levels=2, num_children=3)
        self.assertEqual(len(topo.all_nodes), self._expected_total(2, 3))

    def test_leaf_count_binary_2_levels(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        self.assertEqual(len(topo.leaf_nodes), self._expected_leaves(2, 2))

    def test_leaf_count_binary_3_levels(self):
        topo = NetworkTopology(num_levels=3, num_children=2)
        self.assertEqual(len(topo.leaf_nodes), self._expected_leaves(3, 2))

    def test_single_level_single_child(self):
        topo = NetworkTopology(num_levels=1, num_children=1)
        self.assertEqual(len(topo.all_nodes), 1)
        self.assertEqual(len(topo.leaf_nodes), 1)


class TestTopologyNodeLinkage(unittest.TestCase):

    def setUp(self):
        self.topo = NetworkTopology(num_levels=2, num_children=2, cache_cap=5)

    def test_all_non_root_nodes_have_parent(self):
        for node in self.topo.all_nodes:
            self.assertIsNotNone(node.parent)

    def test_parent_child_relationship_consistent(self):
        for node in self.topo.all_nodes:
            self.assertIn(node, node.parent.children)

    def test_leaf_nodes_have_no_children(self):
        for leaf in self.topo.leaf_nodes:
            self.assertEqual(leaf.children, [])

    def test_all_nodes_in_node_map(self):
        for node in self.topo.all_nodes:
            self.assertIn(node.node_id, self.topo.node_map)
            self.assertIs(self.topo.node_map[node.node_id], node)


class TestTopologyPeerWiring(unittest.TestCase):

    def setUp(self):
        self.topo = NetworkTopology(num_levels=2, num_children=3, cache_cap=5)

    def test_siblings_are_peers_of_each_other(self):
        # For every internal node, each child should list the other children as peers
        all_with_children = [self.topo.cloud_root] + [
            n for n in self.topo.all_nodes if n.children
        ]
        for parent_node in all_with_children:
            siblings = parent_node.children
            for node in siblings:
                expected_peers = {s.node_id for s in siblings
                                  if s.node_id != node.node_id}
                actual_peers   = {p.node_id for p in node.peers}
                self.assertEqual(expected_peers, actual_peers)

    def test_node_not_its_own_peer(self):
        for node in self.topo.all_nodes:
            peer_ids = [p.node_id for p in node.peers]
            self.assertNotIn(node.node_id, peer_ids)

    def test_peer_count_matches_num_children_minus_one(self):
        topo = NetworkTopology(num_levels=2, num_children=3)
        # All level-1 nodes should each have 2 peers (3 children - self)
        level1 = [n for n in topo.all_nodes if n.level == 1]
        for node in level1:
            self.assertEqual(len(node.peers), 2)


class TestPrimeCloudCache(unittest.TestCase):

    def test_cloud_has_all_services_after_prime(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        catalogue_list = generate_service_catalogue(8, seed=42)
        catalogue = {s.service_id: s for s in catalogue_list}
        topo.prime_cloud_cache(catalogue)
        for svc_id in catalogue:
            self.assertIn(svc_id, topo.cloud_root.cache)

    def test_cloud_cache_sizes_populated(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        catalogue_list = generate_service_catalogue(4, seed=0)
        catalogue = {s.service_id: s for s in catalogue_list}
        topo.prime_cloud_cache(catalogue)
        for svc_id, svc in catalogue.items():
            self.assertEqual(topo.cloud_root.cache_sizes[svc_id], svc.size)


class TestTopologySummary(unittest.TestCase):

    def test_summary_contains_cloud(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        self.assertIn("CLOUD", topo.summary())

    def test_summary_contains_levels_value(self):
        topo = NetworkTopology(num_levels=3, num_children=2)
        self.assertIn("3", topo.summary())

    def test_summary_contains_children_value(self):
        topo = NetworkTopology(num_levels=2, num_children=4)
        self.assertIn("4", topo.summary())

    def test_summary_returns_string(self):
        topo = NetworkTopology(num_levels=2, num_children=2)
        self.assertIsInstance(topo.summary(), str)


if __name__ == "__main__":
    unittest.main()
