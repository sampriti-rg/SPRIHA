"""
test_cache_initialiser.py
--------------------------
Unit tests for edge_caching.network.cache_initialiser (CacheInitialiser)

Covers
------
- priority_score returns omega * revenue
- Highest-scored service is placed first
- Each service placed at most once in the non-cloud network
- Placement goes to the deepest level (leaf-first) when capacity allows
- A service that does not fit any edge node is recorded as cloud-only
- PlacementReport records correct node_id, level and score
- level_summary groups service_ids correctly by level
- Full topology integration via NetworkTopology.priority_init()
- Cache state of nodes after init: utilization within [0, 1]
- Services placed at a node are actually stored in node.cache
"""

import unittest
from typing import Dict

from edge_caching.core.node import EdgeNode
from edge_caching.core.service import Service
from edge_caching.network.cache_initialiser import (
    CacheInitialiser, PlacementReport, priority_score,
)
from edge_caching.network.topology import NetworkTopology
from edge_caching.core.service import generate_service_catalogue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(sid: int, size: int = 1, omega_crit: str = "standard",
         revenue: float = 5.0, pop: float = 0.5) -> Service:
    return Service(service_id=sid, size=size, popularity=pop,
                   criticality=omega_crit, revenue=revenue)


def _catalogue(*services: Service) -> Dict[int, Service]:
    return {s.service_id: s for s in services}


def _make_topology(num_levels: int = 3, num_children: int = 2,
                   cache_cap: int = 5) -> NetworkTopology:
    topo = NetworkTopology(num_levels=num_levels,
                           num_children=num_children,
                           cache_cap=cache_cap)
    return topo


# ---------------------------------------------------------------------------
# priority_score helper
# ---------------------------------------------------------------------------

class TestPriorityScore(unittest.TestCase):

    def test_score_equals_omega_times_revenue(self):
        svc = _svc(0, omega_crit="mission_critical", revenue=8.0)
        self.assertAlmostEqual(priority_score(svc), svc.omega * svc.revenue)

    def test_mission_critical_scores_higher_than_best_effort(self):
        mc = _svc(0, omega_crit="mission_critical", revenue=5.0)
        be = _svc(1, omega_crit="best_effort",      revenue=5.0)
        self.assertGreater(priority_score(mc), priority_score(be))

    def test_higher_revenue_increases_score(self):
        lo = _svc(0, omega_crit="standard", revenue=1.0)
        hi = _svc(1, omega_crit="standard", revenue=9.0)
        self.assertGreater(priority_score(hi), priority_score(lo))

    def test_zero_revenue_gives_zero_score(self):
        svc = _svc(0, omega_crit="mission_critical", revenue=0.0)
        self.assertAlmostEqual(priority_score(svc), 0.0)


# ---------------------------------------------------------------------------
# PlacementReport
# ---------------------------------------------------------------------------

class TestPlacementReport(unittest.TestCase):

    def setUp(self):
        self.report = PlacementReport()

    def test_empty_report(self):
        self.assertEqual(self.report.placements,  [])
        self.assertEqual(self.report.cloud_only,  [])
        self.assertEqual(self.report.level_summary, {})

    def test_record_placement(self):
        self.report.record(svc_id=0, score=15.0, node_id="L3_0", level=3)
        self.assertEqual(len(self.report.placements), 1)
        self.assertEqual(self.report.placements[0],
                         (0, 15.0, "L3_0", 3))

    def test_record_cloud_only(self):
        self.report.record_cloud_only(5)
        self.assertIn(5, self.report.cloud_only)

    def test_level_summary_groups_by_level(self):
        self.report.record(0, 10.0, "L3_0", 3)
        self.report.record(1, 8.0,  "L3_1", 3)
        self.report.record(2, 6.0,  "L2_0", 2)
        summary = self.report.level_summary
        self.assertIn(3, summary)
        self.assertIn(2, summary)
        self.assertEqual(sorted(summary[3]), [0, 1])
        self.assertEqual(summary[2], [2])

    def test_print_report_does_not_raise(self):
        self.report.record(0, 10.0, "L3_0", 3)
        self.report.record_cloud_only(7)
        try:
            self.report.print_report()
        except Exception as e:
            self.fail(f"print_report raised {e}")


# ---------------------------------------------------------------------------
# CacheInitialiser.priority_init - placement logic
# ---------------------------------------------------------------------------

class TestCacheInitialiserPlacement(unittest.TestCase):

    def test_each_service_placed_at_most_once(self):
        """No service_id should appear more than once across all node caches."""
        topo = _make_topology(num_levels=3, num_children=2, cache_cap=5)
        cat  = _catalogue(*[_svc(i, size=1) for i in range(8)])
        CacheInitialiser.priority_init(topo.all_nodes, cat, topo.num_levels)

        seen = {}
        for node in topo.all_nodes:
            for sid in node.cache:
                self.assertNotIn(sid, seen,
                    msg=f"Service {sid} placed at {node.node_id} AND {seen.get(sid)}")
                seen[sid] = node.node_id

    def test_highest_score_service_placed_at_leaf(self):
        """
        With a small catalogue and ample leaf capacity, the top-scored service
        should land at a leaf node (deepest level).
        """
        topo = _make_topology(num_levels=2, num_children=2, cache_cap=10)
        mc   = _svc(0, size=1, omega_crit="mission_critical", revenue=10.0)
        be   = _svc(1, size=1, omega_crit="best_effort",      revenue=1.0)
        cat  = _catalogue(mc, be)
        topo.prime_cloud_cache(cat)
        report = CacheInitialiser.priority_init(topo.all_nodes, cat,
                                                topo.num_levels)

        # service 0 (highest score) should be at level 2 (leaf)
        mc_placements = [(n, l) for sid, _, n, l in report.placements
                         if sid == 0]
        self.assertEqual(len(mc_placements), 1)
        self.assertEqual(mc_placements[0][1], topo.num_levels)

    def test_service_stored_in_node_cache(self):
        """Every placed service must actually be in the assigned node's cache."""
        topo = _make_topology(num_levels=2, num_children=2, cache_cap=10)
        cat  = _catalogue(*[_svc(i, size=1) for i in range(4)])
        topo.prime_cloud_cache(cat)
        report = CacheInitialiser.priority_init(topo.all_nodes, cat,
                                                topo.num_levels)

        for svc_id, _, node_id, _ in report.placements:
            node = topo.node_map[node_id]
            self.assertIn(svc_id, node.cache,
                msg=f"Service {svc_id} reported at {node_id} but not in node.cache")

    def test_cache_sizes_recorded_correctly(self):
        """cache_sizes[svc_id] must equal service.size for each placement."""
        topo = _make_topology(num_levels=2, num_children=2, cache_cap=10)
        cat  = {i: _svc(i, size=i % 3 + 1) for i in range(4)}
        topo.prime_cloud_cache(cat)
        CacheInitialiser.priority_init(topo.all_nodes, cat, topo.num_levels)

        for node in topo.all_nodes:
            for sid in node.cache:
                if sid in cat:
                    self.assertEqual(node.cache_sizes[sid], cat[sid].size)

    def test_node_capacity_not_exceeded(self):
        """No node's used_capacity should ever exceed its cache_cap."""
        topo = _make_topology(num_levels=3, num_children=2, cache_cap=5)
        cat  = _catalogue(*[_svc(i, size=2) for i in range(8)])
        topo.prime_cloud_cache(cat)
        CacheInitialiser.priority_init(topo.all_nodes, cat, topo.num_levels)

        for node in topo.all_nodes:
            self.assertLessEqual(node.used_capacity, node.cache_cap,
                msg=f"{node.node_id}: used={node.used_capacity} > cap={node.cache_cap}")

    def test_oversized_service_goes_cloud_only(self):
        """
        A service whose size > cache_cap of every edge node must be
        reported as cloud-only.
        """
        topo = _make_topology(num_levels=2, num_children=2, cache_cap=2)
        big  = _svc(0, size=5, omega_crit="mission_critical", revenue=10.0)
        cat  = _catalogue(big)
        topo.prime_cloud_cache(cat)
        report = CacheInitialiser.priority_init(topo.all_nodes, cat,
                                                topo.num_levels)

        self.assertIn(0, report.cloud_only)
        self.assertEqual(len(report.placements), 0)

    def test_placement_order_descending_score(self):
        """placements list must be ordered by descending score."""
        topo = _make_topology(num_levels=2, num_children=3, cache_cap=15)
        svcs = [
            _svc(0, size=1, omega_crit="best_effort",      revenue=2.0),
            _svc(1, size=1, omega_crit="standard",         revenue=5.0),
            _svc(2, size=1, omega_crit="high",             revenue=4.0),
            _svc(3, size=1, omega_crit="mission_critical", revenue=9.0),
        ]
        cat = _catalogue(*svcs)
        topo.prime_cloud_cache(cat)
        report = CacheInitialiser.priority_init(topo.all_nodes, cat,
                                                topo.num_levels)

        scores = [s for _, s, _, _ in report.placements]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_catalogue_produces_empty_report(self):
        topo   = _make_topology()
        report = CacheInitialiser.priority_init(topo.all_nodes, {}, topo.num_levels)
        self.assertEqual(report.placements,  [])
        self.assertEqual(report.cloud_only,  [])


# ---------------------------------------------------------------------------
# Integration via NetworkTopology.priority_init
# ---------------------------------------------------------------------------

class TestTopologyPriorityInit(unittest.TestCase):

    def setUp(self):
        self.topo = _make_topology(num_levels=3, num_children=2, cache_cap=5)
        cat_list  = generate_service_catalogue(8, seed=42)
        self.catalogue = {s.service_id: s for s in cat_list}
        self.topo.prime_cloud_cache(self.catalogue)

    def test_returns_placement_report(self):
        report = self.topo.priority_init(self.catalogue)
        self.assertIsInstance(report, PlacementReport)

    def test_cloud_still_has_all_services(self):
        self.topo.priority_init(self.catalogue)
        for sid in self.catalogue:
            self.assertIn(sid, self.topo.cloud_root.cache)

    def test_at_least_one_service_placed_at_leaf(self):
        report  = self.topo.priority_init(self.catalogue)
        leaf_placements = [l for _, _, _, l in report.placements
                           if l == self.topo.num_levels]
        self.assertGreater(len(leaf_placements), 0)

    def test_no_service_placed_twice_across_full_topology(self):
        self.topo.priority_init(self.catalogue)
        seen: Dict[int, str] = {}
        for node in self.topo.all_nodes:
            for sid in node.cache:
                self.assertNotIn(sid, seen,
                    msg=f"Service {sid} duplicated across edge nodes")
                seen[sid] = node.node_id

    def test_utilization_within_bounds_after_init(self):
        self.topo.priority_init(self.catalogue)
        for node in self.topo.all_nodes:
            util = node.cache_utilization()
            self.assertGreaterEqual(util, 0.0)
            self.assertLessEqual(util, 1.0)

    def test_init_strategy_none_leaves_edge_caches_empty(self):
        """Passing init_strategy='none' must leave all edge caches empty."""
        from edge_caching.utils.simulation.simulation import Simulation
        sim = Simulation(
            num_levels=2, num_children=2, num_services=4,
            num_episodes=1, req_per_ep=5, prov_interval=5,
            init_strategy="none",
        )
        for node in sim.topology.all_nodes:
            self.assertEqual(len(node.cache), 0,
                msg=f"{node.node_id} should be empty with init_strategy='none'")

    def test_init_strategy_priority_warms_edge_caches(self):
        """Passing init_strategy='priority' must result in at least one non-empty edge node."""
        from edge_caching.utils.simulation.simulation import Simulation
        sim = Simulation(
            num_levels=2, num_children=2, num_services=4,
            num_episodes=1, req_per_ep=5, prov_interval=5,
            init_strategy="priority",
        )
        warm_nodes = [n for n in sim.topology.all_nodes if len(n.cache) > 0]
        self.assertGreater(len(warm_nodes), 0)


if __name__ == "__main__":
    unittest.main()
