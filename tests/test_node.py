"""
test_node.py
------------
Unit tests for edge_caching.core.node (EdgeNode)

Covers
------
- Cache capacity tracking (used / free)
- has_service / insert / delete / eviction logic
- apply_action (INSERT, DELETE, KEEP) return values
- placement cost calculation
- receive_request: cache hit, peer forward, parent forward, dropped
- EDF execution via _enqueue_and_execute
- Statistics snapshot and reset
"""

import unittest
from unittest.mock import MagicMock

from edge_caching.core.node import EdgeNode, ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service
from edge_caching.core.request import Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(sid: int = 0, size: int = 1, popularity: float = 0.8,
         criticality: str = "standard", revenue: float = 5.0) -> Service:
    return Service(service_id=sid, size=size, popularity=popularity,
                   criticality=criticality, revenue=revenue)


def _node(node_id: str = "L1_0", level: int = 1,
          cache_cap: int = 5) -> EdgeNode:
    return EdgeNode(node_id=node_id, level=level, cache_cap=cache_cap)


def _req(svc_id: int = 0, deadline: float = 200.0,
         origin: str = "L3_0") -> Request:
    return Request(service_id=svc_id, origin_node_id=origin,
                   deadline_ms=deadline, arrival_time=0.0)


def _catalogue(*services: Service):
    return {s.service_id: s for s in services}


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

class TestEdgeNodeCapacity(unittest.TestCase):

    def test_initial_used_capacity_is_zero(self):
        node = _node(cache_cap=5)
        self.assertEqual(node.used_capacity, 0)

    def test_initial_free_capacity_equals_cap(self):
        node = _node(cache_cap=5)
        self.assertEqual(node.free_capacity, 5)

    def test_cache_utilization_empty(self):
        self.assertEqual(_node().cache_utilization(), 0.0)

    def test_cache_utilization_full(self):
        node = _node(cache_cap=2)
        node.cache.add(0)
        node.cache_sizes[0] = 2
        self.assertAlmostEqual(node.cache_utilization(), 1.0)

    def test_cache_utilization_partial(self):
        node = _node(cache_cap=4)
        node.cache.add(0)
        node.cache_sizes[0] = 2
        self.assertAlmostEqual(node.cache_utilization(), 0.5)

    def test_zero_cap_utilization_safe(self):
        node = EdgeNode(node_id="X", level=1, cache_cap=0)
        self.assertEqual(node.cache_utilization(), 0.0)


# ---------------------------------------------------------------------------
# has_service / insert / delete
# ---------------------------------------------------------------------------

class TestEdgeNodeCacheOps(unittest.TestCase):

    def test_has_service_false_initially(self):
        node = _node()
        self.assertFalse(node.has_service(0))

    def test_insert_adds_service(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_INSERT, svc)
        self.assertTrue(node.has_service(0))

    def test_insert_idempotent(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_INSERT, svc)
        cost = node.apply_action(ACTION_INSERT, svc)
        self.assertEqual(cost, 0.0)
        self.assertEqual(node.used_capacity, 1)

    def test_delete_removes_service(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_INSERT, svc)
        node.apply_action(ACTION_DELETE, svc)
        self.assertFalse(node.has_service(0))

    def test_delete_nonexistent_is_safe(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=99, size=1)
        node.apply_action(ACTION_DELETE, svc)   # must not raise
        self.assertFalse(node.has_service(99))

    def test_keep_does_not_change_cache(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_KEEP, svc)
        self.assertFalse(node.has_service(0))

    def test_free_capacity_decreases_on_insert(self):
        node = _node(cache_cap=5)
        node.apply_action(ACTION_INSERT, _svc(sid=0, size=2))
        self.assertEqual(node.free_capacity, 3)

    def test_free_capacity_increases_on_delete(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=2)
        node.apply_action(ACTION_INSERT, svc)
        node.apply_action(ACTION_DELETE, svc)
        self.assertEqual(node.free_capacity, 5)


# ---------------------------------------------------------------------------
# Placement cost
# ---------------------------------------------------------------------------

class TestPlacementCost(unittest.TestCase):

    def test_cloud_path_cost_level_1(self):
        from edge_caching.config import TRANSFER_RATE
        node = _node(level=1, cache_cap=10)
        svc  = _svc(sid=0, size=2)
        cost = node.apply_action(ACTION_INSERT, svc)
        expected = 2 * TRANSFER_RATE["cloud"]
        self.assertAlmostEqual(cost, expected)

    def test_parent_path_cost_level_2(self):
        from edge_caching.config import TRANSFER_RATE
        node = _node(level=2, cache_cap=10)
        svc  = _svc(sid=0, size=3)
        cost = node.apply_action(ACTION_INSERT, svc)
        expected = 3 * TRANSFER_RATE["parent"]
        self.assertAlmostEqual(cost, expected)

    def test_delete_returns_zero_cost(self):
        node = _node(cache_cap=10)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_INSERT, svc)
        cost = node.apply_action(ACTION_DELETE, svc)
        self.assertEqual(cost, 0.0)

    def test_keep_returns_zero_cost(self):
        cost = _node(cache_cap=10).apply_action(ACTION_KEEP, _svc())
        self.assertEqual(cost, 0.0)


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------

class TestEviction(unittest.TestCase):

    def test_eviction_occurs_when_full(self):
        node = _node(cache_cap=2)
        svc0 = _svc(sid=0, size=1)
        svc1 = _svc(sid=1, size=1)
        svc2 = _svc(sid=2, size=1)
        node.apply_action(ACTION_INSERT, svc0)
        node.apply_action(ACTION_INSERT, svc1)
        # svc2 should trigger eviction of the least-demanded entry
        node.apply_action(ACTION_INSERT, svc2)
        self.assertEqual(node.used_capacity, 2)
        self.assertTrue(node.has_service(2))

    def test_least_demanded_service_evicted(self):
        node = _node(cache_cap=2)
        svc0 = _svc(sid=0, size=1)
        svc1 = _svc(sid=1, size=1)
        node.apply_action(ACTION_INSERT, svc0)
        node.apply_action(ACTION_INSERT, svc1)
        # Artificially raise demand for svc1
        node.forwarded_count[1] = 10
        node.executed_count[1]  = 10
        svc2 = _svc(sid=2, size=1)
        node.apply_action(ACTION_INSERT, svc2)
        # svc0 (demand=0) should be the victim, not svc1
        self.assertFalse(node.has_service(0))
        self.assertTrue(node.has_service(1))


# ---------------------------------------------------------------------------
# Request routing
# ---------------------------------------------------------------------------

class TestReceiveRequest(unittest.TestCase):

    def test_cache_hit_returns_hit(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1, revenue=5.0)
        cat  = _catalogue(svc)
        node.apply_action(ACTION_INSERT, svc)
        outcome = node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertEqual(outcome, "hit")

    def test_cache_miss_forwards_to_parent(self):
        parent = _node(node_id="L1_0", level=1, cache_cap=5)
        child  = _node(node_id="L2_0", level=2, cache_cap=5)
        child.parent = parent
        svc = _svc(sid=0, size=1, revenue=5.0)
        cat = _catalogue(svc)
        # Service only on parent
        parent.apply_action(ACTION_INSERT, svc)
        outcome = child.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertEqual(outcome, "hit")

    def test_request_dropped_when_deadline_expires_on_hop(self):
        parent = _node(node_id="L1_0", level=1, cache_cap=5)
        child  = _node(node_id="L2_0", level=2, cache_cap=5)
        child.parent = parent
        svc = _svc(sid=0, size=1)
        cat = _catalogue(svc)
        parent.apply_action(ACTION_INSERT, svc)
        # Deadline so tight it expires on the parent hop (10 ms)
        outcome = child.receive_request(_req(svc_id=0, deadline=5.0), cat)
        self.assertEqual(outcome, "dropped")

    def test_cache_miss_forwarded_to_peer(self):
        peer_node = _node(node_id="L2_1", level=2, cache_cap=5)
        node      = _node(node_id="L2_0", level=2, cache_cap=5)
        node.peers = [peer_node]
        svc = _svc(sid=0, size=1, revenue=5.0)
        cat = _catalogue(svc)
        peer_node.apply_action(ACTION_INSERT, svc)
        outcome = node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertEqual(outcome, "hit")

    def test_forwarded_count_incremented_on_miss(self):
        node = _node(cache_cap=5)
        # No parent, no peers -> falls through to cloud-like execution at root
        svc  = _svc(sid=0, size=1)
        cat  = _catalogue(svc)
        node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertGreaterEqual(node.forwarded_count.get(0, 0), 1)

    def test_executed_count_incremented_on_hit(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1, revenue=5.0)
        cat  = _catalogue(svc)
        node.apply_action(ACTION_INSERT, svc)
        node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertEqual(node.executed_count.get(0, 0), 1)

    def test_revenue_accumulated_on_hit(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1, revenue=4.0, criticality="standard")
        cat  = _catalogue(svc)
        node.apply_action(ACTION_INSERT, svc)
        node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        self.assertGreater(node.total_revenue, 0.0)


# ---------------------------------------------------------------------------
# Statistics snapshot and reset
# ---------------------------------------------------------------------------

class TestNodeStats(unittest.TestCase):

    def test_snapshot_contains_expected_keys(self):
        node = _node()
        snap = node.get_stats_snapshot()
        for key in ("node_id", "cache_utilization",
                    "forwarded_count", "executed_count", "cached_services"):
            self.assertIn(key, snap)

    def test_snapshot_node_id_matches(self):
        node = _node(node_id="L2_5")
        self.assertEqual(node.get_stats_snapshot()["node_id"], "L2_5")

    def test_reset_clears_forwarded_and_executed_counts(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1, revenue=5.0)
        cat  = _catalogue(svc)
        node.apply_action(ACTION_INSERT, svc)
        node.receive_request(_req(svc_id=0, deadline=200.0), cat)
        node.reset_interval_stats()
        self.assertEqual(node.forwarded_count, {})
        self.assertEqual(node.executed_count,  {})

    def test_reset_does_not_clear_cache(self):
        node = _node(cache_cap=5)
        svc  = _svc(sid=0, size=1)
        node.apply_action(ACTION_INSERT, svc)
        node.reset_interval_stats()
        self.assertTrue(node.has_service(0))


if __name__ == "__main__":
    unittest.main()
