"""
test_request.py
---------------
Unit tests for edge_caching.core.request

Covers
------
- Request initialisation and field defaults
- remaining_deadline initialised to deadline_ms
- subtract_hop_latency return value and mutation
- is_deadline_valid boundary conditions
- Auto-incrementing request_id across instances
"""

import unittest

from edge_caching.core.request import Request


def _req(deadline: float = 200.0, svc: int = 0) -> Request:
    """Helper that creates a Request with minimal boilerplate."""
    return Request(
        service_id     = svc,
        origin_node_id = "L3_0",
        deadline_ms    = deadline,
        arrival_time   = 0.0,
    )


class TestRequestInitialisation(unittest.TestCase):
    """Fields are set correctly on construction."""

    def test_remaining_deadline_equals_deadline_ms(self):
        req = _req(deadline=150.0)
        self.assertEqual(req.remaining_deadline, 150.0)

    def test_status_defaults_to_pending(self):
        req = _req()
        self.assertEqual(req.status, "pending")

    def test_executed_at_node_defaults_to_none(self):
        req = _req()
        self.assertIsNone(req.executed_at_node)

    def test_service_id_stored(self):
        req = _req(svc=5)
        self.assertEqual(req.service_id, 5)

    def test_origin_node_stored(self):
        req = Request(service_id=0, origin_node_id="L2_3",
                      deadline_ms=100.0, arrival_time=10.0)
        self.assertEqual(req.origin_node_id, "L2_3")


class TestRequestIdAutoIncrement(unittest.TestCase):
    """Each new Request receives a strictly increasing request_id."""

    def test_ids_are_strictly_increasing(self):
        reqs = [_req() for _ in range(5)]
        ids = [r.request_id for r in reqs]
        for a, b in zip(ids, ids[1:]):
            self.assertLess(a, b)

    def test_ids_are_positive(self):
        req = _req()
        self.assertGreater(req.request_id, 0)


class TestSubtractHopLatency(unittest.TestCase):
    """subtract_hop_latency correctly mutates remaining_deadline."""

    def test_returns_true_when_deadline_still_valid(self):
        req = _req(deadline=200.0)
        result = req.subtract_hop_latency(50.0)
        self.assertTrue(result)

    def test_remaining_deadline_decremented(self):
        req = _req(deadline=200.0)
        req.subtract_hop_latency(50.0)
        self.assertAlmostEqual(req.remaining_deadline, 150.0)

    def test_returns_false_when_exactly_zero(self):
        req = _req(deadline=50.0)
        result = req.subtract_hop_latency(50.0)
        self.assertFalse(result)

    def test_returns_false_when_overdue(self):
        req = _req(deadline=30.0)
        result = req.subtract_hop_latency(50.0)
        self.assertFalse(result)

    def test_remaining_deadline_can_go_negative(self):
        req = _req(deadline=10.0)
        req.subtract_hop_latency(50.0)
        self.assertLess(req.remaining_deadline, 0)

    def test_multiple_hops_accumulate(self):
        req = _req(deadline=200.0)
        req.subtract_hop_latency(30.0)
        req.subtract_hop_latency(40.0)
        self.assertAlmostEqual(req.remaining_deadline, 130.0)


class TestIsDeadlineValid(unittest.TestCase):
    """is_deadline_valid reflects remaining_deadline sign."""

    def test_valid_when_positive(self):
        req = _req(deadline=100.0)
        self.assertTrue(req.is_deadline_valid())

    def test_invalid_when_zero(self):
        req = _req(deadline=50.0)
        req.subtract_hop_latency(50.0)
        self.assertFalse(req.is_deadline_valid())

    def test_invalid_when_negative(self):
        req = _req(deadline=10.0)
        req.subtract_hop_latency(100.0)
        self.assertFalse(req.is_deadline_valid())

    def test_valid_after_small_hop(self):
        req = _req(deadline=100.0)
        req.subtract_hop_latency(1.0)
        self.assertTrue(req.is_deadline_valid())


if __name__ == "__main__":
    unittest.main()
