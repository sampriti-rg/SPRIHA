"""
test_service.py
---------------
Unit tests for edge_caching.core.service

Covers
------
- Service dataclass attribute assignment
- omega property mapping for every criticality tier
- drift_popularity stays within [0.05, 1.0]
- generate_service_catalogue length, seeded reproducibility, and attribute ranges
"""

import unittest
import random

from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.config import CRITICALITY_WEIGHTS


class TestServiceOmega(unittest.TestCase):
    """omega property returns the correct weight for each criticality tier."""

    def _make(self, criticality: str) -> Service:
        return Service(service_id=0, size=1, popularity=0.5,
                       criticality=criticality, revenue=5.0)

    def test_omega_mission_critical(self):
        self.assertEqual(self._make("mission_critical").omega,
                         CRITICALITY_WEIGHTS["mission_critical"])

    def test_omega_high(self):
        self.assertEqual(self._make("high").omega,
                         CRITICALITY_WEIGHTS["high"])

    def test_omega_standard(self):
        self.assertEqual(self._make("standard").omega,
                         CRITICALITY_WEIGHTS["standard"])

    def test_omega_best_effort(self):
        self.assertEqual(self._make("best_effort").omega,
                         CRITICALITY_WEIGHTS["best_effort"])

    def test_omega_ordering(self):
        """mission_critical > high > standard > best_effort."""
        tiers = ["mission_critical", "high", "standard", "best_effort"]
        omegas = [self._make(t).omega for t in tiers]
        self.assertEqual(omegas, sorted(omegas, reverse=True))


class TestServiceDriftPopularity(unittest.TestCase):
    """drift_popularity stays within the valid [0.05, 1.0] band."""

    def setUp(self):
        self.svc = Service(service_id=1, size=2, popularity=0.5,
                           criticality="standard", revenue=4.0)

    def test_drift_stays_in_bounds(self):
        random.seed(0)
        for _ in range(1000):
            self.svc.drift_popularity(delta=0.05)
            self.assertGreaterEqual(self.svc.popularity, 0.05)
            self.assertLessEqual(self.svc.popularity, 1.0)

    def test_drift_lower_clamp(self):
        self.svc.popularity = 0.05
        random.seed(0)
        for _ in range(200):
            self.svc.drift_popularity(delta=0.2)
            self.assertGreaterEqual(self.svc.popularity, 0.05)

    def test_drift_upper_clamp(self):
        self.svc.popularity = 1.0
        random.seed(0)
        for _ in range(200):
            self.svc.drift_popularity(delta=0.2)
            self.assertLessEqual(self.svc.popularity, 1.0)

    def test_drift_changes_value(self):
        random.seed(99)
        before = self.svc.popularity
        self.svc.drift_popularity(delta=0.5)
        # With large delta there is a very high chance the value changes
        # (not a hard guarantee, but practically always true)
        changed_at_least_once = False
        for _ in range(20):
            self.svc.drift_popularity(delta=0.5)
            if self.svc.popularity != before:
                changed_at_least_once = True
                break
        self.assertTrue(changed_at_least_once)


class TestGenerateServiceCatalogue(unittest.TestCase):
    """generate_service_catalogue produces a correctly formed catalogue."""

    def test_length(self):
        cat = generate_service_catalogue(8)
        self.assertEqual(len(cat), 8)

    def test_unique_ids(self):
        cat = generate_service_catalogue(8)
        ids = [s.service_id for s in cat]
        self.assertEqual(ids, list(range(8)))

    def test_size_in_range(self):
        for svc in generate_service_catalogue(20, seed=1):
            self.assertIn(svc.size, [1, 2, 3])

    def test_popularity_in_range(self):
        for svc in generate_service_catalogue(20, seed=1):
            self.assertGreaterEqual(svc.popularity, 0.1)
            self.assertLessEqual(svc.popularity, 1.0)

    def test_revenue_in_range(self):
        for svc in generate_service_catalogue(20, seed=1):
            self.assertGreaterEqual(svc.revenue, 1.0)
            self.assertLessEqual(svc.revenue, 10.0)

    def test_criticality_valid(self):
        valid = set(CRITICALITY_WEIGHTS.keys())
        for svc in generate_service_catalogue(20, seed=1):
            self.assertIn(svc.criticality, valid)

    def test_seeded_reproducibility(self):
        cat_a = generate_service_catalogue(8, seed=42)
        cat_b = generate_service_catalogue(8, seed=42)
        for a, b in zip(cat_a, cat_b):
            self.assertEqual(a.service_id, b.service_id)
            self.assertEqual(a.size, b.size)
            self.assertAlmostEqual(a.popularity, b.popularity)
            self.assertEqual(a.criticality, b.criticality)
            self.assertAlmostEqual(a.revenue, b.revenue)

    def test_different_seeds_differ(self):
        cat_a = generate_service_catalogue(8, seed=1)
        cat_b = generate_service_catalogue(8, seed=2)
        sizes_a = [s.size for s in cat_a]
        sizes_b = [s.size for s in cat_b]
        self.assertNotEqual(sizes_a, sizes_b)

    def test_single_service(self):
        cat = generate_service_catalogue(1, seed=0)
        self.assertEqual(len(cat), 1)
        self.assertEqual(cat[0].service_id, 0)


if __name__ == "__main__":
    unittest.main()
