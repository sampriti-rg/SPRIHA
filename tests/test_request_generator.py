"""
test_request_generator.py
--------------------------
Unit tests for edge_caching.utils.simulation.request_generator (RequestGenerator)

Covers
------
- generate() returns exactly num_requests requests
- All requests have a service_id present in the catalogue
- All requests have an origin_node_id belonging to a leaf node
- deadline_ms is within the specified deadline_range_ms
- arrival_time is strictly increasing across the batch
- Generator initial state (sim_time starts at 0)
- Different seeds produce different sequences
- Popularity-weighting: high-popularity services appear more often
- Poisson inter-arrival: exponential gap distribution
- Zipf service selection: head-heavy distribution matches alpha
- _zipf_weights helper: correct normalisation and rank ordering
- get_zipf_distribution: returns one entry per service, sums to 1
- expected_arrival_rate: correct for both models
- Zipf adapts to popularity drift between calls
"""

import math
import unittest
import numpy as np

from edge_caching.utils.simulation.request_generator import RequestGenerator, _zipf_weights
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.core.node import EdgeNode


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_leaf_nodes(n: int = 4) -> list:
    return [EdgeNode(node_id=f"L3_{i}", level=3, cache_cap=5) for i in range(n)]


def _make_catalogue(num_services: int = 8, seed: int = 42) -> dict:
    return {s.service_id: s
            for s in generate_service_catalogue(num_services, seed=seed)}


def _make_gen(seed: int = 42, num_services: int = 8, num_leaves: int = 4,
              model: str = "uniform"):
    cat    = _make_catalogue(num_services, seed=seed)
    leaves = _make_leaf_nodes(num_leaves)
    gen    = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                              seed=seed, model=model)
    return gen, cat, leaves


# ===========================================================================
# Existing tests (uniform model)
# ===========================================================================

class TestRequestGeneratorCount(unittest.TestCase):

    def test_generates_exact_count(self):
        gen, _, _ = _make_gen()
        self.assertEqual(len(gen.generate(20)), 20)

    def test_generates_zero_requests(self):
        gen, _, _ = _make_gen()
        self.assertEqual(len(gen.generate(0)), 0)

    def test_generates_one_request(self):
        gen, _, _ = _make_gen()
        self.assertEqual(len(gen.generate(1)), 1)

    def test_generates_large_batch(self):
        gen, _, _ = _make_gen()
        self.assertEqual(len(gen.generate(500)), 500)


class TestRequestGeneratorFields(unittest.TestCase):

    def setUp(self):
        self.gen, self.catalogue, self.leaf_nodes = _make_gen(seed=0)
        self.requests = self.gen.generate(50)

    def test_service_ids_in_catalogue(self):
        valid_ids = set(self.catalogue.keys())
        for req in self.requests:
            self.assertIn(req.service_id, valid_ids)

    def test_origin_node_ids_are_leaf_nodes(self):
        leaf_ids = {n.node_id for n in self.leaf_nodes}
        for req in self.requests:
            self.assertIn(req.origin_node_id, leaf_ids)

    def test_deadlines_within_default_range(self):
        for req in self.requests:
            self.assertGreaterEqual(req.deadline_ms,  50.0)
            self.assertLessEqual(req.deadline_ms, 300.0)

    def test_arrival_times_strictly_increasing(self):
        for a, b in zip(self.requests, self.requests[1:]):
            self.assertGreater(b.arrival_time, a.arrival_time)

    def test_all_requests_start_pending(self):
        for req in self.requests:
            self.assertEqual(req.status, "pending")


class TestRequestGeneratorDeadlineRange(unittest.TestCase):

    def test_custom_deadline_range_respected(self):
        gen, _, _ = _make_gen(seed=1)
        reqs = gen.generate(30, deadline_range_ms=(100, 200))
        for req in reqs:
            self.assertGreaterEqual(req.deadline_ms, 100.0)
            self.assertLessEqual(req.deadline_ms, 200.0)

    def test_tight_deadline_range(self):
        gen, _, _ = _make_gen(seed=2)
        reqs = gen.generate(20, deadline_range_ms=(99.0, 101.0))
        for req in reqs:
            self.assertGreaterEqual(req.deadline_ms,  99.0)
            self.assertLessEqual(req.deadline_ms, 101.0)


class TestRequestGeneratorInitialState(unittest.TestCase):

    def test_sim_time_zero_on_init(self):
        gen, _, _ = _make_gen(seed=5)
        self.assertEqual(gen._sim_time, 0.0)

    def test_arrival_time_positive_after_first_request(self):
        gen, _, _ = _make_gen(seed=5)
        req = gen.generate(1)[0]
        self.assertGreater(req.arrival_time, 0.0)

    def test_different_seeds_produce_different_sequences(self):
        cat    = _make_catalogue(seed=10)
        leaves = _make_leaf_nodes(4)
        gen_a  = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                                  seed=10, model="uniform")
        ids_a  = [r.service_id for r in gen_a.generate(30)]
        gen_b  = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                                  seed=99, model="uniform")
        ids_b  = [r.service_id for r in gen_b.generate(30)]
        self.assertNotEqual(ids_a, ids_b)


class TestRequestGeneratorPopularityBias(unittest.TestCase):

    def test_high_popularity_service_appears_most(self):
        catalogue = {
            0: Service(service_id=0, size=1, popularity=1.0,
                       criticality="standard", revenue=5.0),
            1: Service(service_id=1, size=1, popularity=0.01,
                       criticality="standard", revenue=5.0),
        }
        leaves  = _make_leaf_nodes(2)
        gen     = RequestGenerator(catalogue=catalogue, leaf_nodes=leaves,
                                   seed=0, model="uniform")
        reqs    = gen.generate(200)
        count_0 = sum(1 for r in reqs if r.service_id == 0)
        count_1 = sum(1 for r in reqs if r.service_id == 1)
        self.assertGreater(count_0, count_1)


# ===========================================================================
# Zipf weight helper
# ===========================================================================

class TestZipfWeights(unittest.TestCase):

    def _make_services(self, pops):
        return [Service(service_id=i, size=1, popularity=p,
                        criticality="standard", revenue=5.0)
                for i, p in enumerate(pops)]

    def test_weights_sum_to_one(self):
        services = self._make_services([0.9, 0.5, 0.3, 0.1])
        weights  = _zipf_weights(services, alpha=1.0)
        self.assertAlmostEqual(sum(weights), 1.0, places=10)

    def test_highest_popularity_gets_highest_weight(self):
        services = self._make_services([0.1, 0.9, 0.5])
        weights  = _zipf_weights(services, alpha=1.0)
        # service 1 has highest popularity -> rank 1 -> highest Zipf weight
        self.assertEqual(weights.index(max(weights)), 1)

    def test_lowest_popularity_gets_lowest_weight(self):
        services = self._make_services([0.1, 0.9, 0.5])
        weights  = _zipf_weights(services, alpha=1.0)
        self.assertEqual(weights.index(min(weights)), 0)

    def test_higher_alpha_increases_skew(self):
        services = self._make_services([0.9, 0.5, 0.3, 0.1])
        w_low  = _zipf_weights(services, alpha=0.5)
        w_high = _zipf_weights(services, alpha=2.0)
        # rank-1 weight should be proportionally larger with higher alpha
        self.assertGreater(w_high[0] / w_high[-1], w_low[0] / w_low[-1])

    def test_single_service_weight_is_one(self):
        services = self._make_services([0.7])
        weights  = _zipf_weights(services, alpha=1.2)
        self.assertAlmostEqual(weights[0], 1.0)

    def test_all_weights_positive(self):
        services = self._make_services([0.9, 0.5, 0.3, 0.1])
        weights  = _zipf_weights(services, alpha=1.2)
        for w in weights:
            self.assertGreater(w, 0.0)


# ===========================================================================
# Poisson inter-arrival model
# ===========================================================================

class TestPoissonInterArrival(unittest.TestCase):
    """
    Verify that the poisson_zipf model produces Exponential inter-arrival gaps.

    For a Poisson(lambda) process, inter-arrival times T ~ Exp(lambda).
    Properties tested:
      - mean(T) ? 1/lambda  (within 25% tolerance over 2000 samples)
      - All gaps > 0
      - Monotonically increasing arrival times
    """

    def setUp(self):
        self.lam  = 2.0
        self.gen, self.cat, _ = _make_gen(seed=0, model="poisson_zipf")
        self.gen.arrival_rate = self.lam
        self.requests = self.gen.generate(2000)

    def test_arrival_times_strictly_increasing(self):
        for a, b in zip(self.requests, self.requests[1:]):
            self.assertGreater(b.arrival_time, a.arrival_time)

    def test_all_gaps_positive(self):
        for a, b in zip(self.requests, self.requests[1:]):
            self.assertGreater(b.arrival_time - a.arrival_time, 0.0)

    def test_mean_gap_close_to_one_over_lambda(self):
        gaps = [b.arrival_time - a.arrival_time
                for a, b in zip(self.requests, self.requests[1:])]
        expected_mean = 1.0 / self.lam
        actual_mean   = sum(gaps) / len(gaps)
        # Allow 25% relative error (exponential distribution has high variance)
        self.assertAlmostEqual(actual_mean, expected_mean,
                               delta=0.25 * expected_mean)

    def test_first_arrival_positive(self):
        self.assertGreater(self.requests[0].arrival_time, 0.0)

    def test_count_matches_requested(self):
        self.assertEqual(len(self.requests), 2000)

    def test_poisson_model_faster_than_uniform_at_high_rate(self):
        """Higher lambda => smaller mean gap than uniform Uniform(1,5)."""
        lam_fast = 5.0   # mean gap = 0.2 ms  <  Uniform(1,5) mean = 3 ms
        cat    = _make_catalogue()
        leaves = _make_leaf_nodes()
        gen    = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                                  seed=7, model="poisson_zipf",
                                  arrival_rate=lam_fast)
        reqs   = gen.generate(500)
        gaps   = [b.arrival_time - a.arrival_time
                  for a, b in zip(reqs, reqs[1:])]
        self.assertLess(sum(gaps) / len(gaps), 1.0)


# ===========================================================================
# Zipf service selection model
# ===========================================================================

class TestZipfServiceSelection(unittest.TestCase):
    """
    Verify that the poisson_zipf model selects services according to Zipf.

    For a steep alpha (e.g. 2.0) and a catalogue with clearly ranked
    popularities, the top-ranked service should receive far more requests
    than the bottom-ranked service over a large sample.
    """

    def setUp(self):
        # 4 services with distinct popularities to produce clear Zipf ranking
        self.catalogue = {
            i: Service(service_id=i, size=1,
                       popularity=[0.9, 0.6, 0.3, 0.1][i],
                       criticality="standard", revenue=5.0)
            for i in range(4)
        }
        leaves    = _make_leaf_nodes(2)
        self.gen  = RequestGenerator(catalogue=self.catalogue,
                                     leaf_nodes=leaves,
                                     seed=0, model="poisson_zipf",
                                     zipf_alpha=1.5)
        self.reqs = self.gen.generate(4000)

    def test_rank1_service_most_requested(self):
        counts = {i: 0 for i in range(4)}
        for req in self.reqs:
            counts[req.service_id] += 1
        # Service 0 has highest popularity -> rank 1 -> highest probability
        most_popular = max(counts, key=counts.get)
        self.assertEqual(most_popular, 0)

    def test_rank4_service_least_requested(self):
        counts = {i: 0 for i in range(4)}
        for req in self.reqs:
            counts[req.service_id] += 1
        least_popular = min(counts, key=counts.get)
        self.assertEqual(least_popular, 3)

    def test_head_over_tail_ratio_matches_zipf(self):
        """
        count(rank1) / count(rank4) should approximate
        (4^alpha) / (1^alpha) = 4^alpha.
        """
        counts = {i: 0 for i in range(4)}
        for req in self.reqs:
            counts[req.service_id] += 1
        ratio_observed  = counts[0] / max(counts[3], 1)
        ratio_expected  = (4 ** 1.5) / (1 ** 1.5)   # ~8.0
        # Allow 40% tolerance (sampling variation over 4000 requests)
        self.assertGreater(ratio_observed, ratio_expected * 0.6)

    def test_all_services_receive_at_least_one_request(self):
        """Even the tail service should appear at least once in 4000 requests."""
        counts = {i: 0 for i in range(4)}
        for req in self.reqs:
            counts[req.service_id] += 1
        for svc_id, cnt in counts.items():
            self.assertGreater(cnt, 0,
                msg=f"Service {svc_id} received 0 requests")

    def test_zipf_adapts_to_popularity_drift(self):
        """
        After reversing popularities, the previously least-requested service
        should become the most requested.
        """
        # Reverse popularities so the old tail (service 3) becomes rank 1
        self.catalogue[0].popularity = 0.1
        self.catalogue[3].popularity = 0.9
        new_reqs = self.gen.generate(2000)
        counts   = {i: 0 for i in range(4)}
        for req in new_reqs:
            counts[req.service_id] += 1
        most_popular = max(counts, key=counts.get)
        self.assertEqual(most_popular, 3)


# ===========================================================================
# Inspection helpers
# ===========================================================================

class TestRequestGeneratorHelpers(unittest.TestCase):

    def setUp(self):
        self.gen, self.cat, _ = _make_gen(model="poisson_zipf", seed=0)

    def test_get_zipf_distribution_keys_match_catalogue(self):
        dist = self.gen.get_zipf_distribution()
        self.assertEqual(set(dist.keys()), set(self.cat.keys()))

    def test_get_zipf_distribution_sums_to_one(self):
        dist = self.gen.get_zipf_distribution()
        self.assertAlmostEqual(sum(dist.values()), 1.0, places=10)

    def test_expected_arrival_rate_poisson(self):
        self.assertAlmostEqual(
            self.gen.expected_arrival_rate(),
            self.gen.arrival_rate,
        )

    def test_expected_arrival_rate_uniform(self):
        gen, _, _ = _make_gen(model="uniform")
        self.assertAlmostEqual(gen.expected_arrival_rate(), 1.0 / 3.0, places=5)

    def test_repr_contains_model(self):
        self.assertIn("poisson_zipf", repr(self.gen))

    def test_repr_contains_service_count(self):
        self.assertIn(str(len(self.cat)), repr(self.gen))


# ===========================================================================
# Model consistency: both models produce valid Request objects
# ===========================================================================

class TestModelConsistency(unittest.TestCase):

    def _check_requests(self, reqs, catalogue, leaf_ids):
        valid_svc = set(catalogue.keys())
        for req in reqs:
            self.assertIn(req.service_id, valid_svc)
            self.assertIn(req.origin_node_id, leaf_ids)
            self.assertGreater(req.deadline_ms, 0)
            self.assertGreater(req.arrival_time, 0)
            self.assertEqual(req.status, "pending")

    def test_poisson_zipf_produces_valid_requests(self):
        gen, cat, leaves = _make_gen(model="poisson_zipf")
        leaf_ids = {n.node_id for n in leaves}
        self._check_requests(gen.generate(50), cat, leaf_ids)

    def test_uniform_produces_valid_requests(self):
        gen, cat, leaves = _make_gen(model="uniform")
        leaf_ids = {n.node_id for n in leaves}
        self._check_requests(gen.generate(50), cat, leaf_ids)

    def test_poisson_and_uniform_differ_in_timing(self):
        """Poisson and uniform should produce different arrival-time sequences."""
        cat    = _make_catalogue(seed=77)
        leaves = _make_leaf_nodes(4)
        gen_p  = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                                  seed=77, model="poisson_zipf")
        gen_u  = RequestGenerator(catalogue=cat, leaf_nodes=leaves,
                                  seed=77, model="uniform")
        times_p = [r.arrival_time for r in gen_p.generate(30)]
        times_u = [r.arrival_time for r in gen_u.generate(30)]
        self.assertNotEqual(times_p, times_u)


if __name__ == "__main__":
    unittest.main()

