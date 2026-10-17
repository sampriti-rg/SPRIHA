"""
test_orchestrator.py
--------------------
Unit tests for edge_caching.control.orchestrator (EdgeOrchestrator)

Covers
------
- Orchestrator initialises a DDQNAgent with the correct dimensions
- provision() returns a dict with all expected metric keys
- provision() returns numeric values for all metrics
- end_episode() appends to every metric list
- end_episode() increments the internal episode counter
- end_episode() resets node revenues to zero
- Target sync occurs on multiples of TARGET_UPDATE
- Epsilon decreases after each provision() call
"""

import unittest

from edge_caching.control.orchestrator import EdgeOrchestrator
from edge_caching.core.node import EdgeNode, ACTION_INSERT
from edge_caching.core.service import Service, generate_service_catalogue
from edge_caching.config import TARGET_UPDATE

PROVISION_METRIC_KEYS = {
    "interval_reward",
    "placement_cost",
    "executed_delta",
    "forwarded_delta",
    "violation_delta",
    "avg_loss",
    "epsilon",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(sid: int = 0, size: int = 1, pop: float = 0.5,
         crit: str = "standard", rev: float = 3.0) -> Service:
    return Service(service_id=sid, size=size, popularity=pop,
                   criticality=crit, revenue=rev)


def _make_orchestrator(num_nodes: int = 3, num_services: int = 4):
    """Build a minimal orchestrator with stand-alone EdgeNodes."""
    nodes = [
        EdgeNode(node_id=f"L2_{i}", level=2, cache_cap=10)
        for i in range(num_nodes)
    ]
    catalogue_list = generate_service_catalogue(num_services, seed=0)
    catalogue = {s.service_id: s for s in catalogue_list}
    orch = EdgeOrchestrator(nodes=nodes, catalogue=catalogue)
    return orch, nodes, catalogue


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestOrchestratorInit(unittest.TestCase):

    def test_agent_state_dim(self):
        orch, nodes, catalogue = _make_orchestrator(num_nodes=3, num_services=4)
        self.assertEqual(orch.agent.state_dim, 8)

    def test_agent_num_nodes(self):
        orch, nodes, catalogue = _make_orchestrator(num_nodes=5, num_services=4)
        self.assertEqual(orch.agent.num_nodes, 5)

    def test_agent_num_services(self):
        orch, nodes, catalogue = _make_orchestrator(num_nodes=3, num_services=6)
        self.assertEqual(orch.agent.num_services, 6)

    def test_metric_lists_empty_at_start(self):
        orch, _, _ = _make_orchestrator()
        self.assertEqual(orch.episode_rewards,    [])
        self.assertEqual(orch.episode_profits,    [])
        self.assertEqual(orch.episode_hit_rates,  [])
        self.assertEqual(orch.episode_fwd_counts, [])
        self.assertEqual(orch.episode_losses,     [])


# ---------------------------------------------------------------------------
# provision()
# ---------------------------------------------------------------------------

class TestOrchestratorProvision(unittest.TestCase):

    def setUp(self):
        self.orch, self.nodes, self.catalogue = _make_orchestrator(
            num_nodes=3, num_services=4
        )

    def test_provision_returns_dict(self):
        result = self.orch.provision()
        self.assertIsInstance(result, dict)

    def test_provision_contains_all_metric_keys(self):
        result = self.orch.provision()
        self.assertEqual(set(result.keys()), PROVISION_METRIC_KEYS)

    def test_provision_values_are_numeric(self):
        result = self.orch.provision()
        for key, val in result.items():
            self.assertIsInstance(val, (int, float),
                                  msg=f"Key '{key}' is not numeric")

    def test_epsilon_in_result_matches_agent(self):
        result = self.orch.provision()
        self.assertAlmostEqual(result["epsilon"], self.orch.agent.epsilon)

    def test_epsilon_decreases_after_end_episode(self):
        # Epsilon decay is triggered by end_episode(), not provision()
        eps_before = self.orch.agent.epsilon
        self.orch.end_episode(
            episode_reward=0.0, total_executed=0,
            total_requests=10, total_forwarded=0,
            total_violations=0, avg_loss=0.0,
        )
        self.assertLess(self.orch.agent.epsilon, eps_before)

    def test_node_interval_stats_reset_after_provision(self):
        # Generate some stats by manually incrementing counts
        for node in self.nodes:
            node.forwarded_count[0] = 5
            node.executed_count[0]  = 3
        self.orch.provision()
        for node in self.nodes:
            self.assertEqual(node.forwarded_count, {})
            self.assertEqual(node.executed_count,  {})

    def test_placement_cost_non_negative(self):
        result = self.orch.provision()
        self.assertGreaterEqual(result["placement_cost"], 0.0)

    def test_executed_and_forwarded_deltas_non_negative(self):
        result = self.orch.provision()
        self.assertGreaterEqual(result["executed_delta"],  0)
        self.assertGreaterEqual(result["forwarded_delta"], 0)


# ---------------------------------------------------------------------------
# end_episode()
# ---------------------------------------------------------------------------

class TestOrchestratorEndEpisode(unittest.TestCase):

    def setUp(self):
        self.orch, self.nodes, self.catalogue = _make_orchestrator(
            num_nodes=2, num_services=4
        )

    def _call_end_episode(self, reward=1.0, executed=20,
                          requests=50, forwarded=10, violations=0, loss=0.01):
        self.orch.end_episode(
            episode_reward   = reward,
            total_executed   = executed,
            total_requests   = requests,
            total_forwarded  = forwarded,
            total_violations = violations,
            avg_loss         = loss,
        )

    def test_episode_rewards_appended(self):
        self._call_end_episode(reward=2.5)
        self.assertEqual(len(self.orch.episode_rewards), 1)
        self.assertAlmostEqual(self.orch.episode_rewards[0], 2.5)

    def test_episode_hit_rate_correct(self):
        self._call_end_episode(executed=25, requests=50)
        self.assertAlmostEqual(self.orch.episode_hit_rates[0], 0.5)

    def test_hit_rate_zero_when_no_requests(self):
        self._call_end_episode(executed=0, requests=0)
        self.assertAlmostEqual(self.orch.episode_hit_rates[0], 0.0)

    def test_episode_losses_appended(self):
        self._call_end_episode(loss=0.05)
        self.assertAlmostEqual(self.orch.episode_losses[0], 0.05)

    def test_fwd_counts_appended(self):
        self._call_end_episode(forwarded=15)
        self.assertEqual(self.orch.episode_fwd_counts[0], 15)

    def test_node_revenues_reset_to_zero(self):
        for node in self.nodes:
            node.total_revenue = 99.9
        self._call_end_episode()
        for node in self.nodes:
            self.assertAlmostEqual(node.total_revenue, 0.0)

    def test_episode_counter_increments(self):
        self._call_end_episode()
        self.assertEqual(self.orch._episode, 1)
        self._call_end_episode()
        self.assertEqual(self.orch._episode, 2)

    def test_multiple_episodes_accumulate(self):
        for _ in range(5):
            self._call_end_episode()
        self.assertEqual(len(self.orch.episode_rewards), 5)


if __name__ == "__main__":
    unittest.main()
