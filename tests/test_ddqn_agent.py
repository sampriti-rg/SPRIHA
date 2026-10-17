"""
test_ddqn_agent.py
------------------
Unit tests for edge_caching.agents.ddqn_agent

Covers
------
QNetwork
  - forward pass output shape
  - output changes with different inputs

ReplayBuffer
  - push / len / sample
  - circular eviction when over capacity

DDQNAgent
  - greedy and random action selection paths
  - store_transition writes to memory
  - learn returns 0.0 when buffer is under-filled
  - learn returns a positive loss when buffer is sufficiently full
  - sync_target copies online weights to target
  - decay_epsilon reduces epsilon and respects EPSILON_MIN
  - build_state_vector shape and value normalisation
  - compute_reward range [-1, +1] and reward sign logic
"""

import unittest
import numpy as np
import torch

from edge_caching.agents.ddqn_agent import QNetwork, ReplayBuffer, DDQNAgent
from edge_caching.core.service import Service
from edge_caching.config import (
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    BATCH_SIZE, HIDDEN_DIM,
)

STATE_DIM = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(sid: int = 0, size: int = 1, pop: float = 0.8,
         crit: str = "standard", rev: float = 5.0) -> Service:
    return Service(service_id=sid, size=size, popularity=pop,
                   criticality=crit, revenue=rev)


def _dummy_state() -> np.ndarray:
    return np.zeros(STATE_DIM, dtype=np.float32)


def _dummy_stats(svc_id: int = 0) -> dict:
    return {
        "forwarded_count"  : {svc_id: 10},
        "executed_count"   : {svc_id: 5},
        "cache_utilization": 0.4,
    }


def _fill_buffer(agent: DDQNAgent, n: int = BATCH_SIZE + 5) -> None:
    """Push n random transitions and align learn counter so learn() fires."""
    from edge_caching.config import LEARN_EVERY, WARMUP_STEPS
    # Push at least WARMUP_STEPS transitions so is_ready() returns True
    n = max(n, WARMUP_STEPS + 1)
    for _ in range(n):
        s  = np.random.rand(STATE_DIM).astype(np.float32)
        a  = np.random.randint(0, 3)
        r  = float(np.random.uniform(-1, 1))
        ns = np.random.rand(STATE_DIM).astype(np.float32)
        agent.store_transition(s, a, r, ns, False)
    # Set counter so _learn_counter % LEARN_EVERY == 0 on next increment
    agent._learn_counter = LEARN_EVERY - 1


# ---------------------------------------------------------------------------
# QNetwork
# ---------------------------------------------------------------------------

class TestQNetwork(unittest.TestCase):

    def setUp(self):
        self.net = QNetwork(state_dim=STATE_DIM, action_dim=3)

    def test_output_shape(self):
        x = torch.zeros(1, STATE_DIM)
        out = self.net(x)
        self.assertEqual(out.shape, (1, 3))

    def test_batch_output_shape(self):
        x = torch.zeros(16, STATE_DIM)
        out = self.net(x)
        self.assertEqual(out.shape, (16, 3))

    def test_output_differs_for_different_inputs(self):
        net = QNetwork(state_dim=STATE_DIM)
        x1  = torch.ones(1, STATE_DIM)
        x2  = torch.zeros(1, STATE_DIM)
        self.assertFalse(torch.allclose(net(x1), net(x2)))

    def test_custom_hidden_dim(self):
        net = QNetwork(state_dim=STATE_DIM, action_dim=3, hidden_dim=64)
        x   = torch.zeros(1, STATE_DIM)
        self.assertEqual(net(x).shape, (1, 3))


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer(unittest.TestCase):

    def _push(self, buf: ReplayBuffer, n: int = 1) -> None:
        for _ in range(n):
            buf.push(
                np.zeros(STATE_DIM, dtype=np.float32),
                0, 0.0,
                np.zeros(STATE_DIM, dtype=np.float32),
                False,
            )

    def test_len_zero_initially(self):
        self.assertEqual(len(ReplayBuffer(capacity=100)), 0)

    def test_len_increases_on_push(self):
        buf = ReplayBuffer(capacity=100)
        self._push(buf, 5)
        self.assertEqual(len(buf), 5)

    def test_len_capped_at_capacity(self):
        buf = ReplayBuffer(capacity=10)
        self._push(buf, 20)
        self.assertEqual(len(buf), 10)

    def test_sample_returns_correct_batch_size(self):
        buf = ReplayBuffer(capacity=100)
        self._push(buf, 50)
        batch = buf.sample(10)
        self.assertEqual(len(batch), 10)

    def test_sample_tuple_has_five_elements(self):
        buf = ReplayBuffer(capacity=100)
        self._push(buf, 10)
        state, action, reward, next_state, done = buf.sample(1)[0]
        self.assertEqual(len(state), STATE_DIM)

    def test_old_entries_evicted_when_full(self):
        """After filling past capacity, oldest entries are gone (FIFO via deque)."""
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.push(
                np.array([float(i)] * STATE_DIM, dtype=np.float32),
                0, 0.0,
                np.zeros(STATE_DIM, dtype=np.float32),
                False,
            )
        states = [buf.buffer[j][0][0] for j in range(len(buf))]
        # Oldest two (0, 1) should have been evicted; newest three (2,3,4) remain
        self.assertNotIn(0.0, states)
        self.assertNotIn(1.0, states)


# ---------------------------------------------------------------------------
# DDQNAgent - setup
# ---------------------------------------------------------------------------

class TestDDQNAgentInit(unittest.TestCase):

    def setUp(self):
        self.agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=6, num_services=8)

    def test_epsilon_starts_at_configured_value(self):
        self.assertAlmostEqual(self.agent.epsilon, EPSILON_START)

    def test_memory_is_empty_initially(self):
        self.assertEqual(len(self.agent.memory), 0)

    def test_online_and_target_weights_equal_at_init(self):
        for p_online, p_target in zip(
            self.agent.online_net.parameters(),
            self.agent.target_net.parameters(),
        ):
            self.assertTrue(torch.allclose(p_online.data, p_target.data))


# ---------------------------------------------------------------------------
# DDQNAgent - action selection
# ---------------------------------------------------------------------------

class TestDDQNAgentActionSelection(unittest.TestCase):

    def setUp(self):
        self.agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=4, num_services=8)

    def test_action_in_valid_range(self):
        for _ in range(50):
            action = self.agent.select_action(_dummy_state())
            self.assertIn(action, [0, 1, 2])

    def test_greedy_action_deterministic_at_epsilon_zero(self):
        self.agent.epsilon = 0.0
        a1 = self.agent.select_action(_dummy_state())
        a2 = self.agent.select_action(_dummy_state())
        self.assertEqual(a1, a2)

    def test_random_action_selected_at_epsilon_one(self):
        self.agent.epsilon = 1.0
        actions = {self.agent.select_action(_dummy_state()) for _ in range(100)}
        # With epsilon=1.0 all three actions should appear eventually
        self.assertGreater(len(actions), 1)


# ---------------------------------------------------------------------------
# DDQNAgent - learning
# ---------------------------------------------------------------------------

class TestDDQNAgentLearning(unittest.TestCase):

    def setUp(self):
        self.agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=4, num_services=8)

    def test_learn_returns_zero_when_buffer_insufficient(self):
        loss = self.agent.learn()
        self.assertEqual(loss, 0.0)

    def test_learn_returns_positive_loss_when_ready(self):
        _fill_buffer(self.agent)
        loss = self.agent.learn()
        self.assertGreater(loss, 0.0)

    def test_loss_recorded_in_history(self):
        _fill_buffer(self.agent)
        self.agent.learn()
        self.assertEqual(len(self.agent.loss_history), 1)

    def test_store_transition_increases_memory(self):
        self.agent.store_transition(_dummy_state(), 0, 0.5, _dummy_state(), False)
        self.assertEqual(len(self.agent.memory), 1)

    def test_sync_target_copies_online_weights(self):
        # Mutate online net then sync
        with torch.no_grad():
            for p in self.agent.online_net.parameters():
                p.add_(torch.ones_like(p))
        self.agent.sync_target()
        for p_online, p_target in zip(
            self.agent.online_net.parameters(),
            self.agent.target_net.parameters(),
        ):
            self.assertTrue(torch.allclose(p_online.data, p_target.data))


# ---------------------------------------------------------------------------
# DDQNAgent - epsilon decay
# ---------------------------------------------------------------------------

class TestDDQNAgentEpsilonDecay(unittest.TestCase):

    def test_decay_reduces_epsilon(self):
        agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=2, num_services=4)
        before = agent.epsilon
        agent.decay_epsilon()
        self.assertLess(agent.epsilon, before)

    def test_epsilon_never_below_minimum(self):
        agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=2, num_services=4)
        for _ in range(5000):
            agent.decay_epsilon()
        self.assertGreaterEqual(agent.epsilon, EPSILON_MIN)

    def test_decay_formula_matches_config(self):
        agent = DDQNAgent(state_dim=STATE_DIM, num_nodes=2, num_services=4)
        before = agent.epsilon
        agent.decay_epsilon()
        expected = max(EPSILON_MIN, before * EPSILON_DECAY)
        self.assertAlmostEqual(agent.epsilon, expected, places=6)


# ---------------------------------------------------------------------------
# DDQNAgent - build_state_vector
# ---------------------------------------------------------------------------

class TestBuildStateVector(unittest.TestCase):

    def test_vector_length(self):
        vec = DDQNAgent.build_state_vector(_svc(), _dummy_stats())
        self.assertEqual(len(vec), STATE_DIM)

    def test_vector_dtype_float32(self):
        vec = DDQNAgent.build_state_vector(_svc(), _dummy_stats())
        self.assertEqual(vec.dtype, np.float32)

    def test_all_values_in_0_1(self):
        vec = DDQNAgent.build_state_vector(_svc(), _dummy_stats())
        self.assertTrue(np.all(vec >= 0.0))
        self.assertTrue(np.all(vec <= 1.0))

    def test_size_normalisation(self):
        svc = _svc(size=3)
        vec = DDQNAgent.build_state_vector(svc, _dummy_stats())
        self.assertAlmostEqual(float(vec[0]), 1.0)

    def test_zero_stats_gives_zero_rates(self):
        stats = {"forwarded_count": {}, "executed_count": {},
                 "cache_utilization": 0.0}
        vec = DDQNAgent.build_state_vector(_svc(), stats)
        self.assertAlmostEqual(float(vec[4]), 0.0)  # forwarding rate
        self.assertAlmostEqual(float(vec[5]), 0.0)  # execution rate

    def test_cache_utilization_reflected(self):
        stats = {"forwarded_count": {}, "executed_count": {},
                 "cache_utilization": 0.75}
        vec = DDQNAgent.build_state_vector(_svc(), stats)
        self.assertAlmostEqual(float(vec[6]), 0.75)


# ---------------------------------------------------------------------------
# DDQNAgent - compute_reward
# ---------------------------------------------------------------------------

class TestComputeReward(unittest.TestCase):

    def test_reward_clipped_to_minus_one_plus_one(self):
        svc = _svc(rev=10.0, pop=1.0, crit="mission_critical")
        r   = DDQNAgent.compute_reward(svc, delta_executed=1000,
                                       delta_forwarded=0,
                                       placement_cost=0.0, hop_cost=0.0)
        self.assertGreaterEqual(r, -1.0)
        self.assertLessEqual(r, 1.0)

    def test_positive_reward_for_executed_service(self):
        svc = _svc(rev=5.0, pop=0.8, crit="standard")
        r   = DDQNAgent.compute_reward(svc, delta_executed=5,
                                       delta_forwarded=0,
                                       placement_cost=0.0, hop_cost=0.0)
        self.assertGreater(r, 0.0)

    def test_negative_reward_for_high_cost(self):
        svc = _svc(rev=0.0, pop=0.1, crit="best_effort")
        r   = DDQNAgent.compute_reward(svc, delta_executed=0,
                                       delta_forwarded=100,
                                       placement_cost=50.0, hop_cost=1.0)
        self.assertLess(r, 0.0)

    def test_zero_deltas_and_zero_cost_gives_zero(self):
        svc = _svc()
        r   = DDQNAgent.compute_reward(svc, delta_executed=0,
                                       delta_forwarded=0,
                                       placement_cost=0.0, hop_cost=0.0)
        self.assertAlmostEqual(r, 0.0)

    def test_higher_criticality_yields_higher_positive_reward(self):
        svc_mc  = _svc(crit="mission_critical", rev=5.0, pop=0.8)
        svc_be  = _svc(crit="best_effort",      rev=5.0, pop=0.8)
        r_mc    = DDQNAgent.compute_reward(svc_mc, delta_executed=1,
                                           delta_forwarded=0,
                                           placement_cost=0.0, hop_cost=0.0)
        r_be    = DDQNAgent.compute_reward(svc_be, delta_executed=1,
                                           delta_forwarded=0,
                                           placement_cost=0.0, hop_cost=0.0)
        self.assertGreater(r_mc, r_be)


if __name__ == "__main__":
    unittest.main()
