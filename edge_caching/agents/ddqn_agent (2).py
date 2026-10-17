"""
ddqn_agent.py
-------------
Double Deep Q-Network (DDQN) Agent for centralised edge-cache provisioning.

Architecture
------------
  Online Network  : selects the greedy action   argmax_a Q_online(s, a)
  Target Network  : evaluates Q-value of that action (stable regression target)

  DDQN Bellman target:
      y = R + gamma * Q_target(s', argmax_a' Q_online(s', a'))

  This decouples action selection from value estimation, eliminating the
  maximisation bias that standard DQN suffers from in over-estimation.

State Vector  (per (service, node) pair  -  10 elements)
---------------------------------------------------------
  idx  field                  range   meaning
  ---  -----                  -----   -------
  0    size_norm              [0,1]   service.size / MAX_SERVICE_SIZE_MB (4500 MB)
  1    popularity             [0,1]   EMA-updated observed demand fraction
  2    omega_norm             [0,1]   criticality weight / 3
  3    revenue_norm           [0,1]   revenue / 10
  4    forwarded_rate         [0,1]   cache-miss forwards this interval (normalised)
  5    executed_rate          [0,1]   local executions this interval (normalised)
  6    cache_utilization      [0,1]   node cache occupancy fraction
  7    sla_violation_rate     [0,1]   deadline violations / total demand
  8    is_cached              {0,1}   1 if this service is already in this node's cache
  9    slack_norm             [0,1]   min(free_capacity / service.size, 1.0)
                                      1.0 = INSERT fits without eviction
                                      <1.0 = eviction required to make room

Action Space  (per (service, node) pair)
-----------------------------------------
  0 = INSERT   add the service to this node's cache
  1 = DELETE   evict the service from this node's cache
  2 = KEEP     no change  (default when uncertain)

ReplayBuffer
------------
  Capacity   : MEMORY_SIZE transitions (FIFO ring-buffer, deque)
  Sampling   : uniform random without replacement
  Ready flag : is_ready() must return True before any gradient update
               (enforced by WARMUP_STEPS > BATCH_SIZE)

Training guards
---------------
  WARMUP_STEPS  : buffer must hold at least this many transitions before learn()
                  does anything.  Prevents fitting to a biased initial batch.
  LEARN_EVERY   : gradient update runs only every N stored transitions
                  per provision cycle, reducing computational overhead.
  GRAD_CLIP_NORM: gradient L2 norm is clipped to this value before each
                  parameter update, preventing exploding gradients.

Target network synchronisation
-------------------------------
  Hard sync  (sync_target)     : full copy of online -> target weights.
                                 Called every TARGET_UPDATE episodes by the
                                 orchestrator.
  Soft sync  (soft_update_target): exponential moving average update:
                                 theta_target = TAU * theta_online
                                              + (1 - TAU) * theta_target
                                 Alternative to hard sync; more stable for
                                 continuous provisioning workflows.

Long-term profit objective
--------------------------
  The DDQN maximises discounted cumulative reward:
      G_t = sum_{k=0}^{inf} gamma^k * R_{t+k}

  With gamma=0.95, the effective horizon is ~1/(1-0.95) = 20 provisioning
  intervals.  Each interval's reward accounts for:
      + placement credit  per cache hit entering EDF queue  (omega-scaled)
      + revenue           per successfully delivered request (pop * rev * omega)
      - forwarding cost   per cache miss                    (hop * omega)
      - placement cost    per INSERT transfer
  served_count (not executed_count) is the primary placement signal so the
  agent is not penalised for EDF scheduling outcomes it cannot control.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Optional

from edge_caching.config import (
    GAMMA, LR, BATCH_SIZE, MEMORY_SIZE,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    HIDDEN_DIM, GRAD_CLIP_NORM, WARMUP_STEPS, LEARN_EVERY, TAU,
    LAMBDA_REVENUE, LAMBDA_FORWARD, LAMBDA_PLACEMENT,
    MAX_SERVICE_SIZE_MB,MAX_FORWARD_NORM, MAX_EXECUTE_NORM
)
from edge_caching.core.node import ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Three-layer fully-connected Q-network with LayerNorm for training stability.

    LayerNorm is applied after the first two linear layers.  It normalises the
    activations across the feature dimension within each sample, preventing
    internal covariate shift without requiring large batch sizes � important
    here because the effective batch is small (BATCH_SIZE=32) and the reward
    signal is sparse.

    Architecture
    ------------
    Input  (state_dim)  ->  Linear -> LayerNorm -> ReLU
                        ->  Linear -> LayerNorm -> ReLU
                        ->  Linear -> Q-values (action_dim)
    """

    def __init__(
        self,
        state_dim  : int,
        action_dim : int = 3,
        hidden_dim : int = HIDDEN_DIM,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer (uniform sampling).

    Stores transitions  (s, a, r, s', done)  as a FIFO deque.
    Once capacity is reached, the oldest transition is evicted automatically.

    Parameters
    ----------
    capacity : maximum number of stored transitions (MEMORY_SIZE from config)

    Methods
    -------
    push(state, action, reward, next_state, done)
        Store one transition.

    sample(batch_size) -> List[Tuple]
        Draw batch_size transitions uniformly at random without replacement.

    is_ready(warmup_steps) -> bool
        True when the buffer holds at least warmup_steps transitions.
        The orchestrator must check this before calling agent.learn().

    __len__
        Current number of stored transitions.
    """

    def __init__(self, capacity: int = MEMORY_SIZE):
        self.capacity = capacity
        self.buffer   : deque = deque(maxlen=capacity)

    def push(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool,
    ) -> None:
        """Store one (s, a, r, s', done) transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Uniform random sample without replacement.
        Caller is responsible to check is_ready() first.
        """
        return random.sample(self.buffer, batch_size)

    def is_ready(self, warmup_steps: int = WARMUP_STEPS) -> bool:
        """
        Returns True when enough transitions exist for a meaningful update.
        WARMUP_STEPS should be >= BATCH_SIZE so the first sample is possible.
        """
        return len(self.buffer) >= max(warmup_steps, BATCH_SIZE)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DDQN Agent
# ---------------------------------------------------------------------------

class DDQNAgent:
    """
    Centralised DDQN agent that drives the Slow Process (provisioning).

    One agent instance is shared across all (service, node) pairs.  The
    agent makes independent INSERT/DELETE/KEEP decisions for each pair
    during every provisioning interval.

    Parameters
    ----------
    state_dim    : length of the 8-element state vector
    num_nodes    : total edge nodes (used for bookkeeping)
    num_services : total services in catalogue (used for bookkeeping)
    """

    def __init__(
        self,
        state_dim    : int,
        num_nodes    : int,
        num_services : int,
    ):
        self.state_dim    = state_dim
        self.num_nodes    = num_nodes
        self.num_services = num_services
        self.action_dim   = 3       # INSERT / DELETE / KEEP

        # Online network: trained every LEARN_EVERY steps
        self.online_net = QNetwork(state_dim, self.action_dim)
        # Target network: updated every TARGET_UPDATE episodes (hard) or
        # via soft_update_target() at the end of each provision cycle
        self.target_net = QNetwork(state_dim, self.action_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # target never in train mode

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()   # Huber loss: less sensitive to outliers

        self.memory = ReplayBuffer(MEMORY_SIZE)

        self.epsilon        : float       = EPSILON_START
        self._total_steps   : int         = 0   # total transitions pushed
        self._learn_counter : int         = 0   # tracks LEARN_EVERY cadence

        # ---- Tracking ------------------------------------------------
        self.loss_history   : List[float] = []
        self.reward_history : List[float] = []

    # ==================================================================
    # Action selection
    # ==================================================================

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.

        During exploration (epsilon high) a random action is chosen to
        discover novel (service, node) placements.  During exploitation
        the online network's argmax Q-value is used.

        The action KEEP (2) is treated as the safe default � the agent
        must actively learn when INSERT or DELETE is worthwhile.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ==================================================================
    # Transition storage
    # ==================================================================

    def store_transition(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool = False,
    ) -> None:
        """
        Push one transition into the replay buffer and increment the
        step counter.  done=True should be passed at the end of each
        episode to correctly zero-out the Bellman bootstrap.
        """
        self.memory.push(state, action, reward, next_state, done)
        self._total_steps += 1

    # ==================================================================
    # Learning  (DDQN Bellman update)
    # ==================================================================

    def learn(self) -> float:
        """
        Run one gradient update step using a uniformly sampled mini-batch.

        Skips if:
        - Buffer not yet warmed up (< WARMUP_STEPS transitions)
        - Step counter not aligned to LEARN_EVERY cadence

        Returns
        -------
        loss_value : float  (Huber loss on the Bellman residual, 0 if skipped)

        DDQN Bellman target
        -------------------
        y_i = r_i + gamma * Q_target(s'_i, argmax_{a'} Q_online(s'_i, a'))

        The online network selects the best next action; the target network
        evaluates its Q-value.  This two-step evaluation eliminates the
        maximisation bias present in vanilla DQN.

        Gradient clipping
        -----------------
        After loss.backward(), gradients are clipped to GRAD_CLIP_NORM
        (L2 norm).  This prevents the large gradient spikes that can occur
        when newly discovered transitions carry very different reward
        magnitudes, keeping training stable throughout long runs.
        """
        if not self.memory.is_ready():
            return 0.0

        self._learn_counter += 1
        if self._learn_counter % LEARN_EVERY != 0:
            return 0.0

        batch      = self.memory.sample(BATCH_SIZE)
        states     = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions    = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
        rewards    = torch.FloatTensor([b[2] for b in batch])
        nxt_states = torch.FloatTensor(np.array([b[3] for b in batch]))
        dones      = torch.FloatTensor([b[4] for b in batch])

        # ---- Current Q-values  (online, differentiable) ---------------
        current_q = self.online_net(states).gather(1, actions).squeeze(1)

        # ---- DDQN target  (no gradient) --------------------------------
        with torch.no_grad():
            # Online selects action
            best_next_actions = self.online_net(nxt_states).argmax(dim=1, keepdim=True)
            # Target evaluates that action
            target_q_vals     = self.target_net(nxt_states).gather(
                1, best_next_actions
            ).squeeze(1)
            # Bellman target: zero-out bootstrap when episode ends
            target_q = rewards + GAMMA * target_q_vals * (1.0 - dones)

        # ---- Huber loss + gradient clipping ----------------------------
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    # ==================================================================
    # Target network synchronisation
    # ==================================================================

    def sync_target(self) -> None:
        """
        Hard synchronisation: copy all online weights to target network.

        Called by the orchestrator every TARGET_UPDATE episodes.
        Provides a completely stable regression target for TARGET_UPDATE
        episodes before the next hard reset.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target(self) -> None:
        """
        Soft (Polyak) synchronisation using TAU from config:

            theta_target <- TAU * theta_online + (1 - TAU) * theta_target

        With TAU=0.005, the target network tracks the online network very
        slowly, giving a quasi-stationary regression target while still
        continuously incorporating new information.  This is more suitable
        than hard sync when provision cycles are very frequent.
        """
        for t_param, o_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters(),
        ):
            t_param.data.copy_(
                TAU * o_param.data + (1.0 - TAU) * t_param.data
            )

    # ==================================================================
    # Epsilon decay
    # ==================================================================

    def decay_epsilon(self) -> None:
        """
        Multiplicative epsilon decay, called once per episode.
        Transitions the agent from exploration-dominated to
        exploitation-dominated behaviour over training.
        """
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    # ==================================================================
    # Checkpoint save / load
    # ==================================================================

    def save_checkpoint(self, path: str) -> None:
        """
        Persist the full agent state so training can be resumed.

        Saves: online_net, target_net, optimizer, epsilon, step counters.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "online_net"    : self.online_net.state_dict(),
            "target_net"    : self.target_net.state_dict(),
            "optimizer"     : self.optimizer.state_dict(),
            "epsilon"       : self.epsilon,
            "total_steps"   : self._total_steps,
            "learn_counter" : self._learn_counter,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """
        Restore agent state from a saved checkpoint.
        Automatically maps tensors to CPU.
        """
        ckpt = torch.load(path, map_location="cpu")
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon        = ckpt["epsilon"]
        self._total_steps   = ckpt["total_steps"]
        self._learn_counter = ckpt["learn_counter"]
        self.target_net.eval()

    # ==================================================================
    # State vector builder
    # ==================================================================

    @staticmethod
    def build_state_vector(
        service     : Service,
        stats       : dict,
        max_forward : float = MAX_FORWARD_NORM,  #100.0,
        max_execute : float = MAX_EXECUTE_NORM,  #100.0,
    ) -> np.ndarray:
        """
        Build the 10-element state vector for one (service, node) pair.

        Vector layout
        -------------
        idx  field                normalisation
        0    size_norm            service.size / MAX_SERVICE_SIZE_MB (4500 MB)
                                  DNS_Resolver(2MB)=0.0004, LiveVideo_4K(2800MB)=0.622
        1    popularity           already [0,1] (EMA-updated each interval)
        2    omega                / 3           (max omega = 3)
        3    revenue              / 10          (max revenue = 10)
        4    forwarded_rate       / max_forward (cache-miss count this interval)
        5    executed_rate        / max_execute (cache-hit  count this interval)
        6    cache_utilization    already [0,1]
        7    sla_violation_rate   already [0,1] (violations / demand)
        8    is_cached            {0, 1}        1 if service is in this node's cache
        9    slack_norm           [0, 1]        min(free_capacity / service.size, 1.0)
                                               1.0 = INSERT fits without any eviction
                                               <1.0 = eviction needed to make room
        """
        svc_id     = service.service_id
        fwd        = stats["forwarded_count"].get(svc_id, 0) / max(max_forward, 1)
        exe        = stats["executed_count"].get(svc_id, 0)  / max(max_execute, 1)
        is_cached  = 1.0 if svc_id in stats.get("cached_services", set()) else 0.0
        slack_norm = min(stats.get("free_capacity", 0.0) / max(service.size, 1.0), 1.0)

        return np.array([
            service.size             / MAX_SERVICE_SIZE_MB,
            service.popularity,
            service.omega            / 3.0,
            service.revenue          / 10.0,
            min(fwd, 1.0),
            min(exe, 1.0),
            stats["cache_utilization"],
            service.sla_violation_rate,
            is_cached,
            slack_norm,
        ], dtype=np.float32)

    # ==================================================================
    # Reward computation
    # ==================================================================

    @staticmethod
    def compute_reward(
        service         : Service,
        delta_served    : int,
        delta_executed  : int,
        delta_forwarded : int,
        placement_cost  : float,
        r_max           : float = 50.0,
    ) -> float:
        """
        Compute the normalised single-step reward for one (service, node, action).

        Formula  (all terms in [currency], dimensionally consistent)
        -------
        R = lambda_r * rev * omega              * delta_served     (placement credit)
          + lambda_r * pop * rev * omega        * delta_executed   (delivery revenue)
          - lambda_f * rev * omega              * delta_forwarded  (miss penalty)
          - lambda_p * placement_cost                              (INSERT transfer cost)

        Clipped to [-1, +1] by dividing by r_max.

        Unit analysis
        -------------
        serve_term   : [1] * [currency/req] * [1] * [req] = [currency] ✓
        rev_term     : [1] * [1] * [currency/req] * [1] * [req] = [currency] ✓
        forward_term : [1] * [currency/req] * [1] * [req] = [currency] ✓
        place_term   : [1] * [currency] = [currency] ✓

        serve_term  (+) : LAMBDA_REVENUE * revenue * omega * delta_served
            Fires when _is_service_cached() returns True, BEFORE EDF scheduling.
            Rewards correct placement even when the EDF scheduler later drops
            the request due to a tight deadline on a congested node.
            Revenue is included so the term is in [currency] and correctly
            weights high-value services over low-value ones.

        rev_term    (+) : LAMBDA_REVENUE * popularity * revenue * omega * delta_executed
            Fires only on successful delivery. Captures actual business revenue
            earned. Popularity weighting means a frequently requested service
            earns more reward per execution than a rarely requested one with
            the same base revenue.

        forward_term (-) : LAMBDA_FORWARD * revenue * omega * delta_forwarded
            Economic value at risk per forwarded request.
            revenue * omega = value of the request that was missed locally.
            Physical latency is already handled by the simulation — each hop
            subtracts from the request deadline in subtract_hop_latency(),
            which kills mission_critical requests (5-40ms budget) before they
            reach cloud. Repeating latency in the reward would double-count
            the physical consequence. No hop_cost needed here.

        place_term  (-) : LAMBDA_PLACEMENT * placement_cost
            One-time Cost-of-Delay on INSERT, amortised over future serve_term
            and rev_term across the service lifetime in cache.
        """
        serve_term   = LAMBDA_REVENUE   * service.revenue * service.omega * delta_served
        rev_term     = LAMBDA_REVENUE   * service.popularity * service.revenue * service.omega * delta_executed
        forward_term = LAMBDA_FORWARD   * service.revenue * service.omega * delta_forwarded
        place_term   = LAMBDA_PLACEMENT * placement_cost

        raw = serve_term + rev_term - forward_term - place_term
        return float(np.clip(raw / r_max, -1.0, 1.0))
