"""
ddqn_agent.py
-------------
Double Deep Q-Network (DDQN) Agent for centralised edge-cache provisioning with GPU Acceleration.

This module implements a DDQN reinforcement learning agent with automatic GPU detection
and acceleration for significantly faster training on CUDA-enabled systems.

GPU Acceleration Features:
- Automatic CUDA GPU detection and utilization
- Fallback to CPU if GPU not available  
- Device-aware tensor operations throughout
- Optimized batch processing for provisioning decisions
- Memory-efficient replay buffer sampling

Architecture:
- Online Network: selects greedy action argmax_a Q_online(s,a)
- Target Network: evaluates Q-value (stable regression target)
- DDQN Bellman target: y = R + gamma * Q_target(s', argmax_a' Q_online(s', a'))
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Tuple, Optional
from edge_caching.agents.prioritized_replay_buffer import PrioritizedReplayBuffer


from edge_caching.config import (
    GAMMA, LR, BATCH_SIZE, MEMORY_SIZE,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    HIDDEN_DIM, GRAD_CLIP_NORM, WARMUP_STEPS, LEARN_EVERY, TAU,
    LAMBDA_REVENUE, LAMBDA_FORWARD, LAMBDA_PLACEMENT,
    MAX_SERVICE_SIZE_MB, MAX_FORWARD_NORM, MAX_EXECUTE_NORM
)
from edge_caching.core.node import ACTION_INSERT, ACTION_DELETE, ACTION_KEEP
from edge_caching.core.service import Service


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """
    Three-layer fully-connected Q-network with LayerNorm for training stability.
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
# Replay Buffer (kept for backward compatibility)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity experience replay buffer with optimized numpy-based sampling.
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
        """Store one transition."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Uniform random sample returning pre-stacked numpy arrays.
        """
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def is_ready(self, warmup_steps: int = WARMUP_STEPS) -> bool:
        return len(self.buffer) >= max(warmup_steps, BATCH_SIZE)

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DDQN Agent with GPU Acceleration
# ---------------------------------------------------------------------------

class DDQNAgent:
    """
    Centralised DDQN agent with automatic GPU acceleration and PER.
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
        self.action_dim   = 3

        # GPU Detection and Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("=" * 72)
        print("DDQN AGENT - COMPUTE DEVICE INFORMATION")
        print("-" * 72)
        if torch.cuda.is_available():
            print(f" GPU Acceleration: ENABLED")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  Expected speedup: 5-10x faster training")
        else:
            print(f" GPU Acceleration: DISABLED (using CPU)")
            print(f"  Training will be slower. Consider installing CUDA-enabled PyTorch:")
            print(f"  https://pytorch.org/get-started/locally/")
        print(f"  Active Device: {self.device}")
        print("=" * 72)
        print()

        # Networks on device
        self.online_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LR)
        self.loss_fn   = nn.SmoothL1Loss()

        # Prioritized Experience Replay
        self.memory = PrioritizedReplayBuffer(
            capacity=MEMORY_SIZE,
            alpha=0.6,
            beta_start=0.4,
            beta_frames=100000,
        )
        print("[DDQNAgent] Using Prioritized Experience Replay (PER)")

        self.epsilon        : float       = EPSILON_START
        self._total_steps   : int         = 0
        self._learn_counter : int         = 0

        self.loss_history   : List[float] = []
        self.reward_history : List[float] = []
        self.predicted_q_history: List[float] = []

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection (GPU-accelerated)."""
        state_t=torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values =self.online_net(state_t)
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = int(q_values.argmax(dim=1).item())

        #---Commented on 10/4/2026 ---The code is working fine comment to use
        # additional logging for predicted Q-values 
        #state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        #with torch.no_grad():
        #    q_values = self.online_net(state_t)
        #return int(q_values.argmax(dim=1).item())
        #---------------------------------------------------------------------#

        self.predicted_q_history.append(q_values[0,action].item())
        return action
    
    def reset_episode_tracking(self) -> None:
        self.predicted_q_history=[]

    def select_actions_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Vectorized batch action selection (GPU-accelerated).
        Significantly faster than loop-based selection.
        """
        batch_size = states.shape[0]
        actions = np.zeros(batch_size, dtype=np.int64)

        explore_mask = np.random.random(batch_size) < self.epsilon
        actions[explore_mask] = np.random.randint(0, self.action_dim, size=explore_mask.sum())

        exploit_mask = ~explore_mask
        if exploit_mask.any():
            states_t = torch.FloatTensor(states[exploit_mask]).to(self.device)
            with torch.no_grad():
                q_values = self.online_net(states_t)
                actions[exploit_mask] = q_values.argmax(dim=1).cpu().numpy()

        return actions

    def store_transition(
        self,
        state      : np.ndarray,
        action     : int,
        reward     : float,
        next_state : np.ndarray,
        done       : bool = False,
    ) -> None:
        """Push one transition into replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
        self._total_steps += 1

    def learn(self) -> float:
        """
        DDQN gradient update with PER support and mixed precision.
        """
        if not self.memory.is_ready():
            return 0.0

        self._learn_counter += 1
        if self._learn_counter % LEARN_EVERY != 0:
            return 0.0

        # PER sampling returns weights and indices
        states_np, actions_np, rewards_np, next_states_np, dones_np, weights, indices = \
            self.memory.sample(BATCH_SIZE)

        # Convert to tensors on device
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Use mixed precision if on CUDA
        if self.device.type == 'cuda':
            from torch.amp import autocast
            with autocast('cuda'):
                # Current Q-values
                current_q = self.online_net(states).gather(1, actions).squeeze(1)

                # DDQN target
                with torch.no_grad():
                    best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                    target_q_vals = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
                    target_q = rewards + GAMMA * target_q_vals * (1.0 - dones)

                # TD-errors for priority update (must be in FP32)
                td_errors = (current_q - target_q).detach().float().cpu().numpy()

                # Weighted loss (importance sampling)
                elementwise_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')
                loss = (weights_t * elementwise_loss).mean()
        else:
            # CPU path (no mixed precision)
            # Current Q-values
            current_q = self.online_net(states).gather(1, actions).squeeze(1)

            # DDQN target
            with torch.no_grad():
                best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
                target_q_vals = self.target_net(next_states).gather(1, best_next_actions).squeeze(1)
                target_q = rewards + GAMMA * target_q_vals * (1.0 - dones)

            # TD-errors for priority update
            td_errors = (current_q - target_q).detach().cpu().numpy()

            # Weighted loss (importance sampling)
            elementwise_loss = nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')
            loss = (weights_t * elementwise_loss).mean()

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        # Update priorities in buffer
        self.memory.update_priorities(indices, td_errors)

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def sync_target(self) -> None:
        """Hard synchronisation (GPU-aware)."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def soft_update_target(self) -> None:
        """Soft Polyak synchronisation (GPU-aware)."""
        for t_param, o_param in zip(
            self.target_net.parameters(),
            self.online_net.parameters(),
        ):
            t_param.data.copy_(TAU * o_param.data + (1.0 - TAU) * t_param.data)

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

    def save_checkpoint(self, path: str) -> None:
        """Save full training checkpoint (device-agnostic)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Extract replay buffer data from SumTree structure
        buffer_data = []
        if hasattr(self.memory, 'tree') and hasattr(self.memory.tree, 'data'):
            # PrioritizedReplayBuffer with SumTree
            for i in range(self.memory.tree.n_entries):
                if self.memory.tree.data[i] is not None:
                    buffer_data.append(self.memory.tree.data[i])
        
        checkpoint = {
            "online_net_state"  : self.online_net.state_dict(),
            "target_net_state"  : self.target_net.state_dict(),
            "optimizer_state"   : self.optimizer.state_dict(),
            "epsilon"           : self.epsilon,
            "total_steps"       : self._total_steps,
            "learn_counter"     : self._learn_counter,
            "loss_history"      : self.loss_history,
            "reward_history"    : self.reward_history,
            "device"            : str(self.device),
            "replay_buffer"     : buffer_data,  # Save extracted buffer data
        }

        torch.save(checkpoint, path)
        print(f"[DDQNAgent] Checkpoint saved to {path}")
        print(f"[DDQNAgent] Replay buffer: {len(buffer_data)} transitions saved")

    def load_checkpoint(self, path: str, inference_only: bool = False) -> None:
        """Load checkpoint with automatic device mapping."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.online_net.load_state_dict(checkpoint["online_net_state"])
        self.target_net.load_state_dict(checkpoint["target_net_state"])

        if not inference_only:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.epsilon        = checkpoint["epsilon"]
            self._total_steps   = checkpoint["total_steps"]
            self._learn_counter = checkpoint["learn_counter"]
            self.loss_history   = checkpoint.get("loss_history", [])
            self.reward_history = checkpoint.get("reward_history", [])
            
            # Load replay buffer if available
            if "replay_buffer" in checkpoint and checkpoint["replay_buffer"]:
                buffer_data = checkpoint["replay_buffer"]
                
                # For PrioritizedReplayBuffer, use push method
                if hasattr(self.memory, 'push'):
                    for transition in buffer_data:
                        if transition is not None and len(transition) == 5:
                            state, action, reward, next_state, done = transition
                            self.memory.push(state, action, reward, next_state, done)
                else:
                    # Fallback for regular ReplayBuffer
                    if hasattr(self.memory, 'clear'):
                        self.memory.clear()
                    for transition in buffer_data:
                        if transition is not None:
                            self.memory.append(transition)
                
                print(f"[DDQNAgent] Replay buffer loaded: {len(self.memory)} transitions")
            else:
                print(f"[DDQNAgent] ⚠ WARNING: No replay buffer found in checkpoint!")

        loaded_device = checkpoint.get("device", "unknown")
        print(f"[DDQNAgent] Checkpoint loaded from {path}")
        print(f"[DDQNAgent] Original device: {loaded_device}, Current device: {self.device}")
    
    def to_device(self, device: str = 'cuda:0') -> None:
        """
        Explicitly move agent to specified device.
        Use this after loading a checkpoint to force GPU placement.
        
        Args:
            device: Target device ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        target_device = torch.device(device)
        print(f"[DDQNAgent] Moving to {target_device}...")
        
        # Update device
        self.device = target_device
        
        # Move networks
        self.online_net = self.online_net.to(target_device)
        self.target_net = self.target_net.to(target_device)
        
        # Move optimizer state
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(target_device)
        
        print(f"[DDQNAgent] ✓ Networks moved to {target_device}")
        print(f"[DDQNAgent] ✓ Online net on: {next(self.online_net.parameters()).device}")
        print(f"[DDQNAgent] ✓ Target net on: {next(self.target_net.parameters()).device}")

    def save_weights_only(self, path: str) -> None:
        """Export online network weights only."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.online_net.state_dict(), path)
        print(f"[DDQNAgent] Weights-only export saved to {path}")

    def load_weights_only(self, path: str) -> None:
        """Load weights for inference."""
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.online_net.eval()
        print(f"[DDQNAgent] Weights loaded from {path} (inference mode)")

    def get_memory_usage(self) -> dict:
        """Report GPU/CPU memory usage."""
        info = {
            "device": str(self.device),
            "buffer_size": len(self.memory),
        }

        if torch.cuda.is_available():
            info["allocated_mb"] = torch.cuda.memory_allocated() / 1e6
            info["cached_mb"] = torch.cuda.memory_reserved() / 1e6

        return info

    # ==================================================================
    # State vector builder
    # ==================================================================

    @staticmethod
    def build_state_vector(
        service     : Service,
        stats       : dict,
        max_forward : float = MAX_FORWARD_NORM,
        max_execute : float = MAX_EXECUTE_NORM,
    ) -> np.ndarray:
        """Build the 10-element state vector for one (service, node) pair."""
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
    ) -> float:
        """
        Improved reward function with balanced penalties.

        Changes:
        1. Reduced violation penalty (encourage exploration)
        2. Higher reward for local execution (favor edge caching)
        3. Lower placement cost penalty (encourage cache updates)
        """

        # Revenue from successfully served requests
        revenue = service.revenue * delta_served

        # Compute violations (served but not executed = dropped)
        violations = max(0, delta_served - delta_executed)

        # NORMALIZED rewards to [-1, 1] range
        violation_penalty = -0.5 * (violations / max(delta_served, 1))  # Ratio instead of absolute

        # KEY CHANGE 2: Bonus for local execution
        local_executed = delta_executed - delta_forwarded
        local_bonus = 0.3 * (local_executed / max(delta_executed, 1))  # Ratio

        # KEY CHANGE 3: Reduced forwarding penalty
        # Old: -1.0, New: -0.3 (encourage trying vs. dropping)
        forward_penalty = -0.2 * (delta_forwarded / max(delta_executed, 1))  # Ratio

        # KEY CHANGE 4: Reduced placement cost weight
        # Old: -1.0, New: -0.5 (encourage cache updates)
        placement_penalty = -0.01 * placement_cost  # Much smaller weight

        total_reward = revenue * (1.0 + violation_penalty + local_bonus + forward_penalty + placement_penalty)

        return total_reward
